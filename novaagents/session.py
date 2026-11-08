"""Process-local supervision for background delegation and serialized parent runs."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
from copy import deepcopy
from threading import Lock
from typing import TYPE_CHECKING, Any, Self, TypedDict
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.runnables import Runnable, RunnableConfig
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Command

_CURRENT_SESSION: contextvars.ContextVar[NovaSession | None] = contextvars.ContextVar("nova_session", default=None)
_TERMINAL = {"success", "error", "cancelled", "interrupted"}
_OPEN_THREADS: set[tuple[int, str]] = set()
_THREAD_OWNERSHIP = Lock()
_ERROR_NOTICE_LIMIT = 1000


class LocalTask(TypedDict):
    """Serializable status, never an execution handle."""

    task_id: str
    agent_name: str
    status: str
    result_ref: str
    error: str | None


def _session_for_config(config: RunnableConfig) -> NovaSession | None:
    session = _CURRENT_SESSION.get()
    configurable = config.get("configurable", {})
    if session and session._open and configurable.get("thread_id") == session.thread_id and "|" not in str(configurable.get("checkpoint_ns", "")):
        return session
    return None


def current_session(config: RunnableConfig | None = None) -> NovaSession:
    """Return the active invocation's session or an actionable error."""
    session = _session_for_config(config) if config is not None else _CURRENT_SESSION.get()
    if session is None or not session._open:
        msg = "Local background delegation requires an open NovaSession; use 'async with NovaSession(agent) as session: await session.ainvoke(...)'."
        raise RuntimeError(msg)
    return session


class NovaSession:
    """Own local tasks and serialize all parent graph/checkpoint execution.

    Args:
        agent: Compiled Nova graph. Use this session for all runs on its thread.
        thread_id: Conversation identifier; defaults to a fresh UUID.
        max_concurrency: Maximum number of running local children.

    Local tasks are not durable workers. Closing cancels queued and running work
    and awaits cleanup. Cancellation of tools with external side effects remains
    cooperative. Separate sessions must use separate thread IDs.
    """

    def __init__(self, agent: CompiledStateGraph, *, thread_id: str | None = None, max_concurrency: int = 4) -> None:
        """Initialize a session without starting work or opening graph resources."""
        if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
            msg = "max_concurrency must be a positive integer."
            raise ValueError(msg)
        self.agent = agent
        self.thread_id = thread_id or uuid4().hex
        self._limit = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self._open = False
        self._closed = False
        self._handles: dict[str, asyncio.Task[None]] = {}
        self._records: dict[str, LocalTask] = {}
        self._results: dict[str, str] = {}
        self._pending: dict[str, LocalTask] = {}
        self._events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        self._wake: asyncio.Task[None] | None = None
        self._failure: Exception | None = None
        self._paused = False
        self._state: dict[str, Any] = {}
        self._config: RunnableConfig = {"configurable": {"thread_id": self.thread_id}}
        self._invoke_options: dict[str, Any] = {}
        self._thread_key = (id(getattr(agent, "checkpointer", None) or agent), self.thread_id)

    async def __aenter__(self) -> Self:
        """Open once; a closed session cannot be reused."""
        if self._open or self._closed:
            msg = "NovaSession can only be opened once."
            raise RuntimeError(msg)
        with _THREAD_OWNERSHIP:
            if self._thread_key in _OPEN_THREADS:
                msg = "Another NovaSession already owns this graph/checkpointer thread. Use a separate thread_id."
                raise RuntimeError(msg)
            _OPEN_THREADS.add(self._thread_key)
        self._open = True
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Stop accepting work, cancel children, and await every owned task."""
        self._open = False
        self._closed = True
        if self._wake is not None:
            self._wake.cancel()
        for task_id, handle in self._handles.items():
            if not handle.done():
                self.cancel(task_id)
        try:
            await asyncio.gather(*self._handles.values(), *([self._wake] if self._wake else []), return_exceptions=True)
        finally:
            with _THREAD_OWNERSHIP:
                _OPEN_THREADS.discard(self._thread_key)
            self._events.put_nowait(None)

    async def ainvoke(self, inputs: dict[str, Any] | Command, config: RunnableConfig | None = None, **kwargs: Any) -> dict[str, Any]:
        """Invoke the parent exclusively; retain state even without a checkpointer.

        Args:
            inputs: Graph input including messages and optional pinned skills.
            config: Runnable config; its thread ID must match this session.
            **kwargs: Graph invocation options, such as runtime context.

        Returns:
            The parent graph's resulting state.
        """
        if not self._open:
            msg = "Open NovaSession with 'async with' before invoking it."
            raise RuntimeError(msg)
        config = config or {}
        configurable = config.get("configurable", {})
        if configurable.get("thread_id", self.thread_id) != self.thread_id:
            msg = "Invocation thread_id must match its NovaSession."
            raise ValueError(msg)
        async with self._lock:
            if not self._open:
                msg = "The session closed while this invocation was queued."
                raise RuntimeError(msg)
            self._config = {**config, "configurable": {**configurable, "thread_id": self.thread_id}}
            self._invoke_options = kwargs
            result = await self._invoke(inputs)
            self._failure = None
            return result

    async def _invoke(self, inputs: dict[str, Any] | Command) -> dict[str, Any]:
        token = _CURRENT_SESSION.set(self)
        try:
            if getattr(self.agent, "checkpointer", None) is None and isinstance(inputs, dict):
                inputs = {**self._state, **inputs, "messages": [*self._state.get("messages", []), *inputs.get("messages", [])]}
            # Output schemas omit private lifecycle metadata; read full values via streaming.
            result: dict[str, Any] = {}
            async for values in self.agent.astream(inputs, self._config, stream_mode="values", **self._invoke_options):
                result = values
                self._state = values
            self._state = result
            self._paused = bool(result.get("__interrupt__"))
            if getattr(self.agent, "checkpointer", None) is not None:
                snapshot = await self.agent.aget_state(self._config)
                self._paused = any(task.interrupts for task in snapshot.tasks)
                self._state = dict(snapshot.values)
                result = dict(snapshot.values)
                if self._paused:
                    result["__interrupt__"] = tuple(interrupt for task in snapshot.tasks for interrupt in task.interrupts)
            for task_id in result.get("local_delivered", []):
                self._pending.pop(task_id, None)
                self._results.pop(task_id, None)
            return result
        finally:
            _CURRENT_SESSION.reset(token)

    def start(
        self, runnable: Runnable, state: dict[str, Any], config: RunnableConfig, name: str, *, context: object = None, tool_call_id: str | None = None
    ) -> LocalTask:
        """Queue an isolated child and return its serializable status immediately."""
        if not self._open:
            msg = "The session is closing and cannot accept work."
            raise RuntimeError(msg)
        task_id = uuid4().hex
        record = LocalTask(task_id=task_id, agent_name=name, status="queued", result_ref=f"/async_tasks/{task_id}.json", error=None)
        self._records[task_id] = record
        child_config: RunnableConfig = {
            "configurable": {"thread_id": f"{self.thread_id}:child:{task_id}", "checkpoint_ns": "", "ls_agent_type": "subagent"},
            "callbacks": config.get("callbacks"),
            "tags": config.get("tags", []),
            "metadata": {
                **config.get("metadata", {}),
                "ls_agent_type": "subagent",
                "lc_subagent_tool_call_id": tool_call_id or task_id,
                "local_task_id": task_id,
            },
        }
        self._handles[task_id] = asyncio.create_task(
            self._run(task_id, runnable, deepcopy(state), child_config, context), context=contextvars.Context()
        )
        return record.copy()

    async def _run(self, task_id: str, runnable: Runnable, state: dict[str, Any], config: RunnableConfig, context: object) -> None:
        try:
            async with self._limit:
                self._records[task_id] = {**self._records[task_id], "status": "running"}
                result = await runnable.ainvoke(state, config, **({"context": context} if context is not None else {}))
                if result.get("__interrupt__"):
                    self._complete(task_id, "error", "Background child interrupted for approval; no side effects were resumed.")
                    return
                self._results[task_id] = self._result_payload(result)
                self._complete(task_id, "success")
        except asyncio.CancelledError:
            self._complete(task_id, "cancelled")
            raise
        except Exception as error:  # noqa: BLE001  # Child failure becomes observable task status; the supervisor stays alive.
            self._complete(task_id, "error", str(error))

    @staticmethod
    def _result_payload(result: dict[str, Any]) -> str:
        if "messages" not in result:
            msg = "CompiledSubAgent must return a state containing a 'messages' key."
            raise ValueError(msg)
        final = next((m for m in reversed(result["messages"]) if m.type == "ai" and m.content), None)
        payload = {"historical_data": True, "messages": [{"type": final.type, "data": final.model_dump()}] if final else []}
        structured = result.get("structured_response")
        if structured is not None:
            if hasattr(structured, "model_dump_json"):
                structured = json.loads(structured.model_dump_json())
            elif dataclasses.is_dataclass(structured) and not isinstance(structured, type):
                structured = dataclasses.asdict(structured)
            payload["structured_response"] = structured
        return json.dumps(payload, ensure_ascii=False)

    def _complete(self, task_id: str, status: str, error: str | None = None) -> None:
        if self._records[task_id]["status"] in _TERMINAL:
            return
        if status != "success":
            self._results[task_id] = json.dumps({"historical_data": True, "status": status, "error": error})
        record: LocalTask = {**self._records[task_id], "status": status, "error": error[:_ERROR_NOTICE_LIMIT] if error else None}
        self._records[task_id] = record
        self._pending[task_id] = record
        self._events.put_nowait({"type": "task_completed", **record})
        if self._open and (self._wake is None or self._wake.done()):
            self._wake = asyncio.create_task(self._resume(), context=contextvars.Context())

    async def _resume(self) -> None:
        try:
            async with self._lock:
                while self._pending and self._open and not self._paused:
                    pending = set(self._pending)
                    await self._invoke({"messages": []})
                    if pending <= self._pending.keys():
                        msg = "Parent did not accept background completions; use a graph from create_nova_agent."
                        raise RuntimeError(msg)  # noqa: TRY301  # Report supervision invariant failures through the same error event.
        except Exception as error:  # noqa: BLE001  # Surface unattended parent failures through both events and wait_until_idle.
            self._failure = error
            self._events.put_nowait({"type": "parent_error", "error": str(error)})

    def check(self, task_id: str) -> LocalTask:
        """Read a task in this session; cross-session identifiers are rejected."""
        if task_id not in self._records:
            msg = "Unknown local task in this session."
            raise ValueError(msg)
        return self._records[task_id].copy()

    async def wait(self, task_id: str) -> LocalTask:
        """Await one task without cancelling it if the waiter is cancelled."""
        self.check(task_id)
        await asyncio.shield(asyncio.gather(self._handles[task_id], return_exceptions=True))
        return self.check(task_id)

    def cancel(self, task_id: str) -> LocalTask:
        """Cancel queued or running work cooperatively."""
        record = self.check(task_id)
        if record["status"] not in _TERMINAL:
            self._handles[task_id].cancel()
            self._complete(task_id, "cancelled")
        return self.check(task_id)

    async def events(self) -> AsyncIterator[dict[str, object]]:
        """Yield immediate completion and parent-error events until session closure."""
        while (event := await self._events.get()) is not None:
            yield event

    async def wait_until_idle(self) -> None:
        """Wait for children and automatic parent resumptions; propagate parent errors."""
        while True:
            async with self._lock:
                handles = [h for h in self._handles.values() if not h.done()]
                if self._wake is not None and not self._wake.done():
                    handles.append(self._wake)
            if not handles:
                break
            await asyncio.gather(*handles, return_exceptions=True)
        if self._failure is not None:
            raise self._failure
