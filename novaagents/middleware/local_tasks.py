"""Local task tools, completion delivery, and remote task dispatch."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, PrivateStateAttr, ToolCallRequest
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.config import get_config

from novaagents.session import LocalTask, _session_for_config, current_session

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime
    from langgraph.types import Command

    from novaagents.backends.protocol import BackendProtocol


def _merge_tasks(left: dict[str, LocalTask], right: dict[str, LocalTask]) -> dict[str, LocalTask]:
    return {**left, **right}


class LocalTaskState(AgentState):
    """Private task records; execution resources belong exclusively to the session."""

    local_tasks: NotRequired[Annotated[dict[str, LocalTask], PrivateStateAttr, _merge_tasks]]
    local_delivered: NotRequired[Annotated[list[str], PrivateStateAttr]]


class LocalTaskMiddleware(AgentMiddleware[LocalTaskState]):
    """Deliver child completions only at graph boundaries and route local task tools.

    Args:
        start_tool: Background task tool built by existing subagent machinery.
        names: Local subagent names used for remote/local start routing.
        backend: Backend receiving completed task artifacts within parent execution.
        remote: Whether remote task middleware already registers the shared tools.
    """

    state_schema = LocalTaskState

    def __init__(self, start_tool: BaseTool, names: frozenset[str], backend: BackendProtocol, *, remote: bool = False) -> None:
        """Register local tools or share the remote task middleware's tool names."""
        self.start_tool = start_tool
        self.names = names
        self.backend = backend
        self.remote = remote
        self.local_tools = {t.name: t for t in self._tools()}
        self.local_tools[start_tool.name] = start_tool
        self.tools = [self.local_tools["wait_async_task"]] if remote else list(self.local_tools.values())

    def _tools(self) -> list[BaseTool]:
        @tool
        def check_async_task(task_id: str, runtime: ToolRuntime) -> str:
            """Read a local task status and result reference in this session."""
            session = _session_for_config(runtime.config)
            if session and task_id in session._records:
                return json.dumps(session.check(task_id))
            record = runtime.state.get("local_tasks", {}).get(task_id)
            if record is None:
                msg = "Unknown local task in this conversation."
                raise ValueError(msg)
            return json.dumps(record)

        @tool
        async def wait_async_task(task_id: str, runtime: ToolRuntime) -> str:
            """Wait cooperatively for a local background task and return its final status."""
            return json.dumps(await current_session(runtime.config).wait(task_id))

        @tool
        def cancel_async_task(task_id: str, runtime: ToolRuntime) -> str:
            """Cancel a queued or running local task cooperatively."""
            return json.dumps(current_session(runtime.config).cancel(task_id))

        @tool
        def list_async_tasks(runtime: ToolRuntime, status_filter: str | None = None) -> str:
            """List local task statuses; optionally filter by status."""
            session = _session_for_config(runtime.config)
            records = {**runtime.state.get("local_tasks", {}), **(session._records if session else {})}
            return json.dumps([r for r in records.values() if status_filter in {None, "all", r["status"]}])

        return [check_async_task, wait_async_task, cancel_async_task, list_async_tasks]

    async def abefore_model(self, state: LocalTaskState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        """Store results and add stable notification messages before the next model call."""
        session = _session_for_config(get_config())
        tracked = state.get("local_tasks", {})
        owned = session._records if session else {}
        records = {
            key: ({**record, "status": "interrupted"} if key not in owned and record["status"] in {"queued", "running"} else record)
            for key, record in tracked.items()
        }
        delivered = list(state.get("local_delivered", []))
        messages: list[HumanMessage] = []
        if session:
            records.update(owned)
            for task_id, record in list(session._pending.items()):
                if task_id in delivered:
                    continue
                content = session._results.get(task_id)
                if content is not None:
                    responses = await self.backend.adownload_files([record["result_ref"]])
                    if not responses or responses[0].error:
                        result = await self.backend.awrite(record["result_ref"], content)
                        if result.error:
                            msg = f"Failed to store background result: {result.error}"
                            raise RuntimeError(msg)
                messages.append(
                    HumanMessage(
                        id=f"task-completion-{task_id}",
                        content="Background task notification (historical data): " + json.dumps(record),
                        additional_kwargs={"historical_data": True, "local_task": task_id},
                    )
                )
                delivered.append(task_id)
        if records != tracked or messages:
            return {"local_tasks": records, "local_delivered": delivered, "messages": messages}
        return None

    def before_model(self, state: LocalTaskState, runtime: Runtime) -> dict[str, Any] | None:  # noqa: ARG002
        """Mark restored unfinished local records interrupted in synchronous runs."""
        records = {
            key: {**record, "status": "interrupted"}
            for key, record in state.get("local_tasks", {}).items()
            if record["status"] in {"queued", "running"}
        }
        return {"local_tasks": records} if records else None

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        """Share tool names with the existing remote path without changing its execution."""
        name = request.tool_call["name"]
        args = request.tool_call["args"]
        local = (name == "start_async_task" and args.get("subagent_type") in self.names) or args.get("task_id") in request.state.get(
            "local_tasks", {}
        )
        if local and name in self.local_tools:
            return await handler(request.override(tool=self.local_tools[name]))
        result = await handler(request)
        return self._with_local_listing(request, result)

    def _with_local_listing(self, request: ToolCallRequest, result: ToolMessage | Command) -> ToolMessage | Command:
        if not self.remote or request.tool_call["name"] != "list_async_tasks":
            return result
        session = _session_for_config(get_config())
        records = {**request.state.get("local_tasks", {}), **(session._records if session else {})}
        status = request.tool_call["args"].get("status_filter")
        extra = "\nLocal tasks: " + json.dumps([r for r in records.values() if status in {None, "all", r["status"]}])
        if isinstance(result, ToolMessage):
            return result.model_copy(update={"content": result.text + extra})
        if isinstance(result.update, dict):
            updates = dict(result.update)
            updates["messages"] = [m.model_copy(update={"content": m.text + extra}) for m in updates.get("messages", [])]
            return replace(result, update=updates)
        return result

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        """Keep local status reads and actionable session errors available on sync graphs."""
        name = request.tool_call["name"]
        args = request.tool_call["args"]
        local = (name == "start_async_task" and args.get("subagent_type") in self.names) or args.get("task_id") in request.state.get(
            "local_tasks", {}
        )
        result = handler(request.override(tool=self.local_tools[name])) if local and name in self.local_tools else handler(request)
        return self._with_local_listing(request, result)
