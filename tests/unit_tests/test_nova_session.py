"""Network-free supervisor tests, independent of provider or graph internals."""

import asyncio
import json
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from novaagents.session import _CURRENT_SESSION, NovaSession, current_session


class Parent:
    """Consume completions at a simulated graph boundary."""

    checkpointer = None

    def __init__(self) -> None:
        self.deliveries = []
        self.inputs = []

    async def astream(self, inputs, config, **kwargs: object):
        self.inputs.append(inputs)
        session = current_session()
        delivered = list(inputs.get("local_delivered", []))
        for task_id in session._pending:
            if task_id not in delivered:
                delivered.append(task_id)
                self.deliveries.append(task_id)
        yield {**inputs, "local_delivered": delivered}


class Child:
    def __init__(self, gate=None, *, fail=False) -> None:
        self.gate = gate
        self.fail = fail
        self.started = asyncio.Event()
        self.cleaned = asyncio.Event()
        self.inputs = []
        self.configs = []

    async def ainvoke(self, inputs, config):
        self.inputs.append(inputs)
        self.configs.append(config)
        self.started.set()
        try:
            if self.gate:
                await self.gate.wait()
            if self.fail:
                msg = "child failed"
                raise ValueError(msg)
            return {"messages": []}
        finally:
            self.cleaned.set()


class TestNovaSession(unittest.IsolatedAsyncioTestCase):
    async def test_parent_continues_while_child_blocked_and_queue_is_bounded(self):
        parent = Parent()
        gate = asyncio.Event()
        children = [Child(gate), Child(), Child()]
        async with NovaSession(parent, max_concurrency=1) as session:
            first = session.start(children[0], {"messages": []}, {}, "first")
            await children[0].started.wait()
            queued = session.start(children[1], {"messages": []}, {}, "queued")
            last = session.start(children[2], {"messages": []}, {}, "last")
            result = await session.ainvoke({"messages": ["parent continues"]})
            assert result["messages"] == ["parent continues"]
            assert not children[1].started.is_set()
            assert not children[2].started.is_set()
            assert session.cancel(queued["task_id"])["status"] == "cancelled"
            gate.set()
            await session.wait_until_idle()
            assert session.check(first["task_id"])["status"] == "success"
            assert session.check(last["task_id"])["status"] == "success"
            assert not children[1].started.is_set()
            assert sorted(parent.deliveries) == sorted([first["task_id"], queued["task_id"], last["task_id"]])
            assert len(parent.deliveries) == 3
            assert children[0].configs[0]["configurable"]["thread_id"] != session.thread_id

    async def test_idle_resumption_events_failure_and_state_retention(self):
        parent = Parent()
        async with NovaSession(parent) as session:
            await session.ainvoke({"messages": ["initial"]})
            record = session.start(Child(fail=True), {}, {}, "fails")
            event = await asyncio.wait_for(anext(session.events()), 2)
            assert event["task_id"] == record["task_id"]
            assert event["status"] == "error"
            await session.wait_until_idle()
            assert parent.deliveries == [record["task_id"]]
            assert parent.inputs[-1]["messages"] == ["initial"]
            assert session.check(record["task_id"])["error"] == "child failed"

    async def test_sessions_isolate_ids_and_snapshot_child_input(self):
        child = Child()
        state = {"files": {"path": "original"}}
        async with NovaSession(Parent()) as first, NovaSession(Parent()) as second:
            record = first.start(child, state, {}, "helper")
            state["files"]["path"] = "changed"
            with self.assertRaisesRegex(ValueError, "Unknown local task"):  # noqa: PT027  # Keep supervisor tests runnable with the standard library alone.
                second.check(record["task_id"])
            await first.wait_until_idle()
            assert child.inputs[0]["files"]["path"] == "original"
            assert not second._records

    async def test_close_cancels_running_and_queued_work_and_waits_for_cleanup(self):
        running = Child(asyncio.Event())
        queued = Child()
        session = NovaSession(Parent(), max_concurrency=1)
        async with session:
            first = session.start(running, {}, {}, "running")
            await running.started.wait()
            second = session.start(queued, {}, {}, "queued")
        assert running.cleaned.is_set()
        assert not queued.started.is_set()
        assert all(handle.done() for handle in session._handles.values())
        assert session.check(first["task_id"])["status"] == "cancelled"
        assert session.check(second["task_id"])["status"] == "cancelled"
        with self.assertRaisesRegex(RuntimeError, "closing"):  # noqa: PT027  # Keep supervisor tests runnable with the standard library alone.
            session.start(Child(), {}, {}, "late")

    async def test_same_checkpoint_thread_cannot_have_two_owners(self):
        parent = Parent()
        async with NovaSession(parent, thread_id="same"):
            with self.assertRaisesRegex(RuntimeError, "already owns"):  # noqa: PT027  # Keep supervisor tests runnable with the standard library alone.
                async with NovaSession(parent, thread_id="same"):
                    self.fail("must reject duplicate ownership")
        async with NovaSession(parent, thread_id="same"):
            pass

    async def test_parent_errors_are_observable(self):
        class BrokenParent(Parent):
            async def astream(self, inputs, config, **kwargs: object):
                yield {"messages": []}
                msg = "parent failed"
                raise ValueError(msg)

        async with NovaSession(BrokenParent()) as session:
            session.start(Child(), {}, {}, "helper")
            with self.assertRaisesRegex(ValueError, "parent failed"):  # noqa: PT027  # Keep supervisor tests runnable with the standard library alone.
                await session.wait_until_idle()

    async def test_outside_session_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, "open NovaSession"):  # noqa: PT027  # Keep supervisor tests runnable with the standard library alone.
            current_session()
        session = NovaSession(Parent())
        with self.assertRaisesRegex(RuntimeError, "async with"):  # noqa: PT027  # Keep supervisor tests runnable with the standard library alone.
            await session.ainvoke({"messages": []})

    async def test_nested_graph_and_foreign_thread_cannot_access_parent_session(self):
        async with NovaSession(Parent(), thread_id="parent") as session:
            token = _CURRENT_SESSION.set(session)
            try:
                assert current_session({"configurable": {"thread_id": "parent", "checkpoint_ns": "tools:call"}}) is session
                for thread_id, namespace in [("other", ""), ("parent", "tools:call|model:child")]:
                    with self.assertRaisesRegex(RuntimeError, "open NovaSession"):  # noqa: PT027  # Standard-library supervisor check.
                        current_session({"configurable": {"thread_id": thread_id, "checkpoint_ns": namespace}})
            finally:
                _CURRENT_SESSION.reset(token)

    async def test_wait_until_idle_includes_foreground_parent_invocations(self):
        gate = asyncio.Event()
        started = asyncio.Event()

        class BusyParent(Parent):
            async def astream(self, inputs, config, **kwargs: object):
                started.set()
                await gate.wait()
                async for values in super().astream(inputs, config, **kwargs):
                    yield values

        async with NovaSession(BusyParent()) as session:
            foreground = asyncio.create_task(session.ainvoke({"messages": []}))
            await started.wait()
            waiting = asyncio.create_task(session.wait_until_idle())
            await asyncio.sleep(0)
            assert not waiting.done()
            gate.set()
            await asyncio.gather(foreground, waiting)

    async def test_idle_completion_does_not_bypass_parent_approval(self):
        class PausedParent(Parent):
            checkpointer = object()

            async def aget_state(self, config):
                return SimpleNamespace(values={"messages": []}, tasks=[SimpleNamespace(interrupts=("approval",))])

        parent = PausedParent()
        async with NovaSession(parent) as session:
            result = await session.ainvoke({"messages": []})
            assert result["__interrupt__"] == ("approval",)
            record = session.start(Child(), {}, {}, "helper")
            await session.wait_until_idle()
            assert len(parent.inputs) == 1
            assert record["task_id"] in session._pending

    async def test_cancelling_a_waiter_does_not_cancel_the_child(self):
        gate = asyncio.Event()
        child = Child(gate)
        async with NovaSession(Parent()) as session:
            record = session.start(child, {}, {}, "helper")
            await child.started.wait()
            waiter = asyncio.create_task(session.wait(record["task_id"]))
            await asyncio.sleep(0)
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            assert session.check(record["task_id"])["status"] == "running"
            gate.set()
            await session.wait_until_idle()
            assert session.check(record["task_id"])["status"] == "success"

    async def test_structured_results_and_nonempty_final_output_are_retained(self):
        @dataclass
        class Findings:
            port: int

        class Message:
            type = "ai"

            def __init__(self, content) -> None:
                self.content = content

            def model_dump(self):
                return {"content": self.content, "id": "original"}

        retained = json.loads(
            NovaSession._result_payload({"messages": [Message("exact detail"), Message("")], "structured_response": Findings(port=8123)})
        )
        assert retained["messages"][0]["data"] == {"content": "exact detail", "id": "original"}
        assert retained["structured_response"] == {"port": 8123}


if __name__ == "__main__":
    unittest.main()
