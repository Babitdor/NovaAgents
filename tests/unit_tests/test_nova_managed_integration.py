"""Exercise Nova graph assembly, local workers, lifecycle, and retained history together."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from novaagents import ContextConfig, ContextMiddleware, NovaSession, create_deep_agent, create_nova_agent
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.middleware.skill_tools_support import SKILLS_SOURCE, call, skills_backend, write_skill


def model(*responses: AIMessage):
    return GenericFakeChatModel(messages=iter([*responses, *(AIMessage(content="done") for _ in range(10))]))


@pytest.mark.parametrize("checkpointed", [False, True])
async def test_local_completion_resumes_idle_graph_and_persists_results(*, checkpointed):
    gate = asyncio.Event()
    started = asyncio.Event()

    async def child(_state):
        started.set()
        await gate.wait()
        return {"messages": [AIMessage(content="exact detail: port 8123")]}

    parent_model = model(
        AIMessage(content="", tool_calls=[call("start_async_task", "start", description="find port", subagent_type="helper")]),
        AIMessage(content="parent continues"),
        AIMessage(content="completion received"),
    )
    agent = create_nova_agent(
        parent_model,
        context_management=ContextConfig(input_budget=100000),
        checkpointer=InMemorySaver() if checkpointed else None,
        subagents=[{"name": "helper", "description": "Lookup", "runnable": RunnableLambda(child)}],
    )
    async with NovaSession(agent, max_concurrency=1) as session:
        result = await session.ainvoke({"messages": [HumanMessage(content="research")]})
        await asyncio.wait_for(started.wait(), 2)
        assert result["messages"][-1].text == "parent continues"
        assert not gate.is_set()
        gate.set()
        await asyncio.wait_for(session.wait_until_idle(), 5)
        assert session._state["messages"][-1].text == "completion received"
        record = next(iter(session._state["local_tasks"].values()))
        assert record["status"] == "success"
        path = record["result_ref"]
        assert path in session._state["files"]
        payload = json.loads(session._state["files"][path]["content"])
        assert payload["messages"][0]["data"]["content"] == "exact detail: port 8123"
        assert sum(m.additional_kwargs.get("local_task") == record["task_id"] for m in session._state["messages"]) == 1


async def test_background_tools_outside_session_report_how_to_open_one():
    parent_model = model(AIMessage(content="", tool_calls=[call("start_async_task", "start", description="go", subagent_type="helper")]))
    agent = create_nova_agent(
        parent_model,
        context_management=ContextConfig(input_budget=100000),
        subagents=[{"name": "helper", "description": "Helper", "runnable": RunnableLambda(lambda _state: {"messages": []})}],
    )
    with pytest.raises(RuntimeError, match="open NovaSession"):
        await agent.ainvoke({"messages": [HumanMessage(content="go")]})


async def test_concurrent_children_skill_collapse_and_exact_recall(tmp_path):
    backend = skills_backend(tmp_path)
    write_skill(tmp_path, "house-style")
    gate = asyncio.Event()
    children_started = []

    async def child(state):
        assert not state.get("active_skills")
        assert not state.get("context_blocks")
        children_started.append(state["messages"][0].text)
        await gate.wait()
        return {"messages": [AIMessage(content="exact detail: signature connect(host, *, port=8123)")]}

    @tool
    def read_results(runtime: ToolRuntime) -> Command:
        """Read completed task artifacts as historical data."""
        paths = [r["result_ref"] for r in runtime.state["local_tasks"].values()]
        contents = backend.download_files(paths)
        text = "\n".join(r.content.decode() for r in contents)
        return Command(update={"messages": [ToolMessage(id="child-results", content=text, tool_call_id=runtime.tool_call_id)]})

    parent_model = model(
        AIMessage(
            id="delegation",
            content="",
            tool_calls=[
                call("start_async_task", "first", description="first", subagent_type="helper"),
                call("start_async_task", "second", description="second", subagent_type="helper"),
                {"name": "activate_skill", "id": "activate", "args": {"name": "house-style"}, "type": "tool_call"},
            ],
        ),
        AIMessage(content="parent continues"),
        AIMessage(content="children completed"),
        AIMessage(id="read-calls", content="", tool_calls=[call("read_results", "read")]),
        AIMessage(content="results read"),
        AIMessage(
            content="",
            tool_calls=[
                call("collapse", "collapse", start_ref="read-calls", end_ref="child-results", summary="Children returned a connection signature.")
            ],
        ),
        AIMessage(content="collapsed"),
    )
    agent = create_nova_agent(
        parent_model,
        tools=[read_results],
        backend=backend,
        skills=[SKILLS_SOURCE],
        checkpointer=InMemorySaver(),
        context_management=ContextConfig(input_budget=100000),
        subagents=[{"name": "helper", "description": "Lookup", "runnable": RunnableLambda(child)}],
    )
    async with NovaSession(agent, max_concurrency=2) as session:
        await session.ainvoke({"messages": [HumanMessage(content="delegate and activate")]})
        gate.set()
        await asyncio.wait_for(session.wait_until_idle(), 5)
        assert sorted(children_started) == ["first", "second"]
        await session.ainvoke({"messages": [HumanMessage(content="read results")]})
        padding = [AIMessage(content=f"later step {i}", id=f"padding-{i}") for i in range(6)]
        await session.ainvoke({"messages": [*padding, HumanMessage(content="collapse the old results")]})
        state = session._state
        block = next(iter(state["context_blocks"].values()))
        assert "child-results" in block["refs"]
        assert "house-style" in state["active_skills"]
        assert any("active_skill" in call["messages"][0].text for call in parent_model.call_history)
        controller = ContextMiddleware(parent_model, backend, ContextConfig(input_budget=100000))
        recall = next(t for t in controller.tools if t.name == "uncollapse")
        retained = json.loads(await recall.coroutine(block["id"], SimpleNamespace(state=state, tool_call_id="recall"), full=True))
        assert "connect(host, *, port=8123)" in json.dumps(retained)
        assert state["context_blocks"][block["id"]]["collapsed"]


def test_compatibility_factory_and_nova_factory_are_distinct():
    assert create_nova_agent is not create_deep_agent
    deep = create_deep_agent(model(AIMessage(content="compatible")))
    assert deep.invoke({"messages": [HumanMessage(content="go")]})["messages"][-1].text == "compatible"


async def test_restored_unfinished_tasks_become_interrupted():
    gate = asyncio.Event()

    async def child(_state):
        await gate.wait()
        return {"messages": [AIMessage(content="done")]}

    parent_model = model(
        AIMessage(content="", tool_calls=[call("start_async_task", "start", description="work", subagent_type="helper")]),
        AIMessage(content="parent continues"),
        AIMessage(content="restored"),
    )
    agent = create_nova_agent(
        parent_model,
        checkpointer=InMemorySaver(),
        context_management=ContextConfig(input_budget=100000),
        subagents=[{"name": "helper", "description": "Helper", "runnable": RunnableLambda(child)}],
    )
    async with NovaSession(agent, thread_id="restart") as session:
        await session.ainvoke({"messages": [HumanMessage(content="start")]})
        assert next(iter(session._state["local_tasks"].values()))["status"] in {"queued", "running"}
    async with NovaSession(agent, thread_id="restart") as restored:
        result = await restored.ainvoke({"messages": [HumanMessage(content="restore")]})
        assert next(iter(result["local_tasks"].values()))["status"] == "interrupted"
