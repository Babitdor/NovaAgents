"""Reversible context retains originals and checks the complete model request."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, messages_to_dict

from novaagents import ContextBudgetError, ContextConfig, ContextMiddleware
from novaagents.middleware._context_blocks import collapse_range, eligible_refs, expand_block, project
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.middleware.test_summarization_middleware import MockBackend


def history():
    old = [AIMessage(content=f"detail {i}: " + "x" * 500, id=f"m{i}") for i in range(8)]
    return [*old, HumanMessage(content="latest task", id="latest"), *(AIMessage(content=f"recent {i}", id=f"r{i}") for i in range(5))]


def count(messages, *, tools=None):
    return sum(len(m.text) for m in messages) + len(str(tools or ""))


def controller(*, budget=100000, model=None):
    return ContextMiddleware(
        model or GenericFakeChatModel(messages=iter([AIMessage(content="summary")]), profile={"max_input_tokens": 100000}),
        MockBackend(),
        ContextConfig(input_budget=budget, token_counter=count),
    )


def request(messages, **state: object):
    return ModelRequest(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="answer")])),
        messages=messages,
        state={"messages": messages, **state},
        runtime=SimpleNamespace(context=None),
    )


def test_three_tiers_round_trip_preserves_message_payloads():
    messages = history()
    messages[0] = AIMessage(id="m0", content="call", tool_calls=[{"id": "call", "name": "lookup", "args": {"path": "/exact/path"}}])
    messages[1] = ToolMessage(id="m1", tool_call_id="call", content=[{"type": "image_url", "image_url": {"url": "https://example.com/retained.png"}}])
    original = messages_to_dict(messages)
    blocks = {}
    tier1 = []
    for i in range(0, 8, 2):
        block = collapse_range(messages, blocks, f"m{i}", f"m{i + 1}", f"summary {i}")
        blocks[block["id"]] = block
        tier1.append(block["id"])
    tier2 = []
    for i in (0, 2):
        block = collapse_range(messages, blocks, tier1[i], tier1[i + 1], "combined")
        blocks[block["id"]] = block
        tier2.append(block["id"])
    top = collapse_range(messages, blocks, tier2[0], tier2[1], "top")
    blocks[top["id"]] = top
    assert top["tier"] == 3
    assert project(messages, blocks)[0].id == top["id"]
    expanded = expand_block(blocks, top["id"], full=True)
    assert messages_to_dict(project(messages, expanded)) == original
    assert messages_to_dict(messages) == original
    assert blocks[top["id"]]["collapsed"]


@pytest.mark.parametrize(("start", "end"), [("unknown", "m1"), ("m3", "m1"), ("latest", "r4"), ("r0", "r1")])
def test_invalid_or_protected_ranges_are_transactional(start, end):
    messages = history()
    original = deepcopy(messages)
    with pytest.raises(ValueError, match=r"references|protected"):
        collapse_range(messages, {}, start, end, "summary")
    assert messages == original


def test_tool_pairs_system_messages_and_partial_overlaps_are_rejected():
    messages = history()
    messages[0] = AIMessage(id="m0", content="", tool_calls=[{"id": "call", "name": "lookup", "args": {}}])
    messages[1] = ToolMessage(id="m1", content="result", tool_call_id="call")
    with pytest.raises(ValueError, match="every tool result"):
        collapse_range(messages, {}, "m0", "m0", "partial")
    block = collapse_range(messages, {}, "m0", "m2", "whole")
    blocks = {block["id"]: {**block, "collapsed": False}}
    with pytest.raises(ValueError, match="partially overlaps"):
        collapse_range(messages, blocks, "m2", "m3", "partial")
    messages.insert(0, SystemMessage(content="system", id="system"))
    with pytest.raises(ValueError, match="protected"):
        collapse_range(messages, {}, "system", "m2", "bad")


def test_cross_thread_blocks_and_tier_four_are_rejected():
    messages = history()
    block = collapse_range(messages, {}, "m0", "m7", "summary")
    with pytest.raises(ValueError, match="Unknown"):
        expand_block({}, block["id"], full=True)
    blocks = {block["id"]: {**block, "tier": 3}}
    with pytest.raises(ValueError, match="Higher tiers"):
        collapse_range(messages, blocks, block["id"], block["id"], "T4")


def test_single_child_hierarchy_projects_outermost_tier():
    messages = history()
    first = collapse_range(messages, {}, "m0", "m7", "long T1 summary")
    blocks = {first["id"]: first}
    second = collapse_range(messages, blocks, first["id"], first["id"], "T2")
    blocks[second["id"]] = second
    assert project(messages, blocks)[0].id == second["id"]
    assert project(messages, expand_block(blocks, second["id"], full=False))[0].id == first["id"]


async def test_parallel_recalls_are_data_reads_and_parallel_expansions_are_rejected():
    messages = history()
    block = collapse_range(messages, {}, "m0", "m7", "summary")
    calls = [{"id": f"recall-{i}", "name": "uncollapse", "args": {"block_id": block["id"]}} for i in range(2)]
    messages.append(AIMessage(content="", tool_calls=calls, id="recalls"))
    runtime = SimpleNamespace(state={"messages": messages, "context_blocks": {block["id"]: block}}, tool_call_id="recall-0")
    middleware = controller()
    result = middleware._uncollapse_content(block["id"], runtime, "recall", full=True)
    assert "detail 0" in result
    for call in calls:
        call["args"]["mode"] = "expand"
    messages[-1] = AIMessage(content="", tool_calls=calls, id="recalls")
    with pytest.raises(ValueError, match="sequentially"):
        middleware._uncollapse_content(block["id"], runtime, "expand", full=True)


def test_long_tool_turn_can_compact_without_losing_latest_user_intent():
    messages = [
        HumanMessage(content="original user intent", id="user"),
        *history()[:8],
        HumanMessage(content="task completed", id="notification", additional_kwargs={"historical_data": True}),
        *(AIMessage(content="recent", id=f"r{i}") for i in range(5)),
    ]
    eligible = eligible_refs(messages)
    assert "user" not in eligible
    assert "m0" in eligible
    assert "r0" not in eligible
    with pytest.raises(ValueError, match="protected"):
        collapse_range(messages, {}, "user", "m0", "drop user intent")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_auto_compaction_changes_projection_only(monkeypatch, *, asynchronous):
    middleware = controller(budget=3000)
    original = history()
    req = request(original)
    monkeypatch.setattr(middleware.accounting, "_create_summary", lambda _messages: "Paths and decisions retained.")

    async def summarize(_messages):
        return "Paths and decisions retained."

    monkeypatch.setattr(middleware.accounting, "_acreate_summary", summarize)
    sent = []

    def handler(current):
        sent.append(current)
        return ModelResponse(result=[AIMessage(content="done")])

    async def ahandler(current):
        return handler(current)

    result = await middleware.awrap_model_call(req, ahandler) if asynchronous else middleware.wrap_model_call(req, handler)
    assert len(sent[0].messages) < len(original)
    assert result.command.update["context_blocks"]
    assert "messages" not in result.command.update
    assert original[0].text.endswith("x" * 500)
    assert sent[0].messages[-6:] == original[-6:]
    assert middleware._count(sent[0]) <= 3000


def test_unknown_budget_and_protected_overflow_fail_before_model():
    middleware = controller()
    middleware.config = ContextConfig(token_counter=count)
    req = request([HumanMessage(content="x", id="user")])
    with pytest.raises(ContextBudgetError, match="unknown"):
        middleware._prepare(req)
    middleware = controller(budget=500)
    req = request([HumanMessage(content="x", id="user")]).override(system_message=SystemMessage(content="protected" * 500))
    with pytest.raises(ContextBudgetError, match="at most two"):
        middleware._prepare(req)


def test_insufficient_summary_reduction_leaves_state_unchanged(monkeypatch):
    middleware = controller(budget=3000)
    messages = history()
    req = request(messages)
    monkeypatch.setattr(middleware.accounting, "_create_summary", lambda _selected: "x" * 20000)
    with pytest.raises(ContextBudgetError, match="did not reduce"):
        middleware._prepare(req)
    assert "context_blocks" not in req.state
    assert req.messages == messages


async def test_recall_search_and_oversized_expansion_leave_projection_unchanged():
    middleware = controller(budget=100)
    messages = history()
    block = collapse_range(messages, {}, "m0", "m7", "summary")
    blocks = {block["id"]: block}
    runtime = SimpleNamespace(state={"messages": messages, "context_blocks": blocks, "context_input_budget": 100000}, tool_call_id="tool")
    tools = {t.name: t for t in middleware.tools}

    result = json.loads(await tools["uncollapse"].coroutine(block["id"], runtime, full=True))
    assert result["historical_data"]
    assert result["messages"][0]["data"]["content"] == messages[0].content
    assert "m3" in tools["search_context"].func("detail 3", runtime)
    runtime.state["context_input_budget"] = 100
    with pytest.raises(ContextBudgetError, match="unchanged"):
        await tools["uncollapse"].coroutine(block["id"], runtime, mode="expand", full=True)
    assert blocks[block["id"]]["collapsed"]
    with pytest.raises(ValueError, match="Unknown"):
        await tools["uncollapse"].coroutine("other-thread-block", runtime)


def test_recovery_is_bounded_to_two_summaries(monkeypatch):
    middleware = controller(budget=800)
    messages = history()
    calls = []

    def summarize(selected):
        calls.append(selected)
        return "s" * max(1, sum(len(m.text) for m in selected) // 2)

    monkeypatch.setattr(middleware.accounting, "_create_summary", summarize)
    with pytest.raises(ContextBudgetError, match="at most two"):
        middleware._prepare(request(messages))
    assert len(calls) == 2


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        ContextConfig(input_budget=budget)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("oversized_group", [False, True])
async def test_summary_uses_model_capacity_instead_of_parent_target(monkeypatch, *, asynchronous, oversized_group):
    model = GenericFakeChatModel(messages=iter([]), profile={"max_input_tokens": 100000})
    middleware = controller(budget=3000, model=model)
    messages = history()
    if oversized_group:
        messages[0] = AIMessage(id="m0", content="", tool_calls=[{"id": "call", "name": "read_file", "args": {"path": "/large"}}])
        messages[1] = ToolMessage(id="m1", tool_call_id="call", content="large result " * 2000)
    else:
        messages[:0] = [AIMessage(id=f"burst{i}", content="old detail " * 300) for i in range(15)]
    selected = []

    def summarize(candidate):
        selected.extend(candidate)
        return "Preserve the API; exact originals remain available."

    async def asummarize(candidate):
        return summarize(candidate)

    monkeypatch.setattr(middleware.accounting, "_create_summary", summarize)
    monkeypatch.setattr(middleware.accounting, "_acreate_summary", asummarize)
    original = messages_to_dict(messages)
    prepared, update = await middleware._aprepare(request(messages)) if asynchronous else middleware._prepare(request(messages))
    assert middleware._count(prepared) <= 3000
    assert count(selected) > 3000
    assert messages_to_dict(messages) == original
    restored = project(messages, expand_block(update["context_blocks"], next(iter(update["context_blocks"])), full=True))
    assert messages_to_dict(restored) == original


def test_small_prefix_does_not_hide_later_complete_tool_group(monkeypatch):
    model = GenericFakeChatModel(messages=iter([]), profile={"max_input_tokens": 15000})
    middleware = controller(budget=3000, model=model)
    messages = [
        AIMessage(content="tiny", id="tiny"),
        HumanMessage(content="preserve user intent", id="latest"),
        AIMessage(content="", id="call", tool_calls=[{"id": "lookup", "name": "lookup", "args": {}}]),
        ToolMessage(content="x" * 6000, tool_call_id="lookup", id="result"),
        *(AIMessage(content="recent", id=f"r{i}") for i in range(5)),
    ]
    selected = []
    monkeypatch.setattr(middleware.accounting, "_create_summary", lambda candidate: selected.extend(candidate) or "lookup completed")
    prepared, _update = middleware._prepare(request(messages))
    assert {m.id for m in selected} == {"call", "result"}
    assert middleware._count(prepared) < 3000
    assert messages[1] in prepared.messages


def test_summary_prompt_and_complete_groups_fit_known_capacity(monkeypatch):
    model = GenericFakeChatModel(messages=iter([]), profile={"max_input_tokens": 5500})
    middleware = controller(budget=3000, model=model)
    messages = history()
    messages[:0] = [
        AIMessage(id="huge-call", content="", tool_calls=[{"id": "huge", "name": "lookup", "args": {}}]),
        ToolMessage(id="huge-result", tool_call_id="huge", content="x" * 10000),
    ]
    prompts = []

    def summarize(selected):
        prompt = middleware.accounting._lc_helper._format_summary_prompt(selected)
        prompts.append(prompt)
        assert count([HumanMessage(content=prompt)]) <= int(5500 * 0.95)
        assert not {"huge-call", "huge-result"} & {m.id for m in selected}
        return "summary"

    monkeypatch.setattr(middleware.accounting, "_create_summary", summarize)
    with pytest.raises(ContextBudgetError):
        middleware._prepare(request(messages))
    assert prompts


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_nonshrinking_range_retries_another_range_without_committing_it(monkeypatch, *, asynchronous):
    middleware = controller(budget=3000)
    messages = history()
    block = collapse_range(messages, {}, "m0", "m2", "old summary " * 150)
    blocks = {block["id"]: block}
    selected = []

    def summarize(candidate):
        selected.append([m.id for m in candidate])
        return "x" * 6000 if len(selected) == 1 else "Preserved decisions."

    async def asummarize(candidate):
        return summarize(candidate)

    monkeypatch.setattr(middleware.accounting, "_create_summary", summarize)
    monkeypatch.setattr(middleware.accounting, "_acreate_summary", asummarize)
    req = request(messages, context_blocks=blocks)
    prepared, update = await middleware._aprepare(req) if asynchronous else middleware._prepare(req)
    assert len(selected) == 2
    assert set(selected[0]) == {f"m{i}" for i in range(3, 8)}
    assert selected[1] == [block["id"]]
    assert middleware._count(prepared) <= 3000
    assert len(update["context_blocks"]) == 2
    assert blocks == {block["id"]: block}
    assert project(messages, blocks)[0].text.endswith(block["summary"])


def test_unknown_model_capacity_bounds_actual_summary_prompts():
    model = GenericFakeChatModel(messages=iter([AIMessage(content="summary")] * 2))
    middleware = controller(budget=5000, model=model)
    req = request(history()).override(system_message=SystemMessage(content="protected" * 80))
    prepared, _update = middleware._prepare(req)
    assert model.call_history
    assert all(count(call["messages"]) <= 5000 for call in model.call_history)
    assert middleware._count(prepared) <= 5000
