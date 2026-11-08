"""Managed activation snapshots outlive history projections without widening tool access."""

from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from novaagents import ContextConfig, ContextMiddleware, ManagedSkillsMiddleware
from novaagents.middleware._context_blocks import collapse_range
from tests.unit_tests.chat_model import GenericFakeChatModel
from tests.unit_tests.middleware.skill_tools_support import SKILLS_SOURCE, skills_backend, write_skill


@tool
def special_lookup(query: str) -> str:
    """Look up exact retained details."""
    return query


def setup_skills(tmp_path):
    write_skill(tmp_path, "house-style", "special_lookup")
    middleware = ManagedSkillsMiddleware(backend=skills_backend(tmp_path), sources=[SKILLS_SOURCE], tools=[special_lookup])
    state = {"messages": [HumanMessage(content="go", id="user")], **middleware.before_agent({}, Runtime(), {})}
    return middleware, state


def make_request(state):
    return ModelRequest(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="done")])), messages=state["messages"], state=state, runtime=Runtime()
    )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_explicit_pins_are_idempotent_complete_snapshots(tmp_path, *, asynchronous):
    middleware, state = setup_skills(tmp_path)
    state["pinned_skills"] = ["house-style", "house-style"]
    update = await middleware.abefore_model(state, Runtime()) if asynchronous else middleware.before_model(state, Runtime())
    active = update["active_skills"]
    assert list(active) == ["house-style"]
    assert active["house-style"]["pinned"]
    assert "Follow these steps." in active["house-style"]["body"]
    assert "messages" not in update
    snapshot = active["house-style"]
    write_skill(tmp_path, "house-style", content="---\nname: house-style\ndescription: New\n---\nDifferent body.")
    state.update(active_skills=active, pinned_skills=["house-style"])
    repeated = await middleware.abefore_model(state, Runtime()) if asynchronous else middleware.before_model(state, Runtime())
    assert repeated["active_skills"]["house-style"] == snapshot


def test_missing_or_unreadable_explicit_pin_fails_visibly(tmp_path):
    middleware, state = setup_skills(tmp_path)
    with pytest.raises(ValueError, match="Unknown skill"):
        middleware.before_model({**state, "pinned_skills": ["missing"]}, Runtime())
    (tmp_path / "skills" / "house-style" / "SKILL.md").write_bytes(b"\xff")
    with pytest.raises(ValueError, match="Cannot activate"):
        middleware.before_model({**state, "pinned_skills": ["house-style"]}, Runtime())
    assert "active_skills" not in state


async def test_lifecycle_tools_visibility_and_refresh(tmp_path):
    middleware, state = setup_skills(tmp_path)
    tools = {t.name: t for t in middleware.tools}
    runtime = SimpleNamespace(state=state, tool_call_id="activate")
    assert "house-style" in tools["search_skills"].func("steps", runtime) or "house-style" in tools["search_skills"].func("house", runtime)
    req, record = middleware.prepare(make_request(state), {})
    assert "special_lookup" not in {t.name for t in req.tools}
    assert not record
    command = await tools["activate_skill"].coroutine("house-style", runtime)
    state["active_skills"] = command.update["active_skills"]
    snapshot = state["active_skills"]["house-style"]
    req, record = middleware.prepare(make_request(state), state["active_skills"])
    assert "special_lookup" in {t.name for t in req.tools}
    assert record == {"special_lookup": "special_lookup"}
    assert req.system_message.text.count(snapshot["body"]) == 1
    write_skill(tmp_path, "house-style", content="---\nname: house-style\ndescription: Changed\n---\nNew instructions.")
    command = await tools["refresh_skills"].coroutine(runtime)
    assert "Source changed" in command.update["messages"][0].content
    assert state["active_skills"]["house-style"] == snapshot
    command = tools["deactivate_skill"].func("house-style", runtime)
    state["active_skills"] = command.update["active_skills"]
    req, record = middleware.prepare(make_request(state), state["active_skills"])
    assert "special_lookup" not in {t.name for t in req.tools}
    assert not record
    command = await tools["activate_skill"].coroutine("house-style", runtime)
    assert "New instructions." in command.update["active_skills"]["house-style"]["body"]
    assert command.update["active_skills"]["house-style"]["hash"] != snapshot["hash"]


def test_later_source_wins_and_sessions_have_independent_snapshots(tmp_path):
    write_skill(tmp_path / "first", "house-style", content="---\nname: house-style\ndescription: First\n---\nFirst body.")
    write_skill(tmp_path / "second", "house-style", content="---\nname: house-style\ndescription: Second\n---\nSecond body.")
    middleware = ManagedSkillsMiddleware(backend=skills_backend(tmp_path), sources=["/first/skills/", "/second/skills/"])
    state = middleware.before_agent({}, Runtime(), {})
    assert len(state["skills_metadata"]) == 1
    assert state["skills_metadata"][0]["description"] == "Second"
    first = middleware.before_model({**state, "pinned_skills": ["house-style"]}, Runtime())
    second = middleware.before_model({**state, "pinned_skills": []}, Runtime())
    assert "Second body." in first["active_skills"]["house-style"]["body"]
    assert second is None


def test_active_skills_survive_projection_and_exclusions(tmp_path):
    skills, state = setup_skills(tmp_path)
    state.update(skills.before_model({**state, "pinned_skills": ["house-style"]}, Runtime()))
    messages = [
        *(AIMessage(content="old " * 500, id=f"m{i}") for i in range(8)),
        HumanMessage(content="current", id="current"),
        *(AIMessage(content="recent", id=f"r{i}") for i in range(5)),
    ]
    block = collapse_range(messages, {}, "m0", "m7", "summary")
    state.update(messages=messages, context_blocks={block["id"]: block})
    middleware = ContextMiddleware(
        make_request(state).model, skills._backend, ContextConfig(input_budget=100000), skills, excluded_tools=frozenset({"special_lookup"})
    )
    rendered, update = middleware._prepare(make_request(state))
    assert rendered.messages[0].id == block["id"]
    assert "Follow these steps." in rendered.system_message.text
    assert "special_lookup" not in {t.name for t in rendered.tools}
    assert not update["_skill_tools_disclosed"]
    assert update["active_skills"]["house-style"]["pinned"]


def test_budget_evicts_agent_selected_skills_oldest_first(tmp_path):
    skills, state = setup_skills(tmp_path)
    active = skills.before_model({**state, "pinned_skills": ["house-style"]}, Runtime())["active_skills"]
    first = {**active["house-style"], "pinned": False, "body": "x" * 5000}
    pinned = {**active["house-style"], "pinned": True, "body": "pinned"}
    state["active_skills"] = {"first": first, "second": first, "pinned": pinned}
    middleware = ContextMiddleware(
        make_request(state).model,
        skills._backend,
        ContextConfig(input_budget=2500, token_counter=lambda ms, **_kw: sum(len(m.text) for m in ms)),
        skills,
    )
    rendered, update = middleware._prepare(make_request(state))
    assert list(update["active_skills"]) == ["pinned"]
    assert rendered.system_message.text.index("Skill first became") < rendered.system_message.text.index("Skill second became")
    assert list(state["active_skills"]) == ["first", "second", "pinned"]
