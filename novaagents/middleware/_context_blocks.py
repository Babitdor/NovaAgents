"""Validated, reversible projections over canonical conversation messages."""

from __future__ import annotations

from typing import TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage, get_notification_metadata

_MAX_TIER = 3


class ContextBlock(TypedDict):
    """Checkpointed summary with a route to its unchanged originals."""

    id: str
    refs: list[str]
    summary: str
    topic: str | None
    children: list[str]
    tier: int
    collapsed: bool


def message_ref(message: AnyMessage) -> str:
    """Require the stable IDs supplied by LangGraph before projection edits."""
    if message.id is None:
        msg = "Context messages require stable IDs assigned by LangGraph."
        raise ValueError(msg)
    return message.id


def message_groups(messages: list[AnyMessage]) -> list[list[AnyMessage]]:
    """Group each assistant tool call with all its results; reject broken pairs."""
    groups: list[list[AnyMessage]] = []
    pending: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                msg = "History contains an orphan or duplicate tool result."
                raise ValueError(msg)
            groups[-1].append(message)
            pending.remove(message.tool_call_id)
        else:
            if pending:
                msg = "History contains unresolved tool calls before another message."
                raise ValueError(msg)
            groups.append([message])
            if isinstance(message, AIMessage):
                pending = {call["id"] for call in message.tool_calls if call["id"]}
                if len(pending) != len(message.tool_calls):
                    msg = "History contains missing or duplicate tool-call IDs."
                    raise ValueError(msg)
    return groups


def eligible_refs(messages: list[AnyMessage]) -> set[str]:
    """Protect system messages, the latest user turn, and five recent groups."""
    groups = message_groups(messages)
    latest_user = max(
        (
            i
            for i, group in enumerate(groups)
            if isinstance(group[0], HumanMessage)
            and not group[0].additional_kwargs.get("historical_data")
            and not get_notification_metadata(group[0])
        ),
        default=-1,
    )
    return {
        message.id
        for index, group in enumerate(groups[: max(0, len(groups) - 5)])
        for message in group
        if index != latest_user and message.id is not None and not isinstance(message, SystemMessage)
    }


def project(messages: list[AnyMessage], blocks: dict[str, ContextBlock]) -> list[AnyMessage]:
    """Render outer collapsed blocks without modifying canonical messages."""
    positions = {message.id: i for i, message in enumerate(messages)}
    collapsed = sorted((b for b in blocks.values() if b["collapsed"]), key=lambda b: (-len(b["refs"]), -b["tier"], b["id"]))
    covered: set[str] = set()
    starts: dict[str, ContextBlock] = {}
    for block in collapsed:
        refs = set(block["refs"])
        if refs & covered:
            continue
        if any(ref not in positions for ref in refs):
            msg = "A context block references missing canonical messages."
            raise ValueError(msg)
        starts[block["refs"][0]] = block
        covered.update(refs)
    result: list[AnyMessage] = []
    for message in messages:
        ref = message_ref(message)
        if ref in starts:
            block = starts[ref]
            result.append(
                HumanMessage(
                    id=block["id"],
                    content=f"Historical summary [{block['id']}] T{block['tier']}: {block['summary']}",
                    additional_kwargs={"context_block": block["id"], "historical_data": True},
                )
            )
        elif ref not in covered:
            result.append(message)
    return result


def collapse_range(  # One validation transaction must finish before creating a block.
    messages: list[AnyMessage], blocks: dict[str, ContextBlock], start: str, end: str, summary: str, topic: str | None = None
) -> ContextBlock:
    """Validate a complete contiguous range and return a new T1, T2, or T3 block."""
    if not summary.strip():
        msg = "A collapse summary must not be empty."
        raise ValueError(msg)
    visible = project(messages, blocks)
    refs = [message_ref(message) for message in visible]
    if start not in refs or end not in refs or refs.index(start) > refs.index(end):
        msg = "Unknown, hidden, or reversed context references; use context_status."
        raise ValueError(msg)
    selected = visible[refs.index(start) : refs.index(end) + 1]
    children = [message_ref(message) for message in selected if message.id in blocks]
    tiers = {blocks[child]["tier"] for child in children}
    if children and (len(children) != len(selected) or len(tiers) != 1 or max(tiers) >= _MAX_TIER):
        msg = "Higher tiers require only complete blocks of the same tier (T1 or T2)."
        raise ValueError(msg)
    originals = [ref for message in selected for ref in (blocks[message_ref(message)]["refs"] if message.id in blocks else [message_ref(message)])]
    if not set(originals) <= eligible_refs(messages):
        msg = "Range includes protected content or messages without stable IDs."
        raise ValueError(msg)
    selected_refs = set(originals)
    for group in message_groups(messages):
        group_refs = {message.id for message in group}
        if group_refs & selected_refs and not group_refs <= selected_refs:
            msg = "Collapse must include an assistant call and every tool result."
            raise ValueError(msg)
    for block in blocks.values():
        overlap = set(block["refs"]) & selected_refs
        if overlap and not set(block["refs"]) <= selected_refs:
            msg = "Collapse partially overlaps an existing block."
            raise ValueError(msg)
    return ContextBlock(
        id=f"ctx-{uuid4().hex}",
        refs=originals,
        summary=summary,
        topic=topic,
        children=children,
        tier=(max(tiers) + 1 if tiers else 1),
        collapsed=True,
    )


def expand_block(blocks: dict[str, ContextBlock], block_id: str, *, full: bool) -> dict[str, ContextBlock]:
    """Return expanded metadata, leaving the supplied transaction unchanged."""
    if block_id not in blocks:
        msg = "Unknown context block in this thread."
        raise ValueError(msg)
    result = dict(blocks)
    block = result[block_id]
    if any(b["collapsed"] and block_id in b["children"] for b in blocks.values()):
        msg = "Expand the enclosing block first."
        raise ValueError(msg)
    result[block_id] = {**block, "collapsed": False}
    if full:
        for child in block["children"]:
            result = expand_block(result, child, full=True)
    return result
