"""Nova's coordinated, reversible context and skills request pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ExtendedModelResponse, ModelRequest, ModelResponse, PrivateStateAttr
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, messages_to_dict
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command

from novaagents.middleware._context_blocks import (
    _MAX_TIER,
    ContextBlock,
    collapse_range,
    eligible_refs,
    expand_block,
    message_groups,
    message_ref,
    project,
)
from novaagents.middleware._utils import append_to_system_message
from novaagents.middleware.managed_skills import _MUTATIONS, ManagedSkillsMiddleware, SkillSnapshot, serial_mutation
from novaagents.middleware.summarization import SummarizationMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models import BaseChatModel

    from novaagents.backends.protocol import BackendProtocol

_SOFT = 0.70
_COMPACT = 0.85
_RECALL_LIMIT = 12000
_SEARCH_LIMIT = 20


class ContextBudgetError(ValueError):
    """The complete request cannot fit without losing protected content."""


@dataclass(frozen=True)
class ContextConfig:
    """Context policy; an explicit budget is required for models without input limits.

    Args:
        input_budget: Maximum input tokens, overriding the model's derived limit.
        token_counter: Counter accepting messages and optionally tool schemas.
    """

    input_budget: int | None = None
    token_counter: Callable[..., int] = count_tokens_approximately

    def __post_init__(self) -> None:
        """Validate an explicit budget before graph construction."""
        if self.input_budget is not None and (
            not isinstance(self.input_budget, int) or isinstance(self.input_budget, bool) or self.input_budget <= 0
        ):
            msg = "input_budget must be a positive integer."
            raise ValueError(msg)


class ContextState(AgentState):
    """Private checkpointed projection metadata; messages remain canonical."""

    context_blocks: NotRequired[Annotated[dict[str, ContextBlock], PrivateStateAttr]]
    context_input_budget: NotRequired[Annotated[int, PrivateStateAttr]]
    context_overhead: NotRequired[Annotated[int, PrivateStateAttr]]


class ContextMiddleware(AgentMiddleware):
    """Own all Nova projection, skill rendering, and final budget enforcement.

    Args:
        model: Configured model used for automatic summaries.
        backend: Backend for bounded recall artifacts.
        config: Input budget and token accounting configuration.
        skills: Optional managed skill lifecycle middleware.
        excluded_tools: Profile exclusions enforced after skill disclosure.
    """

    state_schema = ContextState

    def __init__(
        self,
        model: BaseChatModel,
        backend: BackendProtocol,
        config: ContextConfig,
        skills: ManagedSkillsMiddleware | None = None,
        excluded_tools: frozenset[str] = frozenset(),
    ) -> None:
        """Initialize policy and tools without reading history or backend contents."""
        self.config = config
        self.backend = backend
        self.skills = skills
        self.excluded_tools = excluded_tools
        # Reuse accounting, model limits, and summarization prompts, never its history rewriting hooks.
        self.accounting = SummarizationMiddleware(model, backend=backend, token_counter=config.token_counter, trim_tokens_to_summarize=None)
        self.tools = self._context_tools()

    def _count(self, request: ModelRequest) -> int:
        count = self.accounting._count_tokens(request.messages, request.system_message, request.tools)
        if request.tools and self.accounting._counter_accepts_tools is not True:
            schemas = json.dumps([convert_to_openai_tool(t) for t in request.tools])
            count += self.accounting._count_tokens([HumanMessage(content=schemas)], None, [])
        return count

    def _budget(self, request: ModelRequest) -> int:
        budget = self.config.input_budget or self.accounting._input_budget(request)
        if budget is None:
            msg = "Model input limit is unknown; pass context_management=ContextConfig(input_budget=...)."
            raise ContextBudgetError(msg)
        return budget

    def _filter(self, request: ModelRequest) -> ModelRequest:
        return request.override(tools=[t for t in request.tools if (t.name if isinstance(t, BaseTool) else t.get("name")) not in self.excluded_tools])

    def _render(
        self, request: ModelRequest, blocks: dict[str, ContextBlock], active: dict[str, SkillSnapshot]
    ) -> tuple[ModelRequest, dict[str, str]]:
        request = request.override(messages=project(request.messages, blocks))
        prepared, record = self.skills.prepare(request, active) if self.skills else (request, {})
        return self._filter(prepared), {name: source for name, source in record.items() if name not in self.excluded_tools}

    async def _arender(
        self, request: ModelRequest, blocks: dict[str, ContextBlock], active: dict[str, SkillSnapshot]
    ) -> tuple[ModelRequest, dict[str, str]]:
        request = request.override(messages=project(request.messages, blocks))
        prepared, record = await self.skills.aprepare(request, active) if self.skills else (request, {})
        return self._filter(prepared), {name: source for name, source in record.items() if name not in self.excluded_tools}

    @staticmethod
    def _ranges(messages: list[AnyMessage], blocks: dict[str, ContextBlock], excluded: set[str]) -> list[list[AnyMessage]]:
        eligible = eligible_refs(messages)
        ranges: list[list[AnyMessage]] = []
        previous: int | None = None
        for message in project(messages, blocks):
            block = blocks.get(message_ref(message))
            refs = block["refs"] if block else [message_ref(message)]
            tier = block["tier"] if block else 0
            if not set(refs) <= eligible or message_ref(message) in excluded or tier >= _MAX_TIER:
                previous = None
                continue
            if previous != tier:
                ranges.append([])
            ranges[-1].append(message)
            previous = tier
        return ranges

    def _candidate(self, messages: list[AnyMessage], blocks: dict[str, ContextBlock]) -> list[AnyMessage]:
        return max(self._ranges(messages, blocks, set()), key=self._history_tokens, default=[])

    def _history_tokens(self, messages: list[AnyMessage]) -> int:
        return self.accounting._count_tokens(messages, None, [])

    def _summary_tokens(self, messages: list[AnyMessage]) -> int:
        prompt = self.accounting._lc_helper._format_summary_prompt(messages)
        return self._history_tokens([HumanMessage(content=prompt)])

    def _summary_candidate(self, request: ModelRequest, blocks: dict[str, ContextBlock], budget: int, excluded: set[str]) -> list[AnyMessage]:
        # Parent input targets can be much smaller than the summarizer's actual capacity.
        limit = self.accounting._input_budget(request.override(model=self.accounting.model, model_settings={}))
        capacity = budget if limit is None else limit
        candidates: list[list[AnyMessage]] = []
        for messages in self._ranges(request.messages, blocks, excluded):
            selected: list[AnyMessage] = []
            for group in message_groups(messages):
                if self._summary_tokens([*selected, *group]) > capacity:
                    if selected:
                        candidates.append(selected)
                    selected = []
                    if self._summary_tokens(group) > capacity:
                        continue
                selected.extend(group)
            if selected:
                candidates.append(selected)
        return max(candidates, key=self._history_tokens, default=[])

    def _accept(self, messages: list[AnyMessage], blocks: dict[str, ContextBlock], start: str, end: str, summary: str) -> dict[str, ContextBlock]:
        block = collapse_range(messages, blocks, start, end, summary)
        updated = {**blocks, block["id"]: block}
        before = self.accounting._count_tokens(project(messages, blocks), None, [])
        after = self.accounting._count_tokens(project(messages, updated), None, [])
        if after >= before:
            msg = "Summary did not reduce context tokens; retained originals and projection are unchanged."
            raise ContextBudgetError(msg)
        return updated

    def _finish(
        self,
        original: ModelRequest,
        rendered: ModelRequest,
        blocks: dict[str, ContextBlock],
        active: dict[str, SkillSnapshot],
        record: dict[str, str],
        budget: int,
        notices: list[str],
    ) -> tuple[ModelRequest, dict[str, object]]:
        count = self._count(rendered)
        if count >= budget * _SOFT:
            candidate = self._candidate(original.messages, blocks)
            refs = f"{candidate[0].id} through {candidate[-1].id}" if candidate else "none; recent content is protected"
            notices.append(f"Context usage {count}/{budget}; eligible collapse range: {refs}.")
        if notices:
            rendered = rendered.override(system_message=append_to_system_message(rendered.system_message, "\n".join(notices)))
        if self._count(rendered) > budget:
            msg = "Context exceeds the input budget after at most two recovery attempts; reduce protected input, skills, or tool schemas."
            raise ContextBudgetError(msg)
        overhead = self._count(rendered) - self.accounting._count_tokens(rendered.messages, None, [])
        update: dict[str, object] = {"context_blocks": blocks, "context_input_budget": budget, "context_overhead": overhead}
        if self.skills:
            update.update(active_skills=active, _skill_tools_disclosed=record)
        return rendered, update

    @staticmethod
    def _evict(active: dict[str, SkillSnapshot], notices: list[str]) -> bool:
        name = next((name for name, snapshot in active.items() if not snapshot["pinned"]), None)
        if name is None:
            return False
        del active[name]
        notices.append(f"Skill {name} became inactive under context budget pressure; activate_skill can reload it.")
        return True

    def _prepare(self, request: ModelRequest) -> tuple[ModelRequest, dict[str, object]]:
        budget = self._budget(request)
        blocks = dict(request.state.get("context_blocks", {}))
        active = dict(request.state.get("active_skills", {}))
        notices: list[str] = []
        rendered, record = self._render(request, blocks, active)
        while self._count(rendered) >= budget * _COMPACT and self._evict(active, notices):
            rendered, record = self._render(request, blocks, active)
        excluded: set[str] = set()
        failure: ContextBudgetError | None = None
        for attempt in range(2):
            if self._count(rendered) < budget * (_COMPACT if attempt == 0 else _SOFT):
                break
            candidate = self._summary_candidate(request, blocks, budget, excluded)
            if not candidate:
                break
            summary = self.accounting._create_summary(candidate)
            try:
                blocks = self._accept(request.messages, blocks, message_ref(candidate[0]), message_ref(candidate[-1]), summary)
            except ContextBudgetError as error:
                failure = error
                excluded.update(message_ref(message) for message in candidate)
                continue
            failure = None
            rendered, record = self._render(request, blocks, active)
        if failure is not None and self._count(rendered) > budget:
            raise failure
        return self._finish(request, rendered, blocks, active, record, budget, notices)

    async def _aprepare(self, request: ModelRequest) -> tuple[ModelRequest, dict[str, object]]:
        budget = self._budget(request)
        blocks = dict(request.state.get("context_blocks", {}))
        active = dict(request.state.get("active_skills", {}))
        notices: list[str] = []
        rendered, record = await self._arender(request, blocks, active)
        while self._count(rendered) >= budget * _COMPACT and self._evict(active, notices):
            rendered, record = await self._arender(request, blocks, active)
        excluded: set[str] = set()
        failure: ContextBudgetError | None = None
        for attempt in range(2):
            if self._count(rendered) < budget * (_COMPACT if attempt == 0 else _SOFT):
                break
            candidate = self._summary_candidate(request, blocks, budget, excluded)
            if not candidate:
                break
            summary = await self.accounting._acreate_summary(candidate)
            try:
                blocks = self._accept(request.messages, blocks, message_ref(candidate[0]), message_ref(candidate[-1]), summary)
            except ContextBudgetError as error:
                failure = error
                excluded.update(message_ref(message) for message in candidate)
                continue
            failure = None
            rendered, record = await self._arender(request, blocks, active)
        if failure is not None and self._count(rendered) > budget:
            raise failure
        return self._finish(request, rendered, blocks, active, record, budget, notices)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ExtendedModelResponse:
        """Check the final synchronous request and checkpoint projection metadata."""
        request, update = self._prepare(request)
        return ExtendedModelResponse(model_response=handler(request), command=Command(update=update))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ExtendedModelResponse:
        """Check the final async request and checkpoint projection metadata."""
        request, update = await self._aprepare(request)
        return ExtendedModelResponse(model_response=await handler(request), command=Command(update=update))

    @staticmethod
    def _reply(runtime: ToolRuntime, text: str, blocks: dict[str, ContextBlock]) -> Command:
        return Command(update={"context_blocks": blocks, "messages": [ToolMessage(content=text, tool_call_id=runtime.tool_call_id)]})

    def _uncollapse_content(self, block_id: str, runtime: ToolRuntime, mode: str, *, full: bool) -> str | Command:
        if mode == "expand":
            serial_mutation(runtime, _MUTATIONS)
        blocks = runtime.state.get("context_blocks", {})
        if block_id not in blocks or mode not in {"recall", "expand"}:
            msg = "Unknown block or mode; mode must be recall or expand."
            raise ValueError(msg)
        messages = runtime.state["messages"]
        if mode == "expand":
            updated = expand_block(blocks, block_id, full=full)
            expanded = [*project(messages, updated), ToolMessage(content=f"Expanded {block_id}", tool_call_id=runtime.tool_call_id)]
            count = self.accounting._count_tokens(expanded, None, []) + runtime.state.get("context_overhead", 0)
            if count > runtime.state.get("context_input_budget", 0):
                msg = "Expansion exceeds the input budget; projection is unchanged. Use recall."
                raise ContextBudgetError(msg)
            return self._reply(runtime, f"Expanded {block_id}", updated)
        block = blocks[block_id]
        refs = set(block["refs"])
        retained = [m for m in messages if m.id in refs]
        if not full:
            descendants = {name: b for name, b in blocks.items() if name != block_id and set(b["refs"]) < refs}
            retained = project(retained, descendants)
        return json.dumps({"historical_data": True, "messages": messages_to_dict(retained)}, ensure_ascii=False)

    def _inline_recall(self, content: str, runtime: ToolRuntime) -> bool:
        if len(content) > _RECALL_LIMIT:
            return False
        budget = runtime.state.get("context_input_budget")
        if budget is None:
            return True
        messages = project(runtime.state["messages"], runtime.state.get("context_blocks", {}))
        recalled = [*messages, ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]
        return self.accounting._count_tokens(recalled, None, []) + runtime.state.get("context_overhead", 0) <= budget

    def _context_tools(self) -> list[BaseTool]:  # noqa: C901  # Tool definitions share this controller and one transaction reply helper.
        @tool
        def collapse(start_ref: str, end_ref: str, summary: str, runtime: ToolRuntime, topic: str | None = None) -> Command:
            """Collapse a complete contiguous message range or complete blocks into a reversible summary."""
            serial_mutation(runtime, _MUTATIONS)
            blocks = runtime.state.get("context_blocks", {})
            messages = runtime.state["messages"]
            updated = self._accept(messages, blocks, start_ref, end_ref, summary)
            block = next(b for name, b in updated.items() if name not in blocks)
            block["topic"] = topic
            return self._reply(runtime, block["id"], updated)

        def artifact_path(block_id: str, content: str) -> str:
            version = hashlib.sha256(content.encode()).hexdigest()[:16]
            return f"/context_recall/{block_id}-{version}.json"

        @tool
        async def uncollapse(block_id: str, runtime: ToolRuntime, mode: str = "recall", *, full: bool = False) -> str | Command:
            """Recall retained historical data without replaying tools, or expand in place if the request fits."""
            content = self._uncollapse_content(block_id, runtime, mode, full=full)
            if isinstance(content, Command) or self._inline_recall(content, runtime):
                return content
            path = artifact_path(block_id, content)
            existing = await self.backend.adownload_files([path])
            if not existing or existing[0].error:
                result = await self.backend.awrite(path, content)
                if result.error:
                    msg = f"Cannot store recall artifact: {result.error}"
                    raise ValueError(msg)
            return json.dumps({"historical_data": True, "result_ref": path, "characters": len(content)})

        def uncollapse_sync(block_id: str, runtime: ToolRuntime, mode: str = "recall", *, full: bool = False) -> str | Command:
            content = self._uncollapse_content(block_id, runtime, mode, full=full)
            if isinstance(content, Command) or self._inline_recall(content, runtime):
                return content
            path = artifact_path(block_id, content)
            existing = self.backend.download_files([path])
            if not existing or existing[0].error:
                result = self.backend.write(path, content)
                if result.error:
                    msg = f"Cannot store recall artifact: {result.error}"
                    raise ValueError(msg)
            return json.dumps({"historical_data": True, "result_ref": path, "characters": len(content)})

        uncollapse.func = uncollapse_sync

        @tool
        def search_context(query: str, runtime: ToolRuntime) -> str:
            """Search retained original text and summaries; return bounded snippets with usable references."""
            entries = [(m.id, m.text) for m in runtime.state["messages"]]
            entries.extend((b["id"], b["summary"]) for b in runtime.state.get("context_blocks", {}).values())
            matches = []
            for ref, text in entries:
                index = text.casefold().find(query.casefold())
                if index >= 0:
                    matches.append({"ref": ref, "snippet": text[max(0, index - 100) : index + 300]})
                if len(matches) == _SEARCH_LIMIT:
                    break
            return json.dumps(matches)

        @tool
        def context_status(runtime: ToolRuntime) -> str:
            """List visible message references, block hierarchy, and current input budget."""
            blocks = runtime.state.get("context_blocks", {})
            return json.dumps(
                {
                    "visible_refs": [m.id for m in project(runtime.state["messages"], blocks)],
                    "blocks": [{k: v for k, v in b.items() if k != "summary"} for b in blocks.values()],
                    "input_budget": runtime.state.get("context_input_budget"),
                }
            )

        return [collapse, uncollapse, search_context, context_status]
