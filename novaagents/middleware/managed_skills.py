"""Explicit skill lifecycle backed by immutable, checkpointed snapshots."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import ModelRequest, ModelResponse, PrivateStateAttr
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite

from novaagents.middleware._skill_tools import (
    _acall_resolver,
    _bind_disclosures,
    _call_resolver,
    _discard_rejected_schemas,
    _plan_disclosure,
    _unclaimed_include_names,
)
from novaagents.middleware._utils import append_to_system_message
from novaagents.middleware.skills import (
    SkillMetadata,
    SkillsMiddleware,
    SkillsState,
    _parse_skill_metadata,
    _pinned_skill_body,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.tools import BaseTool

    from novaagents.backends.protocol import BackendProtocol, FileDownloadResponse
    from novaagents.middleware.skills import SkillSource, SkillsStateUpdate, SkillToolResolver

_MUTATIONS = {"collapse", "activate_skill", "deactivate_skill", "refresh_skills"}
_MANAGED_PROMPT = """## Managed skills
{skills_locations}{skills_load_warnings}

{skills_list}

Search with search_skills, then activate_skill to load complete instructions and
associated tools. Active instructions below remain loaded across compaction.
deactivate_skill removes them. refresh_skills reports changed sources; deactivate
and reactivate to adopt a new version. Use read_file for supporting files.
"""


class SkillSnapshot(TypedDict):
    """Validated instructions and tool metadata at one content version."""

    metadata: SkillMetadata
    body: str
    hash: str
    pinned: bool


class ManagedSkillsState(SkillsState):
    """Skill snapshots belong to one conversation, never to a middleware instance."""

    active_skills: NotRequired[Annotated[dict[str, SkillSnapshot], PrivateStateAttr]]
    skill_notices: NotRequired[Annotated[list[str], PrivateStateAttr]]


def serial_mutation(runtime: ToolRuntime, names: set[str]) -> None:
    """Refuse concurrent lifecycle mutations before either can update state."""
    messages = runtime.state.get("messages", [])
    calls = getattr(messages[-1], "tool_calls", []) if messages else []
    if sum(call["name"] in names or (call["name"] == "uncollapse" and call["args"].get("mode") == "expand") for call in calls) > 1:
        msg = "Call lifecycle mutation tools sequentially, one per assistant turn."
        raise ValueError(msg)


def _require_discovery(update: SkillsStateUpdate | None) -> SkillsStateUpdate:
    if update is None:
        msg = "Skill discovery did not return metadata."
        raise RuntimeError(msg)
    return update


class ManagedSkillsMiddleware(SkillsMiddleware):
    """Reuse source discovery and tool gates with explicit activation snapshots."""

    state_schema = ManagedSkillsState

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        sources: Sequence[SkillSource],
        system_prompt: str | None = _MANAGED_PROMPT,
        tools: Sequence[BaseTool | Callable[..., Any]] | SkillToolResolver | None = None,
    ) -> None:
        """Configure lifecycle management over existing backend sources.

        Args:
            backend: Backend containing the skill sources.
            sources: Ordered configured roots; later roots win duplicate names.
            system_prompt: Discovery prompt with the same slots as `SkillsMiddleware`.
            tools: Existing permitted skill tool collection or resolver.
        """
        super().__init__(backend=backend, sources=sources, system_prompt=system_prompt, tools=tools)
        self.tools = self._lifecycle_tools()

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        return "\n".join(f"- {s['name']}: {s['description']} (activate_skill; source {s['path']})" for s in skills) or "No skills discovered."

    def _snapshot(self, skill: SkillMetadata, response: FileDownloadResponse, *, pinned: bool) -> SkillSnapshot:
        body = _pinned_skill_body(skill, response)
        if body is None or not response.content:
            msg = f"Cannot activate skill {skill['name']}: unreadable, empty, oversized, or invalid UTF-8 source."
            raise ValueError(msg)
        content = response.content.decode("utf-8")
        metadata = _parse_skill_metadata(content, skill["path"], skill["name"])
        if metadata is None or metadata["name"] != skill["name"]:
            msg = f"Cannot activate skill {skill['name']}: invalid or renamed source."
            raise ValueError(msg)
        return SkillSnapshot(metadata=metadata, body=body, hash=hashlib.sha256(response.content).hexdigest(), pinned=pinned)

    @staticmethod
    def _find(metadata: Sequence[SkillMetadata], name: str) -> SkillMetadata:
        skill = next((s for s in metadata if s["name"] == name), None)
        if skill is None:
            msg = f"Unknown skill {name!r}; use search_skills or refresh_skills."
            raise ValueError(msg)
        return skill

    def _load(self, metadata: Sequence[SkillMetadata], name: str, *, pinned: bool) -> SkillSnapshot:
        skill = self._find(metadata, name)
        return self._snapshot(skill, self._backend.download_files([skill["path"]])[0], pinned=pinned)

    async def _aload(self, metadata: Sequence[SkillMetadata], name: str, *, pinned: bool) -> SkillSnapshot:
        skill = self._find(metadata, name)
        responses = await self._backend.adownload_files([skill["path"]])
        return self._snapshot(skill, responses[0], pinned=pinned)

    def before_model(self, state: SkillsState, runtime: Runtime) -> dict[str, object] | None:  # noqa: ARG002
        """Load explicitly pinned names transactionally, failing visibly on errors."""
        names = state.get("pinned_skills") or []
        if not names:
            return None
        active = dict(cast("ManagedSkillsState", state).get("active_skills", {}))
        for name in names:
            active[name] = {**active[name], "pinned": True} if name in active else self._load(state.get("skills_metadata") or [], name, pinned=True)
        return {"active_skills": active, "pinned_skills": Overwrite([])}

    async def abefore_model(self, state: SkillsState, runtime: Runtime) -> dict[str, object] | None:  # noqa: ARG002
        """Load pinned names without blocking the event loop."""
        names = state.get("pinned_skills") or []
        if not names:
            return None
        active = dict(cast("ManagedSkillsState", state).get("active_skills", {}))
        for name in names:
            active[name] = (
                {**active[name], "pinned": True} if name in active else await self._aload(state.get("skills_metadata") or [], name, pinned=True)
            )
        return {"active_skills": active, "pinned_skills": Overwrite([])}

    def _lifecycle_tools(self) -> list[BaseTool]:  # noqa: C901  # Related tool definitions share transactional reply handling.
        def reply(runtime: ToolRuntime, text: str, **updates: object) -> Command:
            return Command(update={**updates, "messages": [ToolMessage(content=text, tool_call_id=runtime.tool_call_id)]})

        @tool
        def search_skills(query: str, runtime: ToolRuntime) -> str:
            """Search configured skill names and descriptions; returns at most 20 metadata entries."""
            matches = [s for s in runtime.state.get("skills_metadata") or [] if query.casefold() in f"{s['name']} {s['description']}".casefold()]
            return json.dumps(matches[:20])

        @tool
        async def activate_skill(name: str, runtime: ToolRuntime) -> Command:
            """Activate complete validated instructions; repeated activation preserves the snapshot."""
            serial_mutation(runtime, _MUTATIONS)
            active = dict(runtime.state.get("active_skills", {}))
            if name not in active:
                active[name] = await self._aload(runtime.state.get("skills_metadata") or [], name, pinned=False)
            return reply(runtime, f"Active: {name} ({active[name]['hash']})", active_skills=active)

        @tool
        def deactivate_skill(name: str, runtime: ToolRuntime) -> Command:
            """Deactivate a skill, including an explicitly pinned skill, and hide its associated tools."""
            serial_mutation(runtime, _MUTATIONS)
            active = dict(runtime.state.get("active_skills", {}))
            active.pop(name, None)
            return reply(runtime, f"Inactive: {name}", active_skills=active)

        @tool
        async def refresh_skills(runtime: ToolRuntime) -> Command:
            """Reload discovery and report changed or missing sources; preserve active snapshots."""
            serial_mutation(runtime, _MUTATIONS)
            update = _require_discovery(
                await self.abefore_agent(SkillsState(messages=runtime.state["messages"], skills_metadata=None), Runtime(), {})
            )
            notices: list[str] = []
            for name, snapshot in runtime.state.get("active_skills", {}).items():
                try:
                    current = await self._aload(update["skills_metadata"], name, pinned=snapshot["pinned"])
                except ValueError as error:
                    notices.append(str(error))
                else:
                    if current["hash"] != snapshot["hash"] or current["metadata"]["path"] != snapshot["metadata"]["path"]:
                        notices.append(f"Source changed: {name}; deactivate and reactivate to update.")
            return reply(runtime, json.dumps({"notices": notices, "errors": update.get("skills_load_errors", [])}), **update, skill_notices=notices)

        def activate_sync(name: str, runtime: ToolRuntime) -> Command:
            serial_mutation(runtime, _MUTATIONS)
            active = dict(runtime.state.get("active_skills", {}))
            if name not in active:
                active[name] = self._load(runtime.state.get("skills_metadata") or [], name, pinned=False)
            return reply(runtime, f"Active: {name} ({active[name]['hash']})", active_skills=active)

        def refresh_sync(runtime: ToolRuntime) -> Command:
            serial_mutation(runtime, _MUTATIONS)
            update = _require_discovery(self.before_agent(SkillsState(messages=runtime.state["messages"], skills_metadata=None), Runtime(), {}))
            notices: list[str] = []
            for name, snapshot in runtime.state.get("active_skills", {}).items():
                try:
                    current = self._load(update["skills_metadata"], name, pinned=snapshot["pinned"])
                except ValueError as error:
                    notices.append(str(error))
                else:
                    if current["hash"] != snapshot["hash"] or current["metadata"]["path"] != snapshot["metadata"]["path"]:
                        notices.append(f"Source changed: {name}; deactivate and reactivate to update.")
            return reply(runtime, json.dumps({"notices": notices, "errors": update.get("skills_load_errors", [])}), **update, skill_notices=notices)

        activate_skill.func = activate_sync
        refresh_skills.func = refresh_sync
        return [search_skills, activate_skill, deactivate_skill, refresh_skills]

    def render(self, request: ModelRequest, active: dict[str, SkillSnapshot]) -> ModelRequest:
        """Render each active snapshot once, regardless of compacted history."""
        request = self.modify_request(request)
        parts = [
            f"<active_skill name={json.dumps(name)} source={json.dumps(s['metadata']['path'])} "
            f"hash={json.dumps(s['hash'])}>\n{s['body']}\n</active_skill>"
            for name, s in active.items()
        ]
        parts.extend(request.state.get("skill_notices", []))
        return request.override(system_message=append_to_system_message(request.system_message, "\n".join(parts))) if parts else request

    def disclose(
        self, request: ModelRequest, active: dict[str, SkillSnapshot], resolved: dict[str, list[BaseTool]]
    ) -> tuple[ModelRequest, dict[str, str]]:
        """Use existing schema validation and gates, with activation as the disclosure anchor."""
        reads = [(0, snapshot["metadata"]) for snapshot in active.values()]
        disclosure = _plan_disclosure(reads, request.tools, resolved)
        _discard_rejected_schemas(disclosure, request.model)
        return request.override(tools=_bind_disclosures(request.tools, disclosure)), disclosure.record()

    def prepare(self, request: ModelRequest, active: dict[str, SkillSnapshot]) -> tuple[ModelRequest, dict[str, str]]:
        """Resolve active tool schemas for the controller's final budget check."""
        reads = [(0, snapshot["metadata"]) for snapshot in active.values()]
        names = _unclaimed_include_names(reads, request.tools)
        resolved = {name: _call_resolver(self._skill_tool_resolver, name, request.runtime) for name in names}
        return self.disclose(self.render(request, active), active, resolved)

    async def aprepare(self, request: ModelRequest, active: dict[str, SkillSnapshot]) -> tuple[ModelRequest, dict[str, str]]:
        """Resolve asynchronous skill tool providers."""
        reads = [(0, snapshot["metadata"]) for snapshot in active.values()]
        names = _unclaimed_include_names(reads, request.tools)
        resolved = {name: await _acall_resolver(self._skill_tool_resolver, name, request.runtime) for name in names}
        return self.disclose(self.render(request, active), active, resolved)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """Let the context controller coordinate rendering and disclosure."""
        return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        """Let the context controller coordinate async rendering and disclosure."""
        return await handler(request)
