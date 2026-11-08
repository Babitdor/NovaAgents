> **NovaAgents** (`novaagents`) managed sessions. Derived from Deep Agents patterns; behavior and APIs may differ from upstream.

# NovaAgents Managed Sessions

`create_nova_agent` returns a compiled LangGraph graph with reversible context,
managed skills, and local background task tools. `create_deep_agent` keeps its
inherited middleware and signatures. The factories are distinct.

```python
from langgraph.checkpoint.memory import InMemorySaver
from novaagents import ContextConfig, NovaSession, create_nova_agent

# configured_model is your application's chat model instance.
agent = create_nova_agent(
    model=configured_model,
    checkpointer=InMemorySaver(),
    context_management=ContextConfig(input_budget=32000),
    skills=["/skills/"],
    backend=configured_backend,
)

async with NovaSession(agent, max_concurrency=4) as session:
    answer = await session.ainvoke({
        "messages": [{"role": "user", "content": "Delegate the independent research and continue the analysis."}],
        "pinned_skills": ["research"],
    })
    await session.wait_until_idle()
```

Supply an existing configured backend containing the skill sources. Skill
discovery stays inside those sources; this API does not install skills.

`context_management=None` enables the default controller. It derives an input
budget from the request model's advertised input limit, reserving configured
output tokens and headroom. A model without a usable input limit requires
`ContextConfig(input_budget=...)`. The controller includes system instructions,
active skills, messages, media, and tool schemas in accounting. Token estimates
have the same provider limitations as the existing token counter. A custom
counter may accept `tools=`; otherwise schemas are counted as serialized text.

## Graph execution and session lifetime

Direct `agent.invoke` and `agent.ainvoke` support context and skill tools. Local
background starts require `await session.ainvoke(...)` inside an open session;
otherwise they return an actionable session error. `task` still supports
synchronous delegation, and existing remote async subagents keep their remote
execution path.

The session owns a bounded queue and child task handles. The five local task
tools are `start_async_task`, `check_async_task`, `list_async_tasks`,
`wait_async_task`, and `cancel_async_task`. Starts return immediately with a task
ID. Results are delivered as status notifications with backend artifact
references; large results do not enter the parent conversation. Children have
separate graph execution threads and independent skill/context state. Declare
child `skills` sources and, optionally, `pinned_skills` in the subagent spec.
The parent's active snapshots are never assigned to a background child.

`session.events()` is an async iterator of immediate `task_completed` events
(task ID, status, result reference, and error) and unattended `parent_error`
events. Artifacts become readable when the parent accepts the completion at a
graph boundary. Events can be consumed concurrently with `ainvoke` or
`wait_until_idle`. `wait_until_idle` waits for local children and automatic
parent resumptions, propagating unattended parent errors. It does not poll
remote deployments or answer human approval interrupts.

The session serializes parent runs and batches queued completions. An active
parent receives notifications before its next model call; an idle parent is
automatically invoked. A parent paused for approval remains paused until the
caller supplies the appropriate LangGraph `Command(resume=...)` through
`session.ainvoke`. Stable notification IDs and checkpointed delivery records
prevent repeated completion messages.

Use one session per thread and route every parent invocation on that thread
through it. Concurrent sessions over the same graph/checkpointer and thread are
rejected. Without a checkpointer, the session retains conversation state only
for its own lifetime. Persistence across sessions/restarts requires the
application's configured checkpointer and backend.

Closing stops new work, cancels queued/running local children, and awaits their
cleanup. Tools already performing external side effects can only cancel
cooperatively. Local workers do not survive process exit; restored queued or
running records become `interrupted`. This is not a durable worker service.

## Managed skills

| Tool | Behavior |
| --- | --- |
| `search_skills(query)` | Search names/descriptions in configured roots; bounded metadata results. |
| `activate_skill(name)` | Validate and load the complete source as a versioned snapshot. Repeated activation keeps that snapshot. |
| `deactivate_skill(name)` | Remove instructions and associated tool disclosure, including explicitly pinned skills. |
| `refresh_skills()` | Reload discovery; report changed/missing active sources without replacing snapshots. |

Later sources win duplicate names. Unknown, unreadable, oversized, renamed, or
invalid explicitly activated/pinned skills fail visibly. Snapshots retain the
source, SHA-256 hash, validated metadata, and full instruction body.

Pass `pinned_skills` in graph input to pin skills explicitly. Pinned skills stay
active until deactivated. Agent-selected skills may become inactive under
budget pressure, oldest activation first; the next model request includes a
notice and may reactivate them. Active instructions are rendered once per
request independently of historical file reads, so compaction cannot remove
them. Skill tools use the existing resolver, schema validation, exclusions,
and execution permission gates. Reading a file alone does not activate a skill
or disclose its associated tools in Nova mode.

## Reversible context

Use `context_status()` to obtain message/block references, then
`collapse(start_ref, end_ref, summary, topic=None)`. It validates a complete
contiguous range and requires a measurable token reduction. T1 summarizes
messages, T2 summarizes complete T1 blocks, and T3 summarizes complete T2 blocks.
Every block retains references to canonical originals and its child blocks.

`uncollapse(block_id)` retrieves retained messages as historical JSON data while
keeping the projection collapsed. `full=True` follows the hierarchy to original
messages. Bounded results are returned inline; larger recalls become backend
artifacts. Restored tool calls are data and are never executed.

`uncollapse(block_id, mode="expand", full=False)` restores the projection in
place only if the estimated full request fits the last established budget.
Refusal leaves metadata unchanged. `search_context(query)` searches originals
and summaries and returns bounded snippets with references.

System instructions, pinned skills, the latest user turn, unresolved tool calls,
and the five most recent complete message groups are protected. Broken tool
pairs, partial overlaps, unknown/hidden references, mixed hierarchy tiers, and
cross-thread block IDs are rejected. Call lifecycle mutation tools sequentially
to keep each change transactional.

At 70% usage, the controller recommends an eligible range. At 85%, it deactivates
eligible skills and summarizes older eligible history toward 70%, using the
configured model and the same range validation. It makes at most two summary
attempts, preferring larger complete eligible ranges. Summary inputs include
prompt overhead and use the summarizer model's input limit, reserving output and
headroom. When that limit is unknown, the configured parent input budget is the
fallback. A smaller parent budget therefore does not artificially restrict a
model with a known larger capacity.

A non-shrinking summary is discarded; the remaining attempt may try a different
range. Tool-call groups are never split, and groups too large for the summarizer
are skipped while other ranges are considered. A request that still exceeds the
hard budget raises `ContextBudgetError`. Protected content is not dropped.
The inherited summarizer and Nova controller cannot run together.

## Validation

From this package directory, run `make test`, `make lint`, and `make type`.
`make bench` runs the existing benchmark target, including Nova's synthetic
projection comparison. Provider-free tests cover the controller, skill
lifecycle, session supervision, and their graph integration.

No performance or retrieval-quality improvement is claimed without measurements.
The synthetic comparison checks token reduction and exact retained-detail
retrieval against the inherited summary projection; it does not measure the
quality of model-generated summaries on real workloads.
