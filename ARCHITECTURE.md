# NovaAgents SDK architecture

NovaAgents assembles LangChain agents on the LangGraph runtime. It retains the
Deep Agents lineage and compatibility factory while adding managed sessions,
skill lifecycle management, and reversible context projection.

| Area | Entry point |
| --- | --- |
| Public exports | `novaagents/__init__.py` |
| Compiled graph factories | `novaagents/graph.py` |
| Local task/session supervision | `novaagents/session.py` |
| Context projection and budgets | `novaagents/middleware/context.py` |
| Validated reversible blocks | `novaagents/middleware/_context_blocks.py` |
| Managed skill snapshots | `novaagents/middleware/managed_skills.py` |
| Local and remote subagents | `novaagents/middleware/local_tasks.py`, `async_subagents.py`, `subagents.py` |
| Filesystem and persistence adapters | `novaagents/backends/` |
| Model/provider configuration | `novaagents/profiles/` |

`create_nova_agent` returns a compiled graph. `NovaSession` supplies the lifetime
needed for local background work; task execution resources are not checkpointed.
Canonical graph history retains originals; context blocks change the model-facing
projection. Persistence across restarts depends on the caller's checkpointer and
backend. Local workers do not survive process termination.

`create_deep_agent` remains the compatibility factory. See
[MANAGED_SESSIONS.md](MANAGED_SESSIONS.md) for Nova lifecycle and context details,
and [THREAT_MODEL.md](THREAT_MODEL.md) for tool-boundary constraints.
