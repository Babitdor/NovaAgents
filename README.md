# NovaAgents

NovaAgents (`novaagents`) is a production-oriented agent harness for building stateful AI agents with managed sessions, context management, skills, subagents, and long-term memory.

Requires Python 3.11 or newer. This repository contains the standalone Python SDK.

> **Provenance:** NovaAgents is derived from [Deep Agents](https://github.com/langchain-ai/deepagents) (LangChain). It keeps compatibility where practical, but it is an independent package with Nova-specific APIs and behavior.

## Installation

Install from PyPI (recommended, once published):

```cmd
pip install novaagents
```

Or add it to a project managed by `uv`:

```cmd
uv add novaagents
```

For local development, run from the repository root:

```cmd
uv sync --all-groups --locked
```

Optional extras:

```cmd
pip install "novaagents[aws]"
pip install "novaagents[video]"
```

The `aws` extra adds Bedrock support; `video` adds media helpers using PyAV and Pillow.

## Quickstart

Use `create_nova_agent` for the Nova graph. `create_deep_agent` is a separate compatibility factory with the inherited behavior.

Configure your model provider's credentials before running the example. This example uses `ANTHROPIC_API_KEY`.

```python
from novaagents import create_nova_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-4-6")

agent = create_nova_agent(
    model=model,
    system_prompt="You are a concise, reliable assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Summarize this repository."}]}
)
print(result["messages"][-1].content)
```

## Managed Sessions (Nova-specific)

`NovaSession` supervises local background tasks and serializes parent invocations. Pass it a compiled Nova graph:

```python
import asyncio
from novaagents import NovaSession

async def main():
    # Use the agent constructed in the quickstart above.
    async with NovaSession(agent, thread_id="demo-001") as session:
        result = await session.ainvoke({
            "messages": [{"role": "user", "content": "List files in / and summarize the project."}],
        })
        print(result["messages"][-1].content)
        await session.wait_until_idle()

asyncio.run(main())
```

See [MANAGED_SESSIONS.md](MANAGED_SESSIONS.md) for lifecycle, resume, and background-task behavior.
Persistence across restarts requires a persistent checkpointer and the appropriate backend. Local background workers live only for the current process.

## Nova Features

NovaAgents includes capabilities layered on the Deep Agents foundation:

- **`create_nova_agent`**: Nova graph assembly with Nova middleware defaults.
- **`NovaSession`**: local background task supervision and serialized conversation execution.
- **Context & managed skills middleware**: Nova-specific context loading/handling and skill behavior (see `novaagents.middleware`).
- **Subagents / delegation**: task decomposition via LangGraph subagent patterns.
- **Filesystem & stateful workflows**: tool-driven file operations and checkpointed agent state.
- **Long-term memory patterns**: memory-backed workflows depending on configured backend/tools.

Compatibility APIs from the upstream lineage may still exist; prefer Nova APIs for new integrations and verify behavior against this repository's tests/docs.

## Development

Run from the repository root. On Windows, use **Command Prompt (cmd)**; for this checkout:

```cmd
cd /d B:\nova-agent
uv sync --all-groups --locked
uv run --all-groups ruff check .
uv run --all-groups ruff format . --check
uv run --all-groups ty check novaagents
uv run --group test pytest tests/unit_tests -n auto --disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,::1 --benchmark-disable
```

Use your own checkout path if it differs. The loopback allowance supports Windows asyncio's internal socket pair while blocking external connections. With `make` installed, the equivalent targets are `make test`, `make lint`, and `make type`.

See [DEVELOPMENT.md](DEVELOPMENT.md) for other platforms and [VALIDATION.md](VALIDATION.md) for recorded checks and known limits.

## Repository layout

```text
novaagents/       SDK source, middleware, backends, and model profiles
tests/            Unit tests, integration tests, and benchmarks
examples/         Rubric middleware example
scripts/          Development utilities
pyproject.toml    Package metadata and dependencies
uv.lock           Locked development environment
```

Tests and development utilities stay in the repository. Release distributions contain the SDK, bundled resources, license, metadata, and public package documentation.

## Publish to PyPI

Create a [PyPI account](https://pypi.org/account/register/), verify your email, enable two-factor authentication, and create an API token. A GitHub repository URL is optional.

Build and check the packages from Command Prompt:

```cmd
cd /d B:\nova-agent
uv sync --all-groups --locked
uv build
uv run --group test python -m twine check dist/*
```

Run the development checks above, then review the package name, version, and files in `dist` before uploading. PyPI releases cannot be overwritten. When ready:

```cmd
uv run --group test python -m twine upload dist/*
```

Enter the API token at Twine's credential prompt. If asked for a username, use `__token__`. Keep tokens out of source files and shared messages.

After publication, verify installation independently of this checkout:

```cmd
uv run --no-project --with novaagents python -c "import novaagents; print(novaagents.__version__)"
```

Building and checking do not upload anything. See [PUBLISHING.md](PUBLISHING.md) for TestPyPI and release details.

## Security & Threat Model

See [THREAT_MODEL.md](THREAT_MODEL.md). NovaAgents inherits the general agent/tool risk profile of Deep Agents-derived systems: treat tool outputs, files, web content, and model outputs as untrusted inputs to policy decisions; enforce least privilege and human approval for sensitive actions.

Upstream guidance: [Deep Agents security policy](https://github.com/langchain-ai/deepagents?tab=security-ov-file).

## Changelog & License

- [CHANGELOG.md](CHANGELOG.md)
- [LICENSE](LICENSE) (MIT)

## Upstream Attribution

NovaAgents is derived from the Deep Agents project by LangChain:

- Upstream source: https://github.com/langchain-ai/deepagents
- Upstream docs: https://docs.langchain.com/oss/python/deepagents/overview

## Contributing

Contributions are welcome (features, fixes, docs, tests). Before opening changes, run available checks and record blocked validation honestly. If your work touches upstream-derived behavior, keep Nova-specific intent and compatibility clear; upstream project context: [Deep Agents](https://github.com/langchain-ai/deepagents).
