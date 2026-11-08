# NovaAgents SDK development

This repository contains only the NovaAgents Python SDK. Start with
[DEVELOPMENT.md](DEVELOPMENT.md), [ARCHITECTURE.md](ARCHITECTURE.md), and
[PUBLISHING.md](PUBLISHING.md).

- Runtime code lives in `novaagents/`; tests in `tests/`; the SDK example in `examples/`.
- Work from the repository root with `uv sync --all-groups` and the Makefile targets.
- Use `rg` for focused searches; exclude `.venv`, generated metadata, and caches.
- Preserve public exports and signatures. Add optional parameters as keyword-only
  defaults, document them, and flag experimental behavior in public docstrings.
- Use type hints, descriptive names, Google-style docstrings, and American English.
- Keep dependencies minimal. Justify additions and preserve supported Python versions.
- Add deterministic tests for meaningful observable behavior. Network-free tests
  belong in `tests/unit_tests/`; provider tests in `tests/integration_tests/`.
- Do not add `pytest.mark.asyncio`; asyncio auto mode is configured.
- Warnings fail the test suite. Fix actionable warnings; keep expected-warning
  filters narrow. Do not suppress individual Ruff violations file-wide.
- Preserve authorization, path containment, tool pairing, exact retained history,
  cancellation cleanup, and session isolation. Never deserialize untrusted pickle
  or execute user-controlled Python. Check for races and resource leaks.
- If Corridor `analyzePlan` is available, analyze the plan before editing code and
  apply its security guidance.
- Preserve the MIT license and upstream attribution. Do not commit credentials,
  runtime histories, build output, or local virtual environments.
- Use Conventional Commits. Document behavior changes and actual validation.
- Publishing is an immutable external release: build and review artifacts first.
