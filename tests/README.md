# NovaAgents SDK Tests

Tests for the NovaAgents (`novaagents`) package, derived from Deep Agents.

## Running tests

Install test dependencies, then run:

```bash
make test
```

Or directly:

```bash
uv run --group test pytest tests -v
```

## Test areas

- Unit tests: `tests/unit_tests/` (includes Nova-specific graph/session/managed integration tests such as `test_novaagents.py`, `test_nova_session.py`, `test_nova_managed_integration.py`).
- Integration tests: `tests/integration_tests/` (may require model/provider credentials and network access).
- Benchmarks: `tests/benchmarks/` (e.g., Nova context benchmark).

Some tests require optional dependencies or API keys; see `pyproject.toml` dependency groups and `VALIDATION.md` for environment notes.
