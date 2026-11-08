# SDK development

Run commands from the repository root. `uv` manages the Python environment and
locked dependencies; do not create a separate environment in another directory.

```bash
uv sync --all-groups --locked
make test
make lint
make type
make bench
```

The Makefile defines the standard test, formatting, type, and benchmark targets.
If `make` is unavailable, use these equivalents:

```bash
uv run --all-groups ruff check .
uv run --all-groups ruff format . --check
uv run --all-groups ty check novaagents
uv run --group test pytest tests/unit_tests -n auto --disable-socket --allow-unix-socket --benchmark-disable
```

On Windows, append `--allow-hosts=127.0.0.1,::1` to the unit-test command for
asyncio's internal socket pair. External network connections remain blocked.
Use `make integration_test` for provider-dependent checks with configured keys.

Tests, benchmarks, scripts, and examples remain in the source repository for
maintenance, but are excluded from release artifacts by `MANIFEST.in` and
explicit package discovery. Runtime code and bundled typing/prompt resources
live under `novaagents/`. Local `.venv`, caches, and `dist/` are ignored.

## Warnings fail the suite

Fix actionable warnings before adding filters. Scope expected warnings to the
specific test with `pytest.mark.filterwarnings`; use package-level filters only
for a documented category or third-party dependency. Do not weaken warnings or
lint rules merely to make checks pass.

## Verification

Run the smallest relevant regression tests, then the applicable package gates.
Record actual results and limitations. Changes to release packaging should
build both archives, check metadata, inspect contents, and smoke-test the wheel.
See [PUBLISHING.md](PUBLISHING.md).
