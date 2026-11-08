# Publishing NovaAgents to PyPI

The distribution and import name are both `novaagents`. Build from the repository
root. The existing version is `0.8.0`; keep `pyproject.toml` and
`novaagents/_version.py` synchronized. An uploaded PyPI release cannot be replaced.

## Prepare and verify

```bash
uv sync --all-groups --locked
make test
make lint
uv build
uv run --group test python -m twine check dist/*
```

Review both the wheel and source archive. Only the SDK, bundled resources,
license, packaging metadata, and public package documentation should be present.
Tests and development files stay in this repository, not in the distributions.
Never upload credentials, runtime state, local environments, or research data.

The MIT license retains LangChain's copyright and upstream attribution.
Project metadata links to [Babitdor/NovaAgents](https://github.com/Babitdor/NovaAgents).
Update `[project.urls]` if the repository moves or dedicated documentation becomes available.

## Upload

Create or sign in to a PyPI account, enable two-factor authentication, and set up
a project-scoped API token or trusted publishing. The package name must be
available to that account; a 404 lookup does not guarantee registration.

Use Twine's credential prompt or environment variables from your secret manager.
The token username is `__token__`. Do not put tokens in source files, command
arguments, screenshots, or version control.

Optional rehearsal with separate TestPyPI credentials:

```bash
uv run --group test python -m twine upload --repository testpypi dist/*
```

After reviewing the exact artifacts and confirming the version, upload to PyPI:

```bash
uv run --group test python -m twine upload dist/*
```

Verify installation in a clean environment after publication:

```bash
uv run --no-project --with novaagents python -c "import novaagents; print(novaagents.__version__)"
```

No upload is performed by building or checking the package. See
[VALIDATION.md](VALIDATION.md) for the recorded local checks and remaining limits.
