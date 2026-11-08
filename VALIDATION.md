# SDK packaging validation

Verified 2026-11-08 on Windows using a fresh repository-root environment with
Python 3.11.13. The SDK previously lived under `libs/nova-agents`; all maintained
SDK source, tests, scripts, public documentation, and the test-dependent rubric
example now live at the repository root.

## Checks

| Check | Result |
| --- | --- |
| `uv sync --all-groups --locked` | Passed in the new root environment. |
| `uv lock --check` | Passed; no dependency additions or upgrades. |
| Full network-free unit suite | **3,603 passed, 239 skipped, 3 xfailed**. |
| `ruff check .` | Passed. |
| `ruff format . --check` | Passed; 159 Python files. |
| `ty check novaagents` | No type errors; exits 1 for two inherited Windows `url2pathname` deprecation warnings. |
| `uv build` | Wheel and source archive built successfully. |
| `python -m twine check dist/*` | Both distributions passed. |
| Archive content inspection | SDK resources and metadata only; no tests, examples, research, environments, histories, or caches. |
| Installed-wheel smoke check | Installed the wheel without reinstalling dependencies, verified public imports from `site-packages` using isolated Python mode, then restored the editable development install with `uv sync --all-groups --locked`. |

The unit command used `-n 4 --disable-socket --allow-unix-socket
--allow-hosts=127.0.0.1,::1 --benchmark-disable --timeout=45 -q --tb=short`.
Loopback access permits Windows asyncio's internal socket pair; external network
connections remain blocked. No live-provider integration or benchmark rerun was
performed in this packaging pass.

Fresh Python 3.11 validation exposed two portability issues, now fixed and
covered: zero-timeout grep uses an inclusive deadline check, and Markdown file
attachments have a standard MIME fallback when the platform table omits `.md`.
The required refusal middleware now fails with an actionable dependency error
when missing, instead of returning `None` to a required graph-construction call.

## Release artifacts

- `dist/novaagents-0.8.0-py3-none-any.whl`
- `dist/novaagents-0.8.0.tar.gz`

MIT license metadata uses SPDX syntax and includes the original LangChain
copyright notice. Project metadata links to
[Babitdor/NovaAgents](https://github.com/Babitdor/NovaAgents).

The `novaagents` project lookup returned HTTP 404 during preparation. That does
not reserve the name or guarantee account access. No TestPyPI or PyPI upload was
performed. See [PUBLISHING.md](PUBLISHING.md) for the upload procedure.

## Cleanup and rollback

Unrelated apps, ACP/Talon, partner packages, evals, research, generated OpenWiki,
agent-local configuration, runtime data, and scratch/build caches were removed
from this project by moving them outside it. Tests and developer tools remain
in the source repository but are excluded from release distributions.

- Verified source backup: `B:/nova-agent-source-backup-20261108-213721.zip`.
- Removed files and validation logs: `B:/nova-agent-removed-20261108-2140/`.
- The old `libs/nova-agents` directory is empty but remains held open by Windows.
  Close any terminal/editor using that path before removing the empty directory.
- Automatic approval review rejected recursive deletion with the generic reason
  `blocked by policy`; cleanup therefore preserves removed content in the sibling
  archive instead of deleting it.

The local `.venv` is retained for SDK development and is ignored by version
control and excluded from both release artifacts. Historical research and
validation reports remain recoverable in the sibling archive.
