# Repository Guidelines

## Project Structure & Module Organization
`codex_proxy/` holds the runtime code. Keep CLI entrypoints in `codex_proxy/cli.py`, HTTP server logic in `codex_proxy/server.py`, request/response mapping in `codex_proxy/translator.py`, the upstream model catalog in `codex_proxy/models.py`, and account or auth state in `codex_proxy/accounts.py`, `codex_proxy/auth.py`, and `codex_proxy/config.py`. Put tests in `tests/` with names like `test_server.py` and `test_auth.py`. `README.md` is the user-facing setup guide; update it when commands or behavior change.

## Build, Test, and Development Commands
Use Python 3.11+ and the repo-local virtual environment in `.venv`.

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```
Activates the local environment and installs the package in editable mode with test and lint tools. If `.venv` does not exist yet, create it with `python -m venv .venv`.

```bash
codex-proxy login
codex-proxy models
codex-proxy serve -p 9000
```
Logs into ChatGPT, prints the live model catalog, and starts the local proxy for manual checks.

```bash
pytest
pytest tests/test_server.py
ruff check .
```
Run the full suite, a focused test file, or the linter before opening a PR.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints on new or changed functions, and short module docstrings where helpful. Use `snake_case` for files, functions, and variables; keep Click command names short and action-based, such as `login`, `serve`, and `switch`. Ruff is the formatting gate here, with a 100-character line target.

## Testing Guidelines
Tests use `pytest` and `pytest-asyncio`. Add or update tests whenever you touch routing, auth, account switching, usage lookup, or translation logic. Prefer small, direct test cases that match the current pattern: one behavior per test, realistic request payloads, and explicit assertions on normalized output. Name new files `test_<area>.py` and new tests `test_<expected_behavior>()`.

## Commit & Pull Request Guidelines
Recent history mixes plain imperative subjects (`Fix login callback port handling`) with lightweight prefixes (`feat:`, `fix:`). Keep commit titles short, imperative, and specific to one change. PRs should explain what changed, how you verified it (`pytest`, `ruff check .`, manual `codex-proxy serve` smoke test), and call out any user-visible API or auth changes. Include example requests or screenshots only when behavior at the CLI or HTTP layer changed.

## Security & Configuration Tips
Never commit real tokens, account snapshots, or anything from `~/.codex-proxy/`. When testing auth or usage flows, prefer local environment variables such as `CODEX_PROXY_API_KEY` and `HTTPS_PROXY` instead of hardcoding secrets.
