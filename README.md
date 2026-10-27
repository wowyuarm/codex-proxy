# codex-proxy

Local proxy that exposes ChatGPT Codex models as an OpenAI-compatible API, using your ChatGPT Plus/Pro subscription quota.

It also exposes a minimal Anthropic Messages API shim so Claude Code can use the same local proxy and still spend your ChatGPT Codex quota.

```
Client (curl, CLINE, aider, ...) ──POST /v1/chat/completions──▶ codex-proxy ──▶ chatgpt.com/backend-api/codex/responses
              ◀── OpenAI SSE ──────────────────────────────────────────────────◀── Responses API SSE

Claude Code ──POST /v1/messages──▶ codex-proxy ──▶ chatgpt.com/backend-api/codex/responses
            ◀── Anthropic SSE ────────────────────────────────────────────────◀── Responses API SSE
```

One Python package, no database, no container, no background daemon. Read it, run it, change it.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Login (opens browser for ChatGPT OAuth)
codex-proxy login

# See which models the account can actually use
codex-proxy models

# Start proxy
codex-proxy serve                # default: 0.0.0.0:8787
codex-proxy serve -p 9000        # custom port

# List saved accounts with usage
codex-proxy accounts

# Switch active account by account id and show its usage
codex-proxy switch <account-id>

# Remove one saved account by account id
codex-proxy accounts --remove <account-id>
```

## Authentication

### Downstream (clients → proxy)

When the `CODEX_PROXY_API_KEY` environment variable is set, all requests (except `/health`) must include a valid API key:

```
Authorization: Bearer <key>
# or
X-API-Key: <key>
```

If `CODEX_PROXY_API_KEY` is not set, the proxy runs without authentication (suitable for localhost-only use).

### Upstream (proxy → ChatGPT)

Upstream authentication is handled automatically via ChatGPT OAuth tokens stored in `~/.codex-proxy/credentials.json`. When you save multiple logins, snapshots are also kept under `~/.codex-proxy/accounts/`, and `codex-proxy switch <label|email|account_id>` updates the active `credentials.json`.

Run `codex-proxy login` to add the current account, `codex-proxy accounts` to list all saved accounts with usage and account ids, `codex-proxy switch <account-id>` to change the active account, and `codex-proxy accounts --remove <account-id>` to delete one saved account.

## Network Access

The proxy listens on `0.0.0.0:8787` by default, making it accessible from:

| Source | Address | Notes |
|--------|---------|-------|
| Localhost | `http://localhost:8787` | local tools and agents |
| Tailscale | `http://100.x.x.x:8787` | Remote machines on the same tailnet |

> **Security**: API key authentication + Tailscale ACLs + host firewall. The proxy does not need to be exposed to the public internet.

## Proxy Support

If you're in a region that requires a proxy to access OpenAI services, set `HTTPS_PROXY` before running commands:

```bash
export HTTPS_PROXY=http://127.0.0.1:7890

codex-proxy login    # proxy used for token exchange
codex-proxy serve    # proxy used for upstream API calls
```

## Usage

Once the proxy is running, point any OpenAI-compatible client at it:

```bash
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer codex-proxy" \
  -d '{"model":"gpt-6.1-sol","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

### Available Models

The list is not hardcoded. On startup and whenever the cache expires (5 minutes), the proxy asks the Codex backend which models the logged-in account may call, so new models appear without a release. `codex-proxy models` prints the same list with each model's reasoning levels and context window.

Typical current set:

- `gpt-6.1-sol`
- `gpt-6-astra`
- `gpt-6.1-terra`
- `gpt-6-sol`
- `gpt-6-luna`
- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `gpt-5.6-luna`

The built-in fallback list (used only when the upstream lookup fails) is in `codex_proxy/config.py`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (streaming and non-streaming) |
| POST | `/v1/responses` | Responses API (streaming/non-streaming) |
| POST | `/v1/messages` | Anthropic Messages API shim for Claude Code |
| POST | `/v1/messages/count_tokens` | Approximate Anthropic input token counting |
| GET | `/v1/models` | List models available to the current account |
| GET | `/health` | Health check (no auth required) |

### Supported request fields

Chat Completions requests accept `messages`, `tools`, `tool_choice`, `parallel_tool_calls`, `reasoning_effort`, `reasoning`, `verbosity`, `response_format`, `max_completion_tokens`/`max_tokens`, and `service_tier` (`default`/`priority`).

Parameters the Codex backend has no equivalent for (`temperature`, `top_p`, `seed`, `stop`, `n`, `logprobs`, ...) are dropped with a warning in the log instead of failing the request.

Reasoning levels are checked against the model: a level the model does not offer is dropped rather than sent upstream, where it would return a 400 (`minimal` is mapped to `low`).

## Integration Guide

codex-proxy exposes a standard OpenAI-compatible API at `http://localhost:8787/v1`. Any tool that supports a custom OpenAI base URL can use it directly.

### LiteLLM

```python
import litellm

response = litellm.completion(
    model="openai/gpt-6.1-sol",
    messages=[{"role": "user", "content": "Hello!"}],
    api_base="http://localhost:8787",
    api_key="codex-proxy",
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

If LiteLLM routes Codex-family calls to the Responses API, this proxy also supports:

```python
response = litellm.responses(
    model="openai/gpt-6.1-sol",
    input="Hello!",
    api_base="http://localhost:8787",
    api_key="codex-proxy",
    stream=True,
)
```

`/v1/responses` includes normalization for common payload variants such as `max_output_tokens`, chat-style `tools`/`tool_choice`, `response_format`, and OpenAI-style `input` message items.

### Any OpenAI-compatible CLI Agent

Most CLI agents (e.g., aider, goose) support custom base URLs. For remote access via Tailscale:

```bash
export OPENAI_API_BASE=http://<tailscale-ip>:8787/v1
export OPENAI_API_KEY=codex-proxy
export OPENAI_MODEL=gpt-6.1-sol
```

Adjust the environment variable names to match your tool's conventions.

### Claude Code

Point Claude Code at the local proxy:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8787
export ANTHROPIC_API_KEY=codex-proxy

claude
```

The Anthropic-compatible layer maps Claude-style `/v1/messages` requests onto the same ChatGPT Codex backend used by the OpenAI-compatible endpoints. Claude model names are translated onto Codex models automatically; any model name the proxy does not recognize is passed upstream unchanged, so a name from `codex-proxy models` always works too.

Current model mapping:

- `claude-opus*` → `gpt-6.1-sol`
- `claude-sonnet*` → `gpt-6-sol`
- `claude-haiku*` → `gpt-6-luna`

`thinking` blocks are turned into Codex reasoning effort, and reasoning output comes back as Anthropic `thinking` blocks so Claude Code can display it.

### General Pattern

| Setting | Value |
|---------|-------|
| Base URL | `http://localhost:8787/v1` (or `http://<tailscale-ip>:8787/v1`) |
| API Key | Value of `CODEX_PROXY_API_KEY` (unset by default) |
| Model | Any id from `codex-proxy models` |

## How It Works

1. **OAuth PKCE** — `codex-proxy login` runs a standard OAuth 2.0 + PKCE flow against `auth.openai.com`, storing tokens in `~/.codex-proxy/credentials.json`
2. **Token auto-refresh** — expired tokens are automatically refreshed using the stored refresh token
3. **TLS fingerprint** — uses `curl_cffi` with Chrome impersonation to pass Cloudflare bot checks
4. **Model catalog** — `codex_proxy/models.py` fetches and caches the upstream model list; it drives `/v1/models`, alias mapping, and reasoning-level validation
5. **Request translation** — OpenAI Chat Completions or Anthropic Messages format is converted to ChatGPT Responses API format
6. **Response translation** — Responses API SSE events are translated back to OpenAI chunks or Anthropic events, including tool calls, reasoning, usage, and truncated (`response.incomplete`) responses
7. **API key middleware** — optional downstream authentication via `CODEX_PROXY_API_KEY`

## Development

```bash
source .venv/bin/activate
pytest
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

This project uses the unofficial ChatGPT backend API (`chatgpt.com/backend-api`). It is not endorsed by OpenAI and may break at any time. Use at your own risk.
