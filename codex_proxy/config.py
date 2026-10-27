"""Constants for the OpenAI Codex OAuth flow and API endpoints."""

import os
from pathlib import Path

# OAuth constants (matching the official Codex CLI)
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CALLBACK_PORT = 1455
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/auth/callback"
SCOPE = "openid profile email offline_access"

# JWT claim path for account ID extraction
JWT_CLAIM_PATH = "https://api.openai.com/auth"

# ChatGPT backend API
CHATGPT_BACKEND_URL = "https://chatgpt.com/backend-api"
RESPONSES_ENDPOINT = f"{CHATGPT_BACKEND_URL}/codex/responses"
MODELS_ENDPOINT = f"{CHATGPT_BACKEND_URL}/codex/models"
USAGE_ENDPOINT = f"{CHATGPT_BACKEND_URL}/wham/usage"

# The upstream model catalog is gated by this query parameter: older client
# versions only see retired models. It tracks the official Codex CLI release.
CODEX_CLIENT_VERSION = os.environ.get("CODEX_PROXY_CLIENT_VERSION", "0.164.0")

# Local storage
CONFIG_DIR = Path(os.environ.get("CODEX_PROXY_HOME") or Path.home() / ".codex-proxy")
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
ACCOUNTS_DIR = CONFIG_DIR / "accounts"
REGISTRY_FILE = ACCOUNTS_DIR / "registry.json"

# Server defaults
DEFAULT_PORT = 8787
DEFAULT_HOST = "0.0.0.0"

# Fallback catalog, used only when the upstream model list cannot be fetched.
# Ordered newest-first; `codex-proxy models` shows what the live upstream offers.
FALLBACK_MODELS = [
    "gpt-6.1-sol",
    "gpt-6-astra",
    "gpt-6.1-terra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
]
DEFAULT_MODEL = FALLBACK_MODELS[0]

# Reasoning effort levels accepted by the Codex backend, weakest first.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

# Claude Code model aliases mapped onto Codex models.
CLAUDE_MODEL_ALIASES = {
    "claude-opus": "gpt-6.1-sol",
    "claude-sonnet": "gpt-6-sol",
    "claude-haiku": "gpt-6-luna",
}
