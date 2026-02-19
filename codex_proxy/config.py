"""Constants for the OpenAI Codex OAuth flow and API endpoints."""

from pathlib import Path

# OAuth constants (from pi-mono / OpenAI Codex CLI)
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
CALLBACK_PORT = 1455

# JWT claim path for account ID extraction
JWT_CLAIM_PATH = "https://api.openai.com/auth"

# ChatGPT backend API
CHATGPT_BACKEND_URL = "https://chatgpt.com/backend-api"
RESPONSES_ENDPOINT = f"{CHATGPT_BACKEND_URL}/responses"

# Local storage
CONFIG_DIR = Path.home() / ".codex-proxy"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"

# Server defaults
DEFAULT_PORT = 8787
DEFAULT_HOST = "127.0.0.1"

# Available Codex models
CODEX_MODELS = [
    {"id": "gpt-5.1", "object": "model", "owned_by": "openai"},
    {"id": "gpt-5.1-codex-max", "object": "model", "owned_by": "openai"},
    {"id": "gpt-5.1-codex-mini", "object": "model", "owned_by": "openai"},
    {"id": "gpt-5.3-codex-spark", "object": "model", "owned_by": "openai"},
]
