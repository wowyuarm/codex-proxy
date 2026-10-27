"""Fetch and cache the upstream Codex model catalog."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import aiohttp

from codex_proxy.config import (
    CLAUDE_MODEL_ALIASES,
    CODEX_CLIENT_VERSION,
    DEFAULT_MODEL,
    FALLBACK_MODELS,
    MODELS_ENDPOINT,
)

log = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 300
_FETCH_ATTEMPTS = 3


def _get_proxy() -> str | None:
    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")


def parse_model_catalog(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the upstream catalog into local model records.

    Hidden models (for example `gpt-reserve`) stay in the list because clients
    can still call them by name, but `visible` marks what `/v1/models` returns.
    """
    models = payload.get("models")
    if not isinstance(models, list):
        return []

    parsed: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        if item.get("supported_in_api") is False:
            continue

        levels = [
            level["effort"]
            for level in item.get("supported_reasoning_levels") or []
            if isinstance(level, dict) and isinstance(level.get("effort"), str)
        ]
        parsed.append(
            {
                "slug": slug,
                "display_name": item.get("display_name") or slug,
                "description": item.get("description") or "",
                "visible": item.get("visibility") == "list",
                "priority": item.get("priority") if isinstance(item.get("priority"), int) else 999,
                "context_window": item.get("context_window"),
                "reasoning_efforts": levels,
                "default_reasoning_effort": item.get("default_reasoning_level"),
                "supports_verbosity": bool(item.get("support_verbosity")),
            }
        )

    parsed.sort(key=lambda model: model["priority"])
    return parsed


async def fetch_model_catalog(credentials: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch the live model catalog for one account."""
    access_token = credentials.get("access_token")
    account_id = credentials.get("account_id")
    if not access_token:
        return []

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": "codex-proxy",
    }
    if account_id:
        headers["chatgpt-account-id"] = account_id

    url = f"{MODELS_ENDPOINT}?client_version={CODEX_CLIENT_VERSION}"
    timeout = aiohttp.ClientTimeout(total=15)
    last_error: Exception | None = None
    # Connections through a local HTTP proxy fail intermittently; a couple of
    # retries turn a mostly-successful lookup into a reliable one.
    for attempt in range(1, _FETCH_ATTEMPTS + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, proxy=_get_proxy()) as resp:
                    if resp.status != 200:
                        log.warning("Model catalog lookup failed (%d)", resp.status)
                        return []
                    return parse_model_catalog(await resp.json(content_type=None))
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < _FETCH_ATTEMPTS:
                await asyncio.sleep(0.5 * attempt)
    log.warning("Model catalog lookup failed: %s", last_error)
    return []


class ModelCatalog:
    """Caches the upstream catalog so `/v1/models` does not hit the network per call."""

    def __init__(self, ttl: float = _CACHE_TTL_SECONDS) -> None:
        self.ttl = ttl
        self._models: list[dict[str, Any]] = []
        self._fetched_at = 0.0

    @property
    def stale(self) -> bool:
        return not self._models or time.time() - self._fetched_at > self.ttl

    async def get(self, credentials: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Return the cached catalog, refreshing it when stale."""
        if not self.stale:
            return self._models
        if credentials is not None:
            models = await fetch_model_catalog(credentials)
            if models:
                self._models = models
                self._fetched_at = time.time()
                return self._models
        return self._models

    async def refresh(self, credentials: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Force a refresh, used by the CLI."""
        self._models = []
        self._fetched_at = 0.0
        return await self.get(credentials)


def visible_model_ids(models: list[dict[str, Any]]) -> list[str]:
    """Return callable, listed model ids in priority order."""
    return [model["slug"] for model in models if model["visible"]]


def known_model_ids(models: list[dict[str, Any]]) -> list[str]:
    if not models:
        return list(FALLBACK_MODELS)
    return [model["slug"] for model in models]


def resolve_model_name(name: str | None, models: list[dict[str, Any]]) -> str:
    """Map client model names (Claude aliases, provider prefixes) onto a Codex model."""
    if not isinstance(name, str) or not name:
        return default_model(models)

    candidate = name.split("/", 1)[1] if "/" in name else name
    lowered = candidate.lower()

    for prefix, target in CLAUDE_MODEL_ALIASES.items():
        if lowered.startswith(prefix):
            return target

    known = set(known_model_ids(models))
    if candidate in known:
        return candidate

    # Dated Claude variants such as claude-sonnet-4-5-20250929 land here after
    # the alias step, so also match on the leading family token.
    for prefix, target in CLAUDE_MODEL_ALIASES.items():
        if prefix in lowered:
            return target

    return candidate


def default_model(models: list[dict[str, Any]]) -> str:
    """Pick the highest-priority listed model, or the configured fallback."""
    listed = visible_model_ids(models)
    return listed[0] if listed else DEFAULT_MODEL


def allowed_reasoning_efforts(model: str, models: list[dict[str, Any]]) -> tuple[str, ...] | None:
    """Return upstream-accepted effort levels for one model, if known."""
    for entry in models:
        if entry["slug"] == model:
            levels = entry.get("reasoning_efforts") or []
            return tuple(levels) if levels else None
    return None


def normalize_reasoning_effort(
    effort: Any, model: str, models: list[dict[str, Any]]
) -> str | None:
    """Clamp a client-requested effort to one the model actually accepts.

    Claude Code sends `thinking` budgets and OpenAI clients send `reasoning_effort`;
    the Codex backend rejects unknown levels with a 400, so map them locally.
    """
    if not isinstance(effort, str) or not effort:
        return None

    lowered = effort.lower()
    allowed = allowed_reasoning_efforts(model, models)
    if allowed is None or lowered in allowed:
        return effort

    if lowered == "minimal" and "low" in allowed:
        return "low"
    return None
