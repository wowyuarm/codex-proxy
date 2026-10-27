"""Tests for the upstream model catalog."""

import pytest

from codex_proxy.models import (
    ModelCatalog,
    allowed_reasoning_efforts,
    default_model,
    known_model_ids,
    normalize_reasoning_effort,
    parse_model_catalog,
    resolve_model_name,
    visible_model_ids,
)

_CATALOG_PAYLOAD = {
    "models": [
        {
            "slug": "gpt-6.1-sol",
            "display_name": "GPT-6.1 Sol",
            "description": "Flagship",
            "visibility": "list",
            "priority": 1,
            "context_window": 400000,
            "supported_reasoning_levels": [
                {"effort": "low"},
                {"effort": "medium"},
                {"effort": "high"},
            ],
            "default_reasoning_level": "medium",
            "support_verbosity": True,
        },
        {
            "slug": "gpt-6-luna",
            "display_name": "GPT-6 Luna",
            "visibility": "list",
            "priority": 2,
            "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
        },
        {
            "slug": "gpt-reserve",
            "display_name": "Reserve",
            "visibility": "hide",
            "priority": 9,
        },
        {
            "slug": "gpt-retired",
            "display_name": "Retired",
            "visibility": "list",
            "supported_in_api": False,
        },
    ]
}


def test_parse_model_catalog_normalizes_and_orders_records():
    models = parse_model_catalog(_CATALOG_PAYLOAD)

    assert [model["slug"] for model in models] == ["gpt-6.1-sol", "gpt-6-luna", "gpt-reserve"]
    assert models[0]["display_name"] == "GPT-6.1 Sol"
    assert models[0]["context_window"] == 400000
    assert models[0]["reasoning_efforts"] == ["low", "medium", "high"]
    assert models[0]["default_reasoning_effort"] == "medium"
    assert models[0]["supports_verbosity"] is True


def test_parse_model_catalog_rejects_malformed_payloads():
    assert parse_model_catalog({}) == []
    assert parse_model_catalog({"models": "nope"}) == []
    assert parse_model_catalog({"models": [{"slug": ""}, "junk"]}) == []


def test_visible_models_exclude_hidden_ones():
    models = parse_model_catalog(_CATALOG_PAYLOAD)

    assert visible_model_ids(models) == ["gpt-6.1-sol", "gpt-6-luna"]
    # Hidden models stay callable by name.
    assert "gpt-reserve" in known_model_ids(models)


def test_known_model_ids_falls_back_when_catalog_is_empty():
    assert known_model_ids([]) == [
        "gpt-6.1-sol",
        "gpt-6-astra",
        "gpt-6.1-terra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ]


def test_default_model_prefers_first_visible_entry():
    models = parse_model_catalog(_CATALOG_PAYLOAD)
    assert default_model(models) == "gpt-6.1-sol"
    assert default_model([]) == "gpt-6.1-sol"


def test_resolve_model_name_handles_aliases_prefixes_and_unknowns():
    models = parse_model_catalog(_CATALOG_PAYLOAD)

    assert resolve_model_name("claude-opus-4-1", models) == "gpt-6.1-sol"
    assert resolve_model_name("claude-sonnet-4-5-20250929", models) == "gpt-6-sol"
    assert resolve_model_name("openai/gpt-6-luna", models) == "gpt-6-luna"
    assert resolve_model_name("gpt-reserve", models) == "gpt-reserve"
    # Unknown names pass through so the upstream can report its own error.
    assert resolve_model_name("some-new-model", models) == "some-new-model"
    assert resolve_model_name(None, models) == "gpt-6.1-sol"


def test_reasoning_effort_clamped_to_model_levels():
    models = parse_model_catalog(_CATALOG_PAYLOAD)

    assert allowed_reasoning_efforts("gpt-6.1-sol", models) == ("low", "medium", "high")
    assert normalize_reasoning_effort("high", "gpt-6.1-sol", models) == "high"
    # `minimal` is not accepted upstream; the closest real level is `low`.
    assert normalize_reasoning_effort("minimal", "gpt-6.1-sol", models) == "low"
    # `ultra` is advertised by some catalogs but rejected with a 400.
    assert normalize_reasoning_effort("ultra", "gpt-6.1-sol", models) is None
    # Unknown model: pass the request through untouched.
    assert normalize_reasoning_effort("ultra", "unknown", models) == "ultra"
    assert normalize_reasoning_effort(None, "gpt-6.1-sol", models) is None


@pytest.mark.asyncio
async def test_catalog_caches_until_ttl_expires(monkeypatch):
    calls = []

    async def fake_fetch(credentials):
        calls.append(credentials)
        return parse_model_catalog(_CATALOG_PAYLOAD)

    monkeypatch.setattr("codex_proxy.models.fetch_model_catalog", fake_fetch)
    catalog = ModelCatalog(ttl=3600)

    first = await catalog.get({"access_token": "t"})
    second = await catalog.get({"access_token": "t"})

    assert first == second
    assert len(calls) == 1
    assert visible_model_ids(second) == ["gpt-6.1-sol", "gpt-6-luna"]


@pytest.mark.asyncio
async def test_catalog_keeps_last_good_data_when_refresh_fails(monkeypatch):
    async def good_fetch(credentials):
        return parse_model_catalog(_CATALOG_PAYLOAD)

    monkeypatch.setattr("codex_proxy.models.fetch_model_catalog", good_fetch)
    catalog = ModelCatalog(ttl=0)
    await catalog.get({"access_token": "t"})

    async def failing_fetch(credentials):
        return []

    monkeypatch.setattr("codex_proxy.models.fetch_model_catalog", failing_fetch)
    result = await catalog.get({"access_token": "t"})

    assert visible_model_ids(result) == ["gpt-6.1-sol", "gpt-6-luna"]


@pytest.mark.asyncio
async def test_catalog_does_not_fetch_without_credentials(monkeypatch):
    async def fake_fetch(credentials):
        raise AssertionError("should not fetch without credentials")

    monkeypatch.setattr("codex_proxy.models.fetch_model_catalog", fake_fetch)
    catalog = ModelCatalog()

    assert await catalog.get(None) == []
