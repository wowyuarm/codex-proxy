"""Tests for server routing and /responses normalization."""

from codex_proxy.server import _normalize_responses_body, create_app


def _post_paths() -> set[str]:
    app = create_app()
    paths: set[str] = set()
    for route in app.router.routes():
        if route.method != "POST":
            continue
        path = route.resource.get_info().get("path")
        if path:
            paths.add(path)
    return paths


def test_post_routes_include_chat_completions_and_responses():
    paths = _post_paths()
    assert "/v1/chat/completions" in paths
    assert "/chat/completions" in paths
    assert "/v1/responses" in paths
    assert "/responses" in paths


def test_normalize_responses_body_drops_max_output_tokens_and_sets_defaults():
    body, client_stream = _normalize_responses_body(
        {
            "model": "gpt-5.3-codex",
            "input": "hi",
            "max_output_tokens": 256,
            "stream": False,
        }
    )
    assert client_stream is False
    assert "max_output_tokens" not in body
    assert body["instructions"] == "You are a helpful assistant."
    assert body["store"] is False
    assert body["stream"] is True
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]


def test_normalize_responses_body_keeps_existing_values():
    body, client_stream = _normalize_responses_body(
        {
            "model": "gpt-5.3-codex",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "instructions": "Custom instruction",
            "store": False,
            "stream": True,
        }
    )
    assert client_stream is True
    assert body["instructions"] == "Custom instruction"
    assert body["store"] is False
    assert body["stream"] is True
