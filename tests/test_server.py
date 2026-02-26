"""Tests for server routing."""

from codex_proxy.server import create_app


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
