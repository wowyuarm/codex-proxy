"""Tests for server routing and /responses normalization."""

import time

import pytest

from codex_proxy.models import ModelCatalog
from codex_proxy.server import (
    MODEL_CATALOG_KEY,
    UPSTREAM_SESSION_KEY,
    _aggregate_nonstream_chat_completion,
    _estimate_anthropic_input_tokens,
    _normalize_responses_body,
    _parse_sse_data_line,
    create_app,
    handle_chat_completions,
    handle_count_tokens,
    handle_messages,
    handle_models,
    handle_responses,
)


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
    assert "/v1/messages" in paths
    assert "/messages" in paths
    assert "/v1/messages/count_tokens" in paths
    assert "/messages/count_tokens" in paths


def test_normalize_responses_body_drops_unsupported_params_and_sets_defaults():
    body, client_stream = _normalize_responses_body(
        {
            "model": "gpt-6.1-sol",
            "input": "hi",
            "temperature": 0.2,
            "max_output_tokens": 256,
            "max_completion_tokens": 123,
            "service_tier": "default",
            "prompt_cache_retention": {"type": "ephemeral"},
            "safety_identifier": "factory-cli",
            "stream": False,
        }
    )
    assert client_stream is False
    assert "temperature" not in body
    assert "max_completion_tokens" not in body
    assert "prompt_cache_retention" not in body
    assert "safety_identifier" not in body
    # Token budget and service tier are supported upstream and must survive.
    assert body["max_output_tokens"] == 256
    assert body["service_tier"] == "default"
    assert body["instructions"] == "You are a helpful assistant."
    assert body["store"] is False
    assert body["stream"] is True
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]


def test_normalize_responses_body_migrates_max_tokens_and_drops_bad_service_tier():
    body, _ = _normalize_responses_body(
        {
            "model": "gpt-6.1-sol",
            "input": "hi",
            "max_tokens": 512,
            "service_tier": "flex",
        }
    )
    assert body["max_output_tokens"] == 512
    assert "max_tokens" not in body
    assert "service_tier" not in body


def test_normalize_responses_body_moves_response_format_into_text():
    body, _ = _normalize_responses_body(
        {
            "model": "gpt-6.1-sol",
            "input": "hi",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True},
            },
        }
    )
    assert "response_format" not in body
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "answer",
        "schema": {"type": "object"},
        "strict": True,
    }


def test_normalize_responses_body_keeps_existing_values():
    body, client_stream = _normalize_responses_body(
        {
            "model": "gpt-6.1-sol",
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


def test_normalize_responses_body_converts_chat_style_tools_and_tool_choice():
    body, _ = _normalize_responses_body(
        {
            "model": "gpt-6.1-sol",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_docs",
                        "description": "Search docs",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "search_docs"},
            },
        }
    )

    assert body["tools"] == [
        {
            "type": "function",
            "name": "search_docs",
            "description": "Search docs",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]
    assert body["tool_choice"] == {"type": "function", "name": "search_docs"}


def test_normalize_responses_body_keeps_responses_style_tools_unchanged():
    body, _ = _normalize_responses_body(
        {
            "model": "gpt-6.1-sol",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": "search_docs"},
        }
    )

    assert body["tools"] == [
        {
            "type": "function",
            "name": "search_docs",
            "description": "Search docs",
            "parameters": {"type": "object"},
            "strict": True,
        }
    ]
    assert body["tool_choice"] == {"type": "function", "name": "search_docs"}


def test_normalize_responses_body_normalizes_litellm_input_and_strict_null():
    body, _ = _normalize_responses_body(
        {
            "model": "gpt-6.1-sol",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object"},
                    "strict": None,
                }
            ],
        }
    )

    assert body["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "hello"}],
        },
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "name": "search_docs",
            "description": "Search docs",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]
    assert body["parallel_tool_calls"] is False


def test_normalize_responses_body_converts_openai_image_parts():
    body, _ = _normalize_responses_body(
        {
            "model": "gpt-6.1-sol",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/cat.png",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        }
    )

    assert body["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe"},
                {
                    "type": "input_image",
                    "image_url": "https://example.com/cat.png",
                    "detail": "low",
                },
            ],
        }
    ]


def test_normalize_responses_body_keeps_explicit_parallel_tool_calls():
    body, _ = _normalize_responses_body(
        {
            "model": "gpt-6.1-sol",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "search_docs",
                    "description": "Search docs",
                    "parameters": {"type": "object"},
                }
            ],
            "parallel_tool_calls": True,
        }
    )

    assert body["parallel_tool_calls"] is True


def test_parse_sse_data_line_extracts_payload():
    assert _parse_sse_data_line("data: {\"ok\":true}\n\n") == "{\"ok\":true}"
    assert _parse_sse_data_line("event: ping") is None


def test_aggregate_nonstream_chat_completion_text():
    completion = _aggregate_nonstream_chat_completion(
        [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-6.1-sol",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-6.1-sol",
                "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-6.1-sol",
                "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        ]
    )
    assert completion is not None
    assert completion["object"] == "chat.completion"
    assert completion["model"] == "gpt-6.1-sol"
    assert completion["choices"][0]["message"]["role"] == "assistant"
    assert completion["choices"][0]["message"]["content"] == "Hello"
    assert completion["choices"][0]["finish_reason"] == "stop"
    assert completion["usage"]["total_tokens"] == 3


def test_aggregate_nonstream_chat_completion_tool_calls():
    completion = _aggregate_nonstream_chat_completion(
        [
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 456,
                "model": "gpt-6.1-sol",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 456,
                "model": "gpt-6.1-sol",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "fc_123",
                                    "type": "function",
                                    "function": {"name": "search_docs", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 456,
                "model": "gpt-6.1-sol",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"q":"hel'}}]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 456,
                "model": "gpt-6.1-sol",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'lo"}'}}]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
    )
    assert completion is not None
    message = completion["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] is None
    assert message["tool_calls"][0]["id"] == "fc_123"
    assert message["tool_calls"][0]["function"]["name"] == "search_docs"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"q":"hello"}'
    assert completion["choices"][0]["finish_reason"] == "tool_calls"


def test_estimate_anthropic_input_tokens_counts_system_messages_and_tool_results():
    payload = {
        "system": [{"type": "text", "text": "Be careful."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "World"}],
            },
        ],
    }

    result = _estimate_anthropic_input_tokens(payload)

    assert result["input_tokens"] > 0


def _fake_app(session=None) -> dict:
    """Minimal app mapping the handlers read from."""
    return {UPSTREAM_SESSION_KEY: session, MODEL_CATALOG_KEY: ModelCatalog()}


@pytest.fixture(autouse=True)
def _offline_model_catalog(monkeypatch):
    """Keep handler tests off the network: the catalog resolves to nothing."""

    async def no_catalog(credentials):
        return []

    monkeypatch.setattr("codex_proxy.models.fetch_model_catalog", no_catalog)


class _FakeRequest:
    def __init__(self, body, app):
        self._body = body
        self.app = app

    async def json(self):
        return self._body


class _FakeRequestError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


class _FakeStreamResponse:
    def __init__(self):
        self.content_type = None
        self.headers = {}
        self.prepared = False
        self.writes: list[bytes] = []

    async def prepare(self, request):
        self.prepared = True
        return self

    async def write(self, data):
        self.writes.append(data if isinstance(data, bytes) else data.encode())


class _FakeUpstreamLines:
    def __init__(self, items, status_code=200):
        self._items = list(items)
        self.status_code = status_code

    async def aiter_lines(self):
        for item in self._items:
            if isinstance(item, Exception):
                raise item
            yield item


class _FakeUpstreamContent:
    def __init__(self, items, status_code=200):
        self._items = list(items)
        self.status_code = status_code

    async def aiter_content(self):
        for item in self._items:
            if isinstance(item, Exception):
                raise item
            yield item


class _FakeSession:
    def __init__(self, upstreams):
        self._upstreams = list(upstreams)
        self.calls = 0

    async def post(self, *args, **kwargs):
        upstream = self._upstreams[self.calls]
        self.calls += 1
        return upstream


class _FakeJsonResponse:
    def __init__(self, data, status=200):
        self.data = data
        self.status = status


async def _fake_credentials():
    return {"access_token": "test-token", "account_id": "acct_test"}


@pytest.mark.asyncio
async def test_handle_chat_completions_retries_before_first_downstream_byte(monkeypatch):
    first_error = _FakeRequestError(
        "Failed to perform, curl: (35) Recv failure: Connection reset by peer",
        35,
    )
    session = _FakeSession(
        [
            _FakeUpstreamLines([first_error]),
            _FakeUpstreamLines(
                [
                    'data: {"type":"response.output_item.added","item":{"type":"message","role":"assistant"}}',
                    'data: {"type":"response.output_text.delta","delta":"Hi"}',
                    (
                        'data: {"type":"response.completed","response":{"status":"completed",'
                        '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,'
                        '"input_tokens_details":{"cached_tokens":0}}}}'
                    ),
                ]
            ),
        ]
    )
    request = _FakeRequest(
        {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
        _fake_app(session),
    )

    monkeypatch.setattr("codex_proxy.server.ensure_credentials", _fake_credentials)
    monkeypatch.setattr("codex_proxy.server.web.StreamResponse", _FakeStreamResponse)

    response = await handle_chat_completions(request)

    assert isinstance(response, _FakeStreamResponse)
    assert session.calls == 2
    body = b"".join(response.writes).decode()
    assert '"content": "Hi"' in body
    assert body.count("data: [DONE]") == 1


@pytest.mark.asyncio
async def test_handle_chat_completions_retries_transient_upstream_error_event(monkeypatch):
    session = _FakeSession(
        [
            _FakeUpstreamLines(
                [
                    (
                        'data: {"type":"error","error":{"type":"service_unavailable",'
                        '"message":"An error occurred while processing your request. Please retry."}}'
                    ),
                ]
            ),
            _FakeUpstreamLines(
                [
                    'data: {"type":"response.output_item.added","item":{"type":"message","role":"assistant"}}',
                    'data: {"type":"response.output_text.delta","delta":"Recovered"}',
                    (
                        'data: {"type":"response.completed","response":{"status":"completed",'
                        '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,'
                        '"input_tokens_details":{"cached_tokens":0}}}}'
                    ),
                ]
            ),
        ]
    )
    request = _FakeRequest(
        {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
        _fake_app(session),
    )

    monkeypatch.setattr("codex_proxy.server.ensure_credentials", _fake_credentials)
    monkeypatch.setattr("codex_proxy.server.web.StreamResponse", _FakeStreamResponse)

    response = await handle_chat_completions(request)

    assert isinstance(response, _FakeStreamResponse)
    assert session.calls == 2
    body = b"".join(response.writes).decode()
    assert '"content": "Recovered"' in body
    assert '"type": "error"' not in body
    assert body.count("data: [DONE]") == 1


@pytest.mark.asyncio
async def test_handle_responses_stream_writes_sse_error_after_upstream_reset(monkeypatch):
    stream_error = _FakeRequestError(
        "Failed to perform, curl: (35) Recv failure: Connection reset by peer",
        35,
    )
    session = _FakeSession(
        [
            _FakeUpstreamContent(
                [
                    b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n',
                    stream_error,
                ]
            )
        ]
    )
    request = _FakeRequest(
        {
            "model": "gpt-6.1-sol",
            "input": "hi",
            "stream": True,
        },
        _fake_app(session),
    )

    monkeypatch.setattr("codex_proxy.server.ensure_credentials", _fake_credentials)
    monkeypatch.setattr("codex_proxy.server.web.StreamResponse", _FakeStreamResponse)

    response = await handle_responses(request)

    assert isinstance(response, _FakeStreamResponse)
    body = b"".join(response.writes).decode()
    assert '"type":"response.output_text.delta"' in body or '"type": "response.output_text.delta"' in body
    assert '"type": "error"' in body
    assert "connection_error" in body
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_handle_messages_streams_anthropic_events(monkeypatch):
    session = _FakeSession(
        [
            _FakeUpstreamLines(
                [
                    'data: {"type":"response.output_item.added","item":{"type":"message","id":"msg_1","role":"assistant"}}',
                    'data: {"type":"response.output_text.delta","delta":"Hello"}',
                    'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"fc_1","name":"lookup"}}',
                    'data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"q\\":\\"Ping\\"}"}',
                    'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":2,"output_tokens":4}}}',
                ]
            )
        ]
    )
    request = _FakeRequest(
        {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Ping"}]}],
            "stream": True,
        },
        _fake_app(session),
    )

    monkeypatch.setattr("codex_proxy.server.ensure_credentials", _fake_credentials)
    monkeypatch.setattr("codex_proxy.server.web.StreamResponse", _FakeStreamResponse)

    response = await handle_messages(request)

    assert isinstance(response, _FakeStreamResponse)
    body = b"".join(response.writes).decode()
    assert "event: message_start" in body
    assert '"type": "text_delta"' in body
    assert '"type": "tool_use"' in body
    assert "event: message_stop" in body


@pytest.mark.asyncio
async def test_handle_messages_nonstream_upstream_http_error_uses_anthropic_shape(monkeypatch):
    session = _FakeSession([_FakeUpstreamContent([b"model not found"], status_code=404)])
    request = _FakeRequest(
        {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Ping"}]}],
            "stream": False,
        },
        _fake_app(session),
    )

    monkeypatch.setattr("codex_proxy.server.ensure_credentials", _fake_credentials)
    monkeypatch.setattr("codex_proxy.server.web.json_response", _FakeJsonResponse)

    response = await handle_messages(request)

    assert response.status == 404
    assert response.data == {
        "type": "error",
        "error": {
            "type": "upstream_error",
            "message": "model not found",
        },
    }


@pytest.mark.asyncio
async def test_handle_chat_completions_closes_stream_without_terminal_event(monkeypatch):
    session = _FakeSession(
        [
            _FakeUpstreamLines(
                [
                    'data: {"type":"response.output_text.delta","delta":"partial"}',
                ]
            )
        ]
    )
    request = _FakeRequest(
        {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
        _fake_app(session),
    )

    monkeypatch.setattr("codex_proxy.server.ensure_credentials", _fake_credentials)
    monkeypatch.setattr("codex_proxy.server.web.StreamResponse", _FakeStreamResponse)

    response = await handle_chat_completions(request)

    body = b"".join(response.writes).decode()
    assert '"content": "partial"' in body
    assert '"finish_reason": "stop"' in body
    assert body.count("data: [DONE]") == 1


@pytest.mark.asyncio
async def test_handle_messages_nonstream_upstream_error_event_uses_anthropic_shape(monkeypatch):
    session = _FakeSession(
        [
            _FakeUpstreamLines(
                [
                    'data: {"type":"error","error":{"type":"invalid_request_error","message":"bad model"}}',
                ]
            )
        ]
    )
    request = _FakeRequest(
        {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Ping"}]}],
            "stream": False,
        },
        _fake_app(session),
    )

    monkeypatch.setattr("codex_proxy.server.ensure_credentials", _fake_credentials)
    monkeypatch.setattr("codex_proxy.server.web.json_response", _FakeJsonResponse)

    response = await handle_messages(request)

    assert response.status == 502
    assert response.data == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "bad model"},
    }


@pytest.mark.asyncio
async def test_handle_models_lists_only_visible_models(monkeypatch):
    from codex_proxy.models import parse_model_catalog

    app = _fake_app()
    catalog = app[MODEL_CATALOG_KEY]
    catalog._models = parse_model_catalog(
        {
            "models": [
                {"slug": "gpt-6.1-sol", "visibility": "list", "priority": 1},
                {"slug": "gpt-reserve", "visibility": "hide", "priority": 2},
            ]
        }
    )
    catalog._fetched_at = time.time()

    monkeypatch.setattr("codex_proxy.server.ensure_credentials", _fake_credentials)
    monkeypatch.setattr("codex_proxy.server.web.json_response", _FakeJsonResponse)

    response = await handle_models(_FakeRequest({}, app))

    assert [entry["id"] for entry in response.data["data"]] == ["gpt-6.1-sol"]


@pytest.mark.asyncio
async def test_handle_chat_completions_requires_login(monkeypatch):
    async def missing_credentials():
        raise RuntimeError("Not logged in. Run 'codex-proxy login' first.")

    monkeypatch.setattr("codex_proxy.server.ensure_credentials", missing_credentials)
    monkeypatch.setattr("codex_proxy.server.web.json_response", _FakeJsonResponse)

    response = await handle_chat_completions(
        _FakeRequest(
            {"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "Hi"}]},
            _fake_app(),
        )
    )

    assert response.status == 401
    assert "Not logged in" in response.data["error"]


@pytest.mark.asyncio
async def test_handle_count_tokens_returns_estimate(monkeypatch):
    request = _FakeRequest(
        {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
            "system": "Be careful.",
        },
        _fake_app(),
    )

    monkeypatch.setattr("codex_proxy.server.web.json_response", _FakeJsonResponse)

    response = await handle_count_tokens(request)

    assert response.status == 200
    assert response.data["input_tokens"] > 0
