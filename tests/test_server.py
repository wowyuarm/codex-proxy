"""Tests for server routing and /responses normalization."""

from codex_proxy.server import (
    _aggregate_nonstream_chat_completion,
    _normalize_responses_body,
    _parse_sse_data_line,
    create_app,
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


def test_normalize_responses_body_drops_max_output_tokens_and_sets_defaults():
    body, client_stream = _normalize_responses_body(
        {
            "model": "gpt-5.3-codex",
            "input": "hi",
            "max_output_tokens": 256,
            "max_completion_tokens": 123,
            "stream": False,
        }
    )
    assert client_stream is False
    assert "max_output_tokens" not in body
    assert "max_completion_tokens" not in body
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


def test_normalize_responses_body_converts_chat_style_tools_and_tool_choice():
    body, _ = _normalize_responses_body(
        {
            "model": "gpt-5.3-codex",
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
            "model": "gpt-5.3-codex",
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
            "model": "gpt-5.3-codex",
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


def test_normalize_responses_body_keeps_explicit_parallel_tool_calls():
    body, _ = _normalize_responses_body(
        {
            "model": "gpt-5.3-codex",
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
                "model": "gpt-5.3-codex",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-5.3-codex",
                "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-5.3-codex",
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
    assert completion["model"] == "gpt-5.3-codex"
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
                "model": "gpt-5.3-codex",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-tool",
                "object": "chat.completion.chunk",
                "created": 456,
                "model": "gpt-5.3-codex",
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
                "model": "gpt-5.3-codex",
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
                "model": "gpt-5.3-codex",
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
