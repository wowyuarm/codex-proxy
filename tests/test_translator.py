"""Tests for the translator module."""

import json

from codex_proxy.translator import (
    ResponseStreamTranslator,
    chat_to_responses,
)


class TestChatToResponses:
    def test_basic_user_message(self):
        request = {
            "model": "gpt-5.1",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        result = chat_to_responses(request)
        assert result["model"] == "gpt-5.1"
        assert result["stream"] is True
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"
        assert result["input"][0]["content"] == [{"type": "input_text", "text": "Hello"}]

    def test_system_message_becomes_instructions(self):
        request = {
            "model": "gpt-5.1",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = chat_to_responses(request)
        assert result["instructions"] == "You are helpful."
        # System message should not appear in input
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"

    def test_assistant_message(self):
        request = {
            "model": "gpt-5.1",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        }
        result = chat_to_responses(request)
        assert len(result["input"]) == 2
        assistant_msg = result["input"][1]
        assert assistant_msg["type"] == "message"
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"][0]["text"] == "Hello!"

    def test_tool_call_and_result(self):
        request = {
            "model": "gpt-5.1",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Tokyo"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "Sunny, 25C",
                },
            ],
        }
        result = chat_to_responses(request)
        assert len(result["input"]) == 3

        # Function call
        fc = result["input"][1]
        assert fc["type"] == "function_call"
        assert fc["name"] == "get_weather"
        assert fc["id"].startswith("fc_")

        # Function output
        fo = result["input"][2]
        assert fo["type"] == "function_call_output"
        assert fo["output"] == "Sunny, 25C"
        assert fo["call_id"].startswith("fc_")

    def test_tools_conversion(self):
        request = {
            "model": "gpt-5.1",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        result = chat_to_responses(request)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["strict"] is False

    def test_temperature_and_max_tokens(self):
        request = {
            "model": "gpt-5.1",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        result = chat_to_responses(request)
        assert result["temperature"] == 0.3
        assert result["max_tokens"] == 1024

    def test_multipart_content(self):
        request = {
            "model": "gpt-5.1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ],
        }
        result = chat_to_responses(request)
        content = result["input"][0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "What's in this image?"}
        assert content[1] == {"type": "input_image", "url": "https://example.com/img.png"}


class TestResponseStreamTranslator:
    def test_text_streaming(self):
        t = ResponseStreamTranslator("gpt-5.1")

        # output_item.added with message type
        lines = t.translate_event(
            "response.output_item.added",
            {
                "item": {"type": "message", "role": "assistant"},
            },
        )
        assert len(lines) == 1
        chunk = json.loads(lines[0].removeprefix("data: ").strip())
        assert chunk["choices"][0]["delta"]["role"] == "assistant"

        # text delta
        lines = t.translate_event("response.output_text.delta", {"delta": "Hello"})
        assert len(lines) == 1
        chunk = json.loads(lines[0].removeprefix("data: ").strip())
        assert chunk["choices"][0]["delta"]["content"] == "Hello"

    def test_tool_call_streaming(self):
        t = ResponseStreamTranslator("gpt-5.1")

        # function_call item added
        lines = t.translate_event(
            "response.output_item.added",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "fc_001",
                    "name": "get_weather",
                },
            },
        )
        assert len(lines) == 1
        chunk = json.loads(lines[0].removeprefix("data: ").strip())
        tc = chunk["choices"][0]["delta"]["tool_calls"][0]
        assert tc["index"] == 0
        assert tc["function"]["name"] == "get_weather"

        # arguments delta
        lines = t.translate_event(
            "response.function_call_arguments.delta",
            {
                "delta": '{"city":',
                "item_id": "fc_001",
            },
        )
        assert len(lines) == 1
        chunk = json.loads(lines[0].removeprefix("data: ").strip())
        tc = chunk["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["arguments"] == '{"city":'

    def test_completed_with_usage(self):
        t = ResponseStreamTranslator("gpt-5.1")

        lines = t.translate_event(
            "response.completed",
            {
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                        "input_tokens_details": {"cached_tokens": 20},
                    },
                },
            },
        )
        # Should produce final chunk + [DONE]
        assert len(lines) == 2
        chunk = json.loads(lines[0].removeprefix("data: ").strip())
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert chunk["usage"]["prompt_tokens"] == 100
        assert chunk["usage"]["completion_tokens"] == 50
        assert lines[1].strip() == "data: [DONE]"

    def test_tool_calls_finish_reason(self):
        t = ResponseStreamTranslator("gpt-5.1")

        # Add a tool call first so finish_reason becomes "tool_calls"
        t.translate_event(
            "response.output_item.added",
            {
                "item": {"type": "function_call", "id": "fc_x", "call_id": "fc_x", "name": "fn"},
            },
        )

        lines = t.translate_event(
            "response.completed",
            {
                "response": {"status": "completed", "usage": {}},
            },
        )
        chunk = json.loads(lines[0].removeprefix("data: ").strip())
        assert chunk["choices"][0]["finish_reason"] == "tool_calls"
