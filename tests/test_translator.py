"""Tests for the translator module."""

import json

from codex_proxy.translator import (
    AnthropicStreamTranslator,
    ResponseStreamTranslator,
    anthropic_messages_to_responses,
    chat_to_responses,
)


class TestChatToResponses:
    def test_basic_user_message(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        result = chat_to_responses(request)
        assert result["model"] == "gpt-6.1-sol"
        assert result["stream"] is True
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"
        assert result["input"][0]["content"] == [{"type": "input_text", "text": "Hello"}]

    def test_system_message_becomes_instructions(self):
        request = {
            "model": "gpt-6.1-sol",
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
            "model": "gpt-6.1-sol",
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
            "model": "gpt-6.1-sol",
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
            "model": "gpt-6.1-sol",
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
        assert result["tools"][0]["strict"] is False  # default when not specified

    def test_temperature_and_max_tokens_stripped(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        result = chat_to_responses(request)
        assert "temperature" not in result
        assert "max_tokens" not in result

    def test_reasoning_effort_and_verbosity_mapped(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "medium",
            "verbosity": "medium",
        }
        result = chat_to_responses(request)
        # `summary: auto` makes the backend emit reasoning events at all.
        assert result["reasoning"] == {"effort": "medium", "summary": "auto"}
        assert result["text"] == {"verbosity": "medium"}

    def test_max_tokens_and_reasoning_budget_forwarded(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 2048,
        }
        result = chat_to_responses(request)
        assert result["max_output_tokens"] == 2048
        assert "max_completion_tokens" not in result
        assert "max_tokens" not in result

    def test_response_format_becomes_text_format(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True},
            },
        }
        result = chat_to_responses(request)
        assert "response_format" not in result
        assert result["text"]["format"]["type"] == "json_schema"
        assert result["text"]["format"]["name"] == "answer"

    def test_developer_role_becomes_instructions(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [
                {"role": "developer", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = chat_to_responses(request)
        assert result["instructions"] == "Be brief."
        assert len(result["input"]) == 1

    def test_multiple_system_messages_concatenated(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [
                {"role": "system", "content": "You are a coder."},
                {"role": "system", "content": "Always use Python."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = chat_to_responses(request)
        assert "You are a coder." in result["instructions"]
        assert "Always use Python." in result["instructions"]

    def test_assistant_content_and_tool_calls_both_preserved(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": "Let me check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Tokyo"}',
                            },
                        }
                    ],
                },
            ],
        }
        result = chat_to_responses(request)
        # Should have: user msg + assistant text msg + function_call = 3 items
        assert len(result["input"]) == 3
        assert result["input"][1]["type"] == "message"
        assert result["input"][1]["content"][0]["text"] == "Let me check the weather for you."
        assert result["input"][2]["type"] == "function_call"
        assert result["input"][2]["name"] == "get_weather"

    def test_tool_strict_forwarded(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "fn",
                        "description": "d",
                        "parameters": {},
                        "strict": True,
                    },
                }
            ],
        }
        result = chat_to_responses(request)
        assert result["tools"][0]["strict"] is True

    def test_multipart_content(self):
        request = {
            "model": "gpt-6.1-sol",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/img.png",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }
        result = chat_to_responses(request)
        content = result["input"][0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "input_text", "text": "What's in this image?"}
        assert content[1] == {
            "type": "input_image",
            "image_url": "https://example.com/img.png",
            "detail": "high",
        }


class TestAnthropicMessagesToResponses:
    def test_maps_claude_model_tiers_to_codex_models(self):
        opus = anthropic_messages_to_responses({"model": "claude-opus-4-1", "messages": []})
        sonnet = anthropic_messages_to_responses({"model": "claude-sonnet-4-5", "messages": []})
        haiku = anthropic_messages_to_responses({"model": "claude-haiku-4-5", "messages": []})

        assert opus["model"] == "gpt-6.1-sol"
        assert sonnet["model"] == "gpt-6-sol"
        assert haiku["model"] == "gpt-6-luna"

    def test_maps_adaptive_thinking_to_high_reasoning(self):
        result = anthropic_messages_to_responses(
            {
                "model": "claude-opus-4-1",
                "messages": [],
                "thinking": {"type": "adaptive"},
            }
        )

        assert result["model"] == "gpt-6.1-sol"
        assert result["reasoning"] == {"effort": "high", "summary": "auto"}

    def test_max_tokens_becomes_max_output_tokens(self):
        result = anthropic_messages_to_responses(
            {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 4096}
        )
        assert result["max_output_tokens"] == 4096

    def test_maps_system_tools_and_tool_results(self):
        request = {
            "model": "claude-sonnet-4-5",
            "system": "You are careful.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Ping"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Calling tool."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"q": "Ping"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "Pong",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "tool", "name": "lookup"},
            "stream": True,
        }

        result = anthropic_messages_to_responses(request)

        assert result["instructions"] == "You are careful."
        assert result["stream"] is True
        assert result["tools"] == [
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                "strict": False,
            }
        ]
        assert result["tool_choice"] == {"type": "function", "name": "lookup"}
        assert any(item.get("type") == "function_call" for item in result["input"])
        assert any(item.get("type") == "function_call_output" for item in result["input"])

    def test_preserves_assistant_content_order_around_tool_calls(self):
        request = {
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Ping"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Before"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"q": "Ping"},
                        },
                        {"type": "text", "text": "After"},
                    ],
                },
            ],
        }

        result = anthropic_messages_to_responses(request)

        assert result["input"][1] == {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Before"}],
        }
        assert result["input"][2] == {
            "type": "function_call",
            "id": "fc_toolu_1",
            "call_id": "fc_toolu_1",
            "name": "lookup",
            "arguments": '{"q": "Ping"}',
        }
        assert result["input"][3] == {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "After"}],
        }

    def test_flattens_system_blocks_and_text_content(self):
        request = {
            "model": "claude-sonnet-4-5",
            "system": [
                {"type": "text", "text": "Line one."},
                {"type": "text", "text": "Line two."},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc123",
                            },
                        },
                    ],
                }
            ],
        }

        result = anthropic_messages_to_responses(request)

        assert result["instructions"] == "Line one.\n\nLine two."
        assert result["input"][0]["content"][0] == {"type": "input_text", "text": "Describe this"}
        assert result["input"][0]["content"][1] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,abc123",
        }


class TestResponseStreamTranslator:
    def test_text_streaming(self):
        t = ResponseStreamTranslator("gpt-6.1-sol")

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
        t = ResponseStreamTranslator("gpt-6.1-sol")

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
        # role chunk first, then the tool call header
        assert len(lines) == 2
        chunk = json.loads(lines[1].removeprefix("data: ").strip())
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
        t = ResponseStreamTranslator("gpt-6.1-sol")

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
        # role chunk + final chunk + [DONE]
        assert len(lines) == 3
        chunk = json.loads(lines[1].removeprefix("data: ").strip())
        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert chunk["usage"]["prompt_tokens"] == 100
        assert chunk["usage"]["completion_tokens"] == 50
        assert lines[2].strip() == "data: [DONE]"

    def test_tool_calls_finish_reason(self):
        t = ResponseStreamTranslator("gpt-6.1-sol")

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

    def test_reasoning_delta_exposed_as_reasoning_content(self):
        t = ResponseStreamTranslator("gpt-6.1-sol")

        lines = t.translate_event(
            "response.reasoning_summary_text.delta", {"delta": "Thinking..."}
        )

        chunk = json.loads(lines[-1].removeprefix("data: ").strip())
        assert chunk["choices"][0]["delta"]["reasoning_content"] == "Thinking..."

    def test_incomplete_response_closes_stream_with_length(self):
        t = ResponseStreamTranslator("gpt-6.1-sol")

        lines = t.translate_event(
            "response.incomplete",
            {
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                }
            },
        )

        chunk = json.loads(lines[-2].removeprefix("data: ").strip())
        assert chunk["choices"][0]["finish_reason"] == "length"
        assert lines[-1].strip() == "data: [DONE]"

    def test_content_filter_maps_to_content_filter_finish_reason(self):
        t = ResponseStreamTranslator("gpt-6.1-sol")

        lines = t.translate_event(
            "response.incomplete",
            {
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                }
            },
        )

        chunk = json.loads(lines[-2].removeprefix("data: ").strip())
        assert chunk["choices"][0]["finish_reason"] == "content_filter"

    def test_finish_is_idempotent_after_completed(self):
        t = ResponseStreamTranslator("gpt-6.1-sol")
        t.translate_event("response.completed", {"response": {"status": "completed"}})

        assert t.finish() == []


class TestAnthropicStreamTranslator:
    def test_emits_text_and_tool_events(self):
        translator = AnthropicStreamTranslator("gpt-6.1-sol")

        start_lines = translator.translate_event(
            "response.output_item.added",
            {"item": {"type": "message", "id": "msg_1", "role": "assistant"}},
        )
        text_lines = translator.translate_event("response.output_text.delta", {"delta": "Hello"})
        tool_lines = translator.translate_event(
            "response.output_item.added",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "fc_1",
                    "name": "lookup",
                }
            },
        )

        assert any("message_start" in line for line in start_lines)
        assert any("text_delta" in line for line in text_lines)
        assert any("tool_use" in line for line in tool_lines)

    def test_emits_input_json_deltas_and_message_stop(self):
        translator = AnthropicStreamTranslator("gpt-6.1-sol")
        translator.translate_event(
            "response.output_item.added",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_2",
                    "call_id": "fc_2",
                    "name": "lookup",
                }
            },
        )

        arg_lines = translator.translate_event(
            "response.function_call_arguments.delta",
            {"item_id": "fc_2", "delta": '{"q":"hel'},
        )
        done_lines = translator.translate_event(
            "response.completed",
            {
                "response": {
                    "status": "completed",
                    "usage": {"input_tokens": 3, "output_tokens": 5},
                }
            },
        )

        assert any("input_json_delta" in line for line in arg_lines)
        assert any('"stop_reason": "tool_use"' in line for line in done_lines)
        assert done_lines[-1].startswith("event: message_stop")

    def test_completed_handles_missing_usage_details(self):
        translator = AnthropicStreamTranslator("gpt-6.1-sol")
        translator.translate_event(
            "response.output_item.added",
            {"item": {"type": "message", "id": "msg_1", "role": "assistant"}},
        )

        done_lines = translator.translate_event(
            "response.completed",
            {"response": {"status": "completed", "usage": None, "incomplete_details": None}},
        )

        assert any('"input_tokens": 0' in line for line in done_lines)
        assert done_lines[-1].startswith("event: message_stop")

    def test_tool_only_reply_does_not_add_empty_text_block(self):
        translator = AnthropicStreamTranslator("gpt-6.1-sol")

        start_lines = translator.translate_event(
            "response.output_item.added",
            {"item": {"type": "message", "id": "msg_1", "role": "assistant"}},
        )
        tool_lines = translator.translate_event(
            "response.output_item.added",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "fc_1",
                    "name": "lookup",
                }
            },
        )

        assert not any('"type": "text"' in line for line in start_lines + tool_lines)
        assert translator.build_response()["content"] == [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup",
                "input": {},
            }
        ]

    def test_reasoning_delta_becomes_thinking_block(self):
        translator = AnthropicStreamTranslator("gpt-6.1-sol")

        lines = translator.translate_event(
            "response.reasoning_summary_text.delta", {"delta": "Consider this."}
        )

        assert any('"type": "thinking"' in line for line in lines)
        assert any('"type": "thinking_delta"' in line for line in lines)
        assert translator.build_response()["content"][0]["thinking"] == "Consider this."

    def test_finish_closes_open_message_once(self):
        translator = AnthropicStreamTranslator("gpt-6.1-sol")
        translator.translate_event("response.output_text.delta", {"delta": "Hi"})

        lines = translator.finish()

        assert any("message_delta" in line for line in lines)
        assert lines[-1].startswith("event: message_stop")
        assert translator.finish() == []

    def test_incomplete_maps_to_max_tokens_stop_reason(self):
        translator = AnthropicStreamTranslator("gpt-6.1-sol")
        translator.translate_event("response.output_text.delta", {"delta": "Hi"})

        lines = translator.translate_event(
            "response.incomplete",
            {
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                }
            },
        )

        assert any('"stop_reason": "max_tokens"' in line for line in lines)
        assert lines[-1].startswith("event: message_stop")

