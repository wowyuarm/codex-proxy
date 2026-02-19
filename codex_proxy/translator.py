"""Translate between OpenAI Chat Completions and ChatGPT Responses API formats."""

import json
import time
import uuid
from typing import Any


def chat_to_responses(request: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat Completions request to a Responses API request body.

    Input: standard OpenAI /v1/chat/completions request
    Output: ChatGPT /backend-api/responses request body
    """
    messages = request.get("messages", [])
    instructions = None
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            # System messages become 'instructions'
            instructions = _extract_text(content)

        elif role == "user":
            input_items.append(
                {
                    "role": "user",
                    "content": _to_input_content(content),
                }
            )

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Assistant with tool calls → function_call items
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    call_id = _normalize_tool_call_id(tc.get("id", ""))
                    input_items.append(
                        {
                            "type": "function_call",
                            "id": call_id,
                            "call_id": call_id,
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", ""),
                        }
                    )
            else:
                # Plain assistant message
                text = _extract_text(content)
                if text:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    )

        elif role == "tool":
            # Tool result → function_call_output
            call_id = _normalize_tool_call_id(msg.get("tool_call_id", ""))
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _extract_text(content),
                }
            )

    body: dict[str, Any] = {
        "model": request.get("model", "gpt-5.1"),
        "stream": True,
        "store": False,
        "input": input_items,
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }

    if instructions:
        body["instructions"] = instructions

    if request.get("tools"):
        body["tools"] = _convert_tools(request["tools"])

    if request.get("temperature") is not None:
        body["temperature"] = request["temperature"]

    if request.get("max_tokens") is not None:
        body["max_tokens"] = request["max_tokens"]

    return body


def _extract_text(content: Any) -> str:
    """Extract plain text from content (string or content parts array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "\n".join(parts)
    return str(content) if content else ""


def _to_input_content(content: Any) -> list[dict[str, Any]]:
    """Convert message content to Responses API input content format."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        result = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    result.append({"type": "input_text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    result.append({"type": "input_image", "url": url})
        return result
    return [{"type": "input_text", "text": str(content)}]


def _normalize_tool_call_id(call_id: str) -> str:
    """Ensure tool call ID starts with 'fc_' and is <= 64 chars."""
    if not call_id.startswith("fc_"):
        call_id = f"fc_{call_id}"
    return call_id[:64]


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions tools to Responses API format."""
    result = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", {})
            result.append(
                {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                    "strict": False,
                }
            )
    return result


# --- Response SSE translation ---


def make_chunk(
    chunk_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI Chat Completions streaming chunk."""
    chunk: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def format_sse(data: str) -> str:
    """Format a string as an SSE data line."""
    return f"data: {data}\n\n"


class ResponseStreamTranslator:
    """Stateful translator that converts Responses API SSE events
    into OpenAI Chat Completions SSE chunks."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.sent_role = False
        self.tool_call_index = -1
        # Track function_call item IDs to assign sequential indices
        self._fc_items: dict[str, int] = {}

    def translate_event(self, event_type: str, event_data: dict[str, Any]) -> list[str]:
        """Translate a single Responses API event into SSE lines.

        Returns a list of SSE-formatted strings (may be empty).
        """
        handler = _EVENT_HANDLERS.get(event_type)
        if handler:
            return handler(self, event_data)
        return []

    def _on_output_item_added(self, data: dict[str, Any]) -> list[str]:
        item = data.get("item", {})
        item_type = item.get("type", "")

        if item_type == "message":
            if not self.sent_role:
                self.sent_role = True
                chunk = make_chunk(self.chunk_id, self.model, {"role": "assistant", "content": ""})
                return [format_sse(json.dumps(chunk))]

        elif item_type == "function_call":
            self.tool_call_index += 1
            idx = self.tool_call_index
            item_id = item.get("id", "")
            self._fc_items[item_id] = idx
            call_id = _normalize_tool_call_id(item.get("call_id", item_id))
            delta = {
                "tool_calls": [
                    {
                        "index": idx,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": item.get("name", ""), "arguments": ""},
                    }
                ]
            }
            chunk = make_chunk(self.chunk_id, self.model, delta)
            return [format_sse(json.dumps(chunk))]

        return []

    def _on_text_delta(self, data: dict[str, Any]) -> list[str]:
        delta_text = data.get("delta", "")
        if not delta_text:
            return []
        if not self.sent_role:
            self.sent_role = True
        chunk = make_chunk(self.chunk_id, self.model, {"content": delta_text})
        return [format_sse(json.dumps(chunk))]

    def _on_function_args_delta(self, data: dict[str, Any]) -> list[str]:
        delta_text = data.get("delta", "")
        if not delta_text:
            return []
        item_id = data.get("item_id", "")
        idx = self._fc_items.get(item_id, self.tool_call_index)
        delta = {
            "tool_calls": [
                {
                    "index": idx,
                    "function": {"arguments": delta_text},
                }
            ]
        }
        chunk = make_chunk(self.chunk_id, self.model, delta)
        return [format_sse(json.dumps(chunk))]

    def _on_completed(self, data: dict[str, Any]) -> list[str]:
        response = data.get("response", {})
        usage_data = response.get("usage", {})

        # Map stop reason
        status = response.get("status", "completed")
        finish_reason = "stop"
        if status == "incomplete":
            finish_reason = "length"
        if self.tool_call_index >= 0:
            finish_reason = "tool_calls"

        usage = None
        if usage_data:
            input_details = usage_data.get("input_tokens_details", {})
            usage = {
                "prompt_tokens": usage_data.get("input_tokens", 0),
                "completion_tokens": usage_data.get("output_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
                "prompt_tokens_details": {
                    "cached_tokens": input_details.get("cached_tokens", 0),
                },
            }

        chunk = make_chunk(self.chunk_id, self.model, {}, finish_reason, usage)
        lines = [format_sse(json.dumps(chunk))]
        lines.append(format_sse("[DONE]"))
        return lines

    def _on_error(self, data: dict[str, Any]) -> list[str]:
        chunk = make_chunk(self.chunk_id, self.model, {}, "stop")
        chunk["error"] = data
        return [format_sse(json.dumps(chunk)), format_sse("[DONE]")]


_EVENT_HANDLERS: dict[str, Any] = {
    "response.output_item.added": ResponseStreamTranslator._on_output_item_added,
    "response.output_text.delta": ResponseStreamTranslator._on_text_delta,
    "response.content_part.delta": ResponseStreamTranslator._on_text_delta,
    "response.function_call_arguments.delta": ResponseStreamTranslator._on_function_args_delta,
    "response.completed": ResponseStreamTranslator._on_completed,
    "error": ResponseStreamTranslator._on_error,
    "response.failed": ResponseStreamTranslator._on_error,
}
