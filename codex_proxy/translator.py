"""Translate between OpenAI Chat Completions, Anthropic Messages, and the
ChatGPT Responses API formats."""

import json
import logging
import time
import uuid
from typing import Any

from codex_proxy.config import DEFAULT_MODEL
from codex_proxy.models import normalize_reasoning_effort, resolve_model_name

log = logging.getLogger(__name__)

# Chat Completions parameters the Codex backend rejects outright. They are
# dropped (with a warning) instead of forwarded so clients get an answer
# rather than a 400 from upstream.
_UNSUPPORTED_PARAMS = {
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "logprobs",
    "top_logprobs",
    "seed",
    "n",
    "stream_options",
    "user",
    "modalities",
    "audio",
    "prediction",
    "prompt_cache_retention",
    "safety_identifier",
}

_SERVICE_TIERS = {"default", "priority"}


def chat_to_responses(
    request: dict[str, Any], models: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Convert a Chat Completions request to a Responses API request body."""
    dropped = _UNSUPPORTED_PARAMS & request.keys()
    if dropped:
        log.warning("Dropping unsupported parameters: %s", ", ".join(sorted(dropped)))

    input_items: list[dict[str, Any]] = []
    instruction_parts: list[str] = []

    for msg in request.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in {"system", "developer"}:
            text = _extract_text(content)
            if text:
                instruction_parts.append(text)

        elif role == "user":
            input_items.append({"role": "user", "content": _to_input_content(content)})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            text = _extract_text(content)
            if text:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for tool_call in tool_calls or []:
                if not isinstance(tool_call, dict):
                    continue
                fn = tool_call.get("function", {})
                call_id = _normalize_tool_call_id(tool_call.get("id", ""))
                input_items.append(
                    {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", ""),
                    }
                )

        elif role == "tool":
            call_id = _normalize_tool_call_id(msg.get("tool_call_id", ""))
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _extract_text(content),
                }
            )

    model = resolve_model_name(request.get("model"), models or [])

    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "store": False,
        "instructions": "\n\n".join(instruction_parts) or "You are a helpful assistant.",
        "input": input_items,
        "include": ["reasoning.encrypted_content"],
    }

    reasoning = _build_reasoning(request, model, models or [])
    if reasoning:
        body["reasoning"] = reasoning

    text = _build_text_controls(request)
    if text:
        body["text"] = text

    max_output_tokens = first_int(
        request.get("max_output_tokens"),
        request.get("max_completion_tokens"),
        request.get("max_tokens"),
    )
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens

    service_tier = request.get("service_tier")
    if isinstance(service_tier, str) and service_tier in _SERVICE_TIERS:
        body["service_tier"] = service_tier

    if request.get("tools"):
        body["tools"] = _convert_tools(request["tools"])
        body["tool_choice"] = request.get("tool_choice", "auto")
        body["parallel_tool_calls"] = request.get("parallel_tool_calls", True)

    return body


def first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _build_reasoning(
    request: dict[str, Any], model: str, models: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Build the `reasoning` block from OpenAI or Claude-style request fields."""
    reasoning: dict[str, Any] = {}

    effort = normalize_reasoning_effort(request.get("reasoning_effort"), model, models)
    if effort:
        reasoning["effort"] = effort

    raw = request.get("reasoning")
    if isinstance(raw, dict):
        effort = normalize_reasoning_effort(raw.get("effort"), model, models)
        if effort:
            reasoning["effort"] = effort
        summary = raw.get("summary")
        if isinstance(summary, str) and summary in {"auto", "concise", "detailed"}:
            reasoning["summary"] = summary

    # Clients that want visible thinking ask for a summary; without this the
    # backend sends no reasoning events at all.
    if "effort" in reasoning and "summary" not in reasoning:
        reasoning["summary"] = "auto"

    return reasoning or None


def _build_text_controls(request: dict[str, Any]) -> dict[str, Any]:
    """Merge `response_format` and `verbosity` into the Responses `text` block."""
    text: dict[str, Any] = {}

    raw_text = request.get("text")
    if isinstance(raw_text, dict):
        text.update(raw_text)

    fmt = _convert_response_format(request.get("response_format"))
    if fmt:
        text["format"] = fmt

    verbosity = request.get("verbosity")
    if isinstance(verbosity, str) and verbosity in {"low", "medium", "high"}:
        text["verbosity"] = verbosity

    return text


def _convert_response_format(response_format: Any) -> dict[str, Any] | None:
    """Convert Chat Completions `response_format` into Responses `text.format`."""
    if not isinstance(response_format, dict):
        return None

    fmt_type = response_format.get("type")
    if fmt_type == "json_object":
        return {"type": "json_object"}
    if fmt_type == "text":
        return None
    if fmt_type == "json_schema":
        schema_spec = response_format.get("json_schema")
        if not isinstance(schema_spec, dict):
            return None
        converted: dict[str, Any] = {
            "type": "json_schema",
            "name": schema_spec.get("name") or "response",
            "schema": schema_spec.get("schema") or {},
            "strict": bool(schema_spec.get("strict")),
        }
        description = schema_spec.get("description")
        if isinstance(description, str) and description:
            converted["description"] = description
        return converted
    return None


def _extract_text(content: Any) -> str:
    """Extract plain text from content (string or content parts array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "output_text"}:
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
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "output_text"}:
                result.append({"type": "input_text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                    detail = image_url.get("detail")
                else:
                    url = image_url or ""
                    detail = None

                input_image: dict[str, Any] = {"type": "input_image", "image_url": url}
                if detail:
                    input_image["detail"] = detail
                result.append(input_image)
            elif part.get("type") == "image":
                input_image = _anthropic_image_to_input_image(part)
                if input_image is not None:
                    result.append(input_image)
        return result
    return [{"type": "input_text", "text": str(content)}]


def anthropic_messages_to_responses(
    request: dict[str, Any], models: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Convert Anthropic Messages payloads into ChatGPT Responses payloads."""
    system = request.get("system")
    messages = request.get("messages", [])
    model = resolve_model_name(request.get("model"), models or [])
    input_items: list[dict[str, Any]] = []

    def flush_user_content(content: list[dict[str, Any]]) -> None:
        if content:
            input_items.append({"role": "user", "content": list(content)})
            content.clear()

    def flush_assistant_content(content: list[dict[str, Any]]) -> None:
        if content:
            input_items.append({"type": "message", "role": "assistant", "content": list(content)})
            content.clear()

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
        content = msg.get("content", [])

        if role == "user":
            text_or_media: list[dict[str, Any]] = []
            for part in _as_content_blocks(content):
                if part.get("type") == "tool_result":
                    flush_user_content(text_or_media)
                    call_id = _normalize_tool_call_id(part.get("tool_use_id", ""))
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": _extract_tool_result_output(part.get("content")),
                        }
                    )
                    continue

                normalized_part = _anthropic_part_to_input_content(part)
                if normalized_part is not None:
                    text_or_media.append(normalized_part)

            flush_user_content(text_or_media)

        elif role == "assistant":
            assistant_text: list[dict[str, Any]] = []
            for part in _as_content_blocks(content):
                part_type = part.get("type")
                if part_type == "tool_use":
                    flush_assistant_content(assistant_text)
                    call_id = _normalize_tool_call_id(part.get("id", ""))
                    input_items.append(
                        {
                            "type": "function_call",
                            "id": call_id,
                            "call_id": call_id,
                            "name": part.get("name", ""),
                            "arguments": json.dumps(part.get("input", {}), ensure_ascii=False),
                        }
                    )
                    continue
                # Thinking blocks cannot be replayed upstream: we have no
                # encrypted reasoning payload to pair them with.
                if part_type in {"thinking", "redacted_thinking"}:
                    continue

                normalized_part = _anthropic_part_to_assistant_content(part)
                if normalized_part is not None:
                    assistant_text.append(normalized_part)

            flush_assistant_content(assistant_text)

    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "store": False,
        "instructions": _extract_anthropic_system_text(system) or "You are a helpful assistant.",
        "input": input_items,
        "include": ["reasoning.encrypted_content"],
    }

    max_tokens = first_int(request.get("max_tokens"))
    if max_tokens is not None:
        body["max_output_tokens"] = max_tokens

    tools = request.get("tools")
    if isinstance(tools, list) and tools:
        body["tools"] = _convert_anthropic_tools(tools)
        body["tool_choice"] = _convert_anthropic_tool_choice(request.get("tool_choice"))
        body["parallel_tool_calls"] = False

    reasoning = _convert_anthropic_thinking(request.get("thinking"), model, models or [])
    if reasoning:
        body["reasoning"] = reasoning

    return body


def _normalize_tool_call_id(call_id: str) -> str:
    """Ensure tool call ID starts with 'fc_' and is <= 64 chars."""
    if not call_id.startswith("fc_"):
        call_id = f"fc_{call_id}"
    return call_id[:64]


def _tool_call_id_to_anthropic(call_id: str) -> str:
    """Convert an upstream function call ID into an Anthropic-style tool_use ID."""
    normalized = _normalize_tool_call_id(call_id)
    suffix = normalized[3:] if normalized.startswith("fc_") else normalized
    suffix = suffix.replace("-", "").replace("_", "")
    return f"toolu_{suffix[:40] or uuid.uuid4().hex[:12]}"


def _as_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [part for part in content if isinstance(part, dict)]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _extract_anthropic_system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
    return ""


def _anthropic_part_to_input_content(part: dict[str, Any]) -> dict[str, Any] | None:
    part_type = part.get("type")
    if part_type == "text":
        return {"type": "input_text", "text": part.get("text", "")}
    if part_type == "image":
        return _anthropic_image_to_input_image(part)
    return None


def _anthropic_part_to_assistant_content(part: dict[str, Any]) -> dict[str, Any] | None:
    if part.get("type") == "text":
        return {"type": "output_text", "text": part.get("text", "")}
    return None


def _anthropic_image_to_input_image(part: dict[str, Any]) -> dict[str, Any] | None:
    source = part.get("source")
    if not isinstance(source, dict):
        return None

    if source.get("type") == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {
            "type": "input_image",
            "image_url": f"data:{media_type};base64,{data}",
        }

    if source.get("type") == "url":
        url = source.get("url", "")
        if isinstance(url, str) and url:
            return {"type": "input_image", "image_url": url}

    return None


def _extract_tool_result_output(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def _convert_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        result.append(
            {
                "type": "function",
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
                "strict": _normalize_tool_strict(tool.get("strict")),
            }
        )
    return result


def _convert_anthropic_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return "auto"

    choice_type = tool_choice.get("type")
    if choice_type == "tool":
        return {"type": "function", "name": tool_choice.get("name", "")}
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    return "auto"


def _convert_anthropic_thinking(
    thinking: Any, model: str = DEFAULT_MODEL, models: list[dict[str, Any]] | None = None
) -> dict[str, str] | None:
    if not thinking:
        return None

    if isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type == "adaptive":
            return {"effort": "high", "summary": "auto"}
        if thinking_type == "enabled":
            budget = thinking.get("budget_tokens")
            if not isinstance(budget, int):
                budget = thinking.get("budgetTokens")
            if isinstance(budget, int):
                if budget >= 16000:
                    return {"effort": "high", "summary": "auto"}
                if budget <= 2048:
                    return {"effort": "low", "summary": "auto"}
            return {"effort": "medium", "summary": "auto"}
        if thinking_type == "disabled":
            return None

    effort = normalize_reasoning_effort("medium", model, models or [])
    return {"effort": effort or "medium", "summary": "auto"}


def _normalize_tool_strict(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions tools to Responses API format."""
    result = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function", {})
        result.append(
            {
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
                "strict": _normalize_tool_strict(fn.get("strict")),
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
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def format_sse(data: str) -> str:
    """Format a string as an SSE data line."""
    return f"data: {data}\n\n"


def build_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    """Translate Responses usage into Chat Completions usage."""
    usage_data = response.get("usage")
    if not isinstance(usage_data, dict) or not usage_data:
        return None
    input_details = usage_data.get("input_tokens_details") or {}
    output_details = usage_data.get("output_tokens_details") or {}
    return {
        "prompt_tokens": usage_data.get("input_tokens", 0),
        "completion_tokens": usage_data.get("output_tokens", 0),
        "total_tokens": usage_data.get("total_tokens", 0),
        "prompt_tokens_details": {"cached_tokens": input_details.get("cached_tokens", 0)},
        "completion_tokens_details": {
            "reasoning_tokens": output_details.get("reasoning_tokens", 0)
        },
    }


def _finish_reason_for(response: dict[str, Any], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    status = response.get("status")
    if status == "incomplete":
        details = response.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        if reason == "content_filter":
            return "content_filter"
        return "length"
    return "stop"


class ResponseStreamTranslator:
    """Converts Responses SSE events into OpenAI Chat Completions SSE chunks."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.sent_role = False
        self.tool_call_index = -1
        self._fc_items: dict[str, int] = {}
        self.finished = False

    def translate_event(self, event_type: str, event_data: dict[str, Any]) -> list[str]:
        handler = _EVENT_HANDLERS.get(event_type)
        if handler:
            return handler(self, event_data)
        return []

    def _emit_role(self) -> list[str]:
        if self.sent_role:
            return []
        self.sent_role = True
        chunk = make_chunk(self.chunk_id, self.model, {"role": "assistant", "content": ""})
        return [format_sse(json.dumps(chunk))]

    def _on_output_item_added(self, data: dict[str, Any]) -> list[str]:
        item = data.get("item", {})
        item_type = item.get("type", "")

        if item_type == "message":
            return self._emit_role()

        if item_type == "function_call":
            self.tool_call_index += 1
            idx = self.tool_call_index
            self._fc_items[item.get("id", "")] = idx
            call_id = _normalize_tool_call_id(item.get("call_id") or item.get("id", ""))
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
            lines = self._emit_role()
            lines.append(format_sse(json.dumps(make_chunk(self.chunk_id, self.model, delta))))
            return lines

        return []

    def _on_text_delta(self, data: dict[str, Any]) -> list[str]:
        delta_text = data.get("delta", "")
        if not delta_text:
            return []
        lines = self._emit_role()
        chunk = make_chunk(self.chunk_id, self.model, {"content": delta_text})
        lines.append(format_sse(json.dumps(chunk)))
        return lines

    def _on_reasoning_delta(self, data: dict[str, Any]) -> list[str]:
        """Expose reasoning as `reasoning_content`, the field LiteLLM and
        OpenRouter-style clients read."""
        delta_text = data.get("delta", "")
        if not delta_text:
            return []
        lines = self._emit_role()
        chunk = make_chunk(self.chunk_id, self.model, {"reasoning_content": delta_text})
        lines.append(format_sse(json.dumps(chunk)))
        return lines

    def _on_function_args_delta(self, data: dict[str, Any]) -> list[str]:
        delta_text = data.get("delta", "")
        if not delta_text:
            return []
        item_id = data.get("item_id", "")
        idx = self._fc_items.get(item_id, self.tool_call_index)
        delta = {"tool_calls": [{"index": idx, "function": {"arguments": delta_text}}]}
        return [format_sse(json.dumps(make_chunk(self.chunk_id, self.model, delta)))]

    def _on_terminal(self, data: dict[str, Any]) -> list[str]:
        """Handle `response.completed` and `response.incomplete` alike.

        Treating `response.incomplete` as terminal matters: without it a
        truncated response leaves the stream open and the client hangs.
        """
        response = data.get("response", {})
        finish_reason = _finish_reason_for(response, self.tool_call_index >= 0)
        usage = build_usage(response)
        lines = self._emit_role()
        chunk = make_chunk(self.chunk_id, self.model, {}, finish_reason, usage)
        lines.append(format_sse(json.dumps(chunk)))
        lines.append(format_sse("[DONE]"))
        self.finished = True
        return lines

    def _on_error(self, data: dict[str, Any]) -> list[str]:
        chunk = make_chunk(self.chunk_id, self.model, {}, "stop")
        chunk["error"] = data
        self.finished = True
        return [format_sse(json.dumps(chunk)), format_sse("[DONE]")]

    def finish(self) -> list[str]:
        """Close the stream when upstream ends without a terminal event.

        Idempotent: a completed stream already emitted its final chunk, so a
        second call would leave the client with two `finish_reason` chunks.
        """
        if self.finished:
            return []
        self.finished = True
        lines = self._emit_role()
        chunk = make_chunk(self.chunk_id, self.model, {}, "stop")
        lines.append(format_sse(json.dumps(chunk)))
        lines.append(format_sse("[DONE]"))
        return lines


_EVENT_HANDLERS: dict[str, Any] = {
    "response.output_item.added": ResponseStreamTranslator._on_output_item_added,
    "response.output_text.delta": ResponseStreamTranslator._on_text_delta,
    "response.content_part.delta": ResponseStreamTranslator._on_text_delta,
    "response.reasoning_summary_text.delta": ResponseStreamTranslator._on_reasoning_delta,
    "response.reasoning_text.delta": ResponseStreamTranslator._on_reasoning_delta,
    "response.function_call_arguments.delta": ResponseStreamTranslator._on_function_args_delta,
    "response.completed": ResponseStreamTranslator._on_terminal,
    "response.incomplete": ResponseStreamTranslator._on_terminal,
    "error": ResponseStreamTranslator._on_error,
    "response.failed": ResponseStreamTranslator._on_error,
}


def _format_anthropic_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class AnthropicStreamTranslator:
    """Translate Responses SSE events into Anthropic Messages SSE events."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.started = False
        self.closed = False
        self.current_block_index: int | None = None
        self.current_block_type: str | None = None
        self.current_item_id: str | None = None
        self.current_stop_reason = "end_turn"
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.content: list[dict[str, Any]] = []
        self.error: dict[str, Any] | None = None
        self._tool_call_by_item_id: dict[str, str] = {}
        self._tool_input_buffers: dict[str, str] = {}

    def translate_event(self, event_type: str, event_data: dict[str, Any]) -> list[str]:
        handler = _ANTHROPIC_EVENT_HANDLERS.get(event_type)
        if handler:
            return handler(self, event_data)
        return []

    def build_response(self) -> dict[str, Any]:
        return {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": self.content or [{"type": "text", "text": ""}],
            "stop_reason": self.current_stop_reason,
            "stop_sequence": None,
            "usage": dict(self.usage),
        }

    def _ensure_started(self) -> list[str]:
        if self.started:
            return []
        self.started = True
        return [
            _format_anthropic_sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def _close_current_block(self) -> list[str]:
        if self.current_block_index is None:
            return []
        index = self.current_block_index
        self.current_block_index = None
        self.current_block_type = None
        self.current_item_id = None
        return [
            _format_anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": index})
        ]

    def _start_text_block(self) -> list[str]:
        if self.current_block_type == "text":
            return []
        lines = self._ensure_started()
        lines.extend(self._close_current_block())
        self.current_block_index = len(self.content)
        self.current_block_type = "text"
        self.current_item_id = None
        self.content.append({"type": "text", "text": ""})
        lines.append(
            _format_anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.current_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        return lines

    def _start_thinking_block(self) -> list[str]:
        if self.current_block_type == "thinking":
            return []
        lines = self._ensure_started()
        lines.extend(self._close_current_block())
        self.current_block_index = len(self.content)
        self.current_block_type = "thinking"
        self.current_item_id = None
        self.content.append({"type": "thinking", "thinking": "", "signature": ""})
        lines.append(
            _format_anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.current_block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        )
        return lines

    def _start_tool_block(self, item: dict[str, Any]) -> list[str]:
        lines = self._ensure_started()
        lines.extend(self._close_current_block())
        item_id = item.get("id", "")
        tool_id = _tool_call_id_to_anthropic(item.get("call_id") or item_id)
        self._tool_call_by_item_id[item_id] = tool_id
        self._tool_input_buffers[item_id] = ""
        self.current_stop_reason = "tool_use"
        self.current_block_index = len(self.content)
        self.current_block_type = "tool_use"
        self.current_item_id = item_id
        self.content.append(
            {"type": "tool_use", "id": tool_id, "name": item.get("name", ""), "input": {}}
        )
        lines.append(
            _format_anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.current_block_index,
                    "content_block": dict(self.content[self.current_block_index]),
                },
            )
        )
        return lines

    def _on_output_item_added(self, data: dict[str, Any]) -> list[str]:
        item = data.get("item", {})
        item_type = item.get("type")
        if item_type == "message":
            return self._ensure_started()
        if item_type == "function_call":
            return self._start_tool_block(item)
        return []

    def _on_text_delta(self, data: dict[str, Any]) -> list[str]:
        delta_text = data.get("delta", "")
        if not delta_text:
            return []
        lines = self._start_text_block()
        assert self.current_block_index is not None
        self.content[self.current_block_index]["text"] += delta_text
        lines.append(
            _format_anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.current_block_index,
                    "delta": {"type": "text_delta", "text": delta_text},
                },
            )
        )
        return lines

    def _on_reasoning_delta(self, data: dict[str, Any]) -> list[str]:
        delta_text = data.get("delta", "")
        if not delta_text:
            return []
        lines = self._start_thinking_block()
        assert self.current_block_index is not None
        self.content[self.current_block_index]["thinking"] += delta_text
        lines.append(
            _format_anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.current_block_index,
                    "delta": {"type": "thinking_delta", "thinking": delta_text},
                },
            )
        )
        return lines

    def _on_function_args_delta(self, data: dict[str, Any]) -> list[str]:
        item_id = data.get("item_id", "")
        delta_text = data.get("delta", "")
        if not item_id or not delta_text:
            return []
        index = self.current_block_index
        if item_id != self.current_item_id:
            index = next(
                (
                    idx
                    for idx, block in enumerate(self.content)
                    if block.get("type") == "tool_use"
                    and block.get("id") == self._tool_call_by_item_id.get(item_id)
                ),
                None,
            )
        if index is None:
            return []
        self._tool_input_buffers[item_id] = self._tool_input_buffers.get(item_id, "") + delta_text
        try:
            self.content[index]["input"] = json.loads(self._tool_input_buffers[item_id])
        except json.JSONDecodeError:
            pass
        return [
            _format_anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": delta_text},
                },
            )
        ]

    def _on_terminal(self, data: dict[str, Any]) -> list[str]:
        """Close the message for both `response.completed` and `response.incomplete`."""
        response = data.get("response", {})
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        incomplete = (
            response.get("incomplete_details")
            if isinstance(response.get("incomplete_details"), dict)
            else {}
        )
        reason = incomplete.get("reason")
        if self.current_stop_reason != "tool_use":
            if reason == "content_filter":
                self.current_stop_reason = "refusal"
            elif reason in {"max_tokens", "max_output_tokens"} or (
                response.get("status") == "incomplete"
            ):
                self.current_stop_reason = "max_tokens"
            elif reason == "model_context_window_exceeded":
                self.current_stop_reason = "model_context_window_exceeded"

        self.usage = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
        lines = self._ensure_started()
        lines.extend(self._close_current_block())
        lines.append(
            _format_anthropic_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": self.current_stop_reason, "stop_sequence": None},
                    "usage": dict(self.usage),
                },
            )
        )
        lines.append(_format_anthropic_sse("message_stop", {"type": "message_stop"}))
        self.closed = True
        return lines

    def finish(self) -> list[str]:
        """Close the message when upstream ends without a terminal event.

        Idempotent: a closed message already emitted `message_stop`.
        """
        if self.closed:
            return []
        return self._on_terminal({})

    def _on_error(self, data: dict[str, Any]) -> list[str]:
        payload = data.get("error", data)
        self.error = payload if isinstance(payload, dict) else {"message": str(payload)}
        self.closed = True
        return [
            _format_anthropic_sse(
                "error",
                {"type": "error", "error": self.error},
            )
        ]


_ANTHROPIC_EVENT_HANDLERS: dict[str, Any] = {
    "response.output_item.added": AnthropicStreamTranslator._on_output_item_added,
    "response.output_text.delta": AnthropicStreamTranslator._on_text_delta,
    "response.content_part.delta": AnthropicStreamTranslator._on_text_delta,
    "response.reasoning_summary_text.delta": AnthropicStreamTranslator._on_reasoning_delta,
    "response.reasoning_text.delta": AnthropicStreamTranslator._on_reasoning_delta,
    "response.function_call_arguments.delta": AnthropicStreamTranslator._on_function_args_delta,
    "response.completed": AnthropicStreamTranslator._on_terminal,
    "response.incomplete": AnthropicStreamTranslator._on_terminal,
    "error": AnthropicStreamTranslator._on_error,
    "response.failed": AnthropicStreamTranslator._on_error,
}
