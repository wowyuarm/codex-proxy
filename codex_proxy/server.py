"""aiohttp proxy server exposing OpenAI-compatible and Anthropic-compatible endpoints."""

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from aiohttp import web
from curl_cffi import CurlOpt
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import RequestException

from codex_proxy.auth import ensure_credentials, extract_account_id
from codex_proxy.config import FALLBACK_MODELS, RESPONSES_ENDPOINT
from codex_proxy.models import ModelCatalog, resolve_model_name, visible_model_ids
from codex_proxy.translator import (
    AnthropicStreamTranslator,
    ResponseStreamTranslator,
    anthropic_messages_to_responses,
    chat_to_responses,
    first_int,
)

log = logging.getLogger(__name__)

# Parameters the Codex backend rejects on /responses. Dropping them keeps
# OpenAI-compatible clients working instead of failing the whole request.
_UNSUPPORTED_RESPONSES_PARAMS = {
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
    "background",
    "truncation",
    "metadata",
}
_SERVICE_TIERS = {"default", "priority"}
_UPSTREAM_TIMEOUT_SECONDS = 120
_MAX_UPSTREAM_ATTEMPTS = 2
_RETRIABLE_UPSTREAM_ERROR_CODES = {18, 35, 52, 55, 56, 92}
_RETRIABLE_UPSTREAM_EVENT_MARKERS = (
    "serviceunavailable",
    "service_unavailable",
    "api_connection_error",
    "connection error",
    "temporarily unavailable",
    "please retry",
    "an error occurred while processing your request",
)

_PUBLIC_PATHS = {"/health"}

MODEL_CATALOG_KEY: web.AppKey[ModelCatalog] = web.AppKey("model_catalog", ModelCatalog)
UPSTREAM_SESSION_KEY: web.AppKey[AsyncSession] = web.AppKey("upstream_session", AsyncSession)


# --- API key authentication middleware ---


def _configured_api_key() -> str | None:
    return os.environ.get("CODEX_PROXY_API_KEY")


@web.middleware
async def api_key_middleware(request: web.Request, handler):
    """Reject requests without a valid API key (when CODEX_PROXY_API_KEY is set)."""
    api_key = _configured_api_key()
    if not api_key or request.path in _PUBLIC_PATHS:
        return await handler(request)

    auth = request.headers.get("Authorization", "")
    provided = None
    if auth.startswith("Bearer "):
        provided = auth[7:]
    elif request.headers.get("X-API-Key"):
        provided = request.headers["X-API-Key"]

    if provided != api_key:
        return web.json_response(
            {"error": {"message": "Invalid or missing API key", "type": "auth_error"}},
            status=401,
        )
    return await handler(request)


# --- Upstream payload normalization for /responses ---


def _normalize_tool_strict(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def _normalize_input_content(content: Any) -> Any:
    """Normalize message content parts for the ChatGPT codex upstream."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return content

    normalized: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            normalized.append(part)
            continue

        if part.get("type") == "text":
            normalized.append({"type": "input_text", "text": part.get("text", "")})
            continue

        if part.get("type") == "image_url":
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
            normalized.append(input_image)
            continue

        normalized.append(part)
    return normalized


def _normalize_responses_input(input_value: Any) -> Any:
    """Normalize /responses input items from OpenAI-style payloads."""
    if isinstance(input_value, str):
        return [
            {"role": "user", "content": [{"type": "input_text", "text": input_value}]}
        ]
    if not isinstance(input_value, list):
        return input_value

    normalized: list[Any] = []
    converted = 0
    for item in input_value:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        new_item = dict(item)
        if new_item.get("type") == "message":
            converted += 1
            new_item.pop("type", None)
        if "content" in new_item:
            new_item["content"] = _normalize_input_content(new_item["content"])
        normalized.append(new_item)

    if converted:
        log.info("Normalized %d OpenAI-style input message items on /responses", converted)
    return normalized


def _normalize_responses_tools(tools: Any) -> Any:
    """Accept both Chat Completions and Responses tool schemas on /responses."""
    if not isinstance(tools, list):
        return tools

    converted_count = 0
    normalized_tools: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized_tools.append(tool)
            continue

        fn = tool.get("function")
        if tool.get("type") == "function" and isinstance(fn, dict):
            normalized_tools.append(
                {
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                    "strict": _normalize_tool_strict(fn.get("strict")),
                }
            )
            converted_count += 1
            continue

        if tool.get("type") == "function":
            direct = dict(tool)
            direct["strict"] = _normalize_tool_strict(direct.get("strict"))
            normalized_tools.append(direct)
            continue

        normalized_tools.append(tool)

    if converted_count:
        log.info("Normalized %d chat-style tools on /responses", converted_count)
    return normalized_tools


def _normalize_responses_tool_choice(tool_choice: Any) -> Any:
    """Normalize chat.completions-style tool_choice for /responses."""
    if not isinstance(tool_choice, dict):
        return tool_choice

    fn = tool_choice.get("function")
    if tool_choice.get("type") == "function" and isinstance(fn, dict):
        return {"type": "function", "name": fn.get("name", "")}

    return tool_choice


def _normalize_response_format(response_format: Any) -> dict[str, Any] | None:
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") == "json_object":
        return {"type": "json_object"}
    if response_format.get("type") == "json_schema":
        spec = response_format.get("json_schema")
        if isinstance(spec, dict):
            return {
                "type": "json_schema",
                "name": spec.get("name") or "response",
                "schema": spec.get("schema") or {},
                "strict": bool(spec.get("strict")),
            }
    return None


def _normalize_responses_body(
    raw_body: dict[str, Any], models: list[dict[str, Any]] | None = None
) -> tuple[dict[str, Any], bool]:
    """Normalize an OpenAI-style /responses request for the ChatGPT codex upstream."""
    body = dict(raw_body)
    client_stream = bool(body.get("stream"))

    dropped = _UNSUPPORTED_RESPONSES_PARAMS & body.keys()
    for key in dropped:
        body.pop(key, None)
    if dropped:
        log.warning("Dropping unsupported /responses parameters: %s", ", ".join(sorted(dropped)))

    if not body.get("instructions"):
        body["instructions"] = "You are a helpful assistant."

    if body.get("store") is not False:
        body["store"] = False

    # The max token budget is supported upstream under `max_output_tokens`;
    # the Chat Completions spellings are folded into it.
    if "max_output_tokens" not in body:
        budget = first_int(raw_body.get("max_completion_tokens"), raw_body.get("max_tokens"))
        if budget is not None:
            body["max_output_tokens"] = budget
    body.pop("max_tokens", None)
    body.pop("max_completion_tokens", None)

    service_tier = body.get("service_tier")
    if isinstance(service_tier, str) and service_tier in _SERVICE_TIERS:
        body["service_tier"] = service_tier
    else:
        body.pop("service_tier", None)

    if "input" in body:
        body["input"] = _normalize_responses_input(body.get("input"))

    if "tools" in body:
        body["tools"] = _normalize_responses_tools(body.get("tools"))
    if "tool_choice" in body:
        body["tool_choice"] = _normalize_responses_tool_choice(body.get("tool_choice"))
    if body.get("tools") and "parallel_tool_calls" not in body:
        # Streaming chunks carry a fixed tool index, so sequential calls avoid
        # merging distinct calls in downstream aggregators.
        body["parallel_tool_calls"] = False

    # `response_format` is not a Responses field; it belongs in `text.format`.
    body.pop("response_format", None)
    text = body.get("text") if isinstance(body.get("text"), dict) else {}
    fmt = _normalize_response_format(raw_body.get("response_format"))
    if fmt:
        text = {**text, "format": fmt}
    if text:
        body["text"] = text

    body["model"] = resolve_model_name(body.get("model"), models or [])

    # ChatGPT Codex requires stream=true; non-streaming clients are served by
    # aggregating the upstream SSE stream locally.
    body["stream"] = True
    return body, client_stream


# --- Shared upstream plumbing ---


def _build_upstream_headers(credentials: dict[str, Any], accept: str) -> dict[str, str]:
    """Build auth headers for ChatGPT backend requests."""
    access_token = credentials["access_token"]
    account_id = credentials.get("account_id") or extract_account_id(access_token)
    return {
        "Authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "Content-Type": "application/json",
        "Accept": accept,
    }


def _make_stream_response() -> web.StreamResponse:
    response = web.StreamResponse()
    response.content_type = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


async def _ensure_stream_prepared(
    response: web.StreamResponse, request: web.Request
) -> web.StreamResponse:
    if not response.prepared:
        await response.prepare(request)
    return response


def _build_error_payload(message: str, error_type: str, detail: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"message": message, "type": error_type}
    if detail:
        error["detail"] = detail
    return {"error": error}


def _build_anthropic_error_payload(
    message: str, error_type: str, detail: str | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"message": message, "type": error_type}
    if detail:
        error["detail"] = detail
    return {"type": "error", "error": error}


def _is_retriable_upstream_error(exc: Exception) -> bool:
    """Return whether an upstream curl/network error is worth one reconnect attempt."""
    code = getattr(exc, "code", None)
    if isinstance(exc, RequestException) and code in _RETRIABLE_UPSTREAM_ERROR_CODES:
        return True

    lowered = str(exc).lower()
    return "connection reset by peer" in lowered or "recv failure" in lowered


def _should_retry_upstream_error(exc: Exception, attempt: int, sent_to_client: bool) -> bool:
    """Retry only before any downstream bytes are sent."""
    return (
        not sent_to_client
        and attempt < _MAX_UPSTREAM_ATTEMPTS
        and _is_retriable_upstream_error(exc)
    )


def _is_retriable_upstream_event(event_type: str, event_data: dict[str, Any]) -> bool:
    if event_type not in {"error", "response.failed"}:
        return False
    lowered = json.dumps(event_data, ensure_ascii=False).lower()
    return any(marker in lowered for marker in _RETRIABLE_UPSTREAM_EVENT_MARKERS)


async def _read_upstream_text(upstream: Any) -> str:
    try:
        chunks = []
        async for chunk in upstream.aiter_content():
            chunks.append(chunk.decode(errors="replace") if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)
    except Exception:
        text = getattr(upstream, "text", "")
        if isinstance(text, bytes):
            return text.decode(errors="replace")
        return str(text)


def _extract_upstream_error(error_text: str) -> tuple[str, str]:
    """Pull the human-readable message and type out of an upstream error body."""
    try:
        payload = json.loads(error_text)
    except json.JSONDecodeError:
        return error_text[:500], "upstream_error"

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail, "invalid_request_error"

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error_text[:500]
        return str(message), str(error.get("type") or "upstream_error")

    return error_text[:500], "upstream_error"


async def _iter_upstream_events(upstream: Any) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded `data:` payloads from an upstream SSE stream."""
    async for line_bytes in upstream.aiter_lines():
        line = line_bytes.decode() if isinstance(line_bytes, bytes) else line_bytes
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            return
        try:
            event_data = json.loads(data_str)
        except json.JSONDecodeError:
            log.warning("Unparseable upstream SSE data: %s", data_str[:200])
            continue
        if isinstance(event_data, dict):
            yield event_data


class _UpstreamRequest:
    """One upstream call with connection-error retries built in."""

    def __init__(self, request: web.Request, credentials: dict[str, Any], body: dict[str, Any]):
        self.request = request
        self.credentials = credentials
        self.body = body
        self.session: AsyncSession = request.app[UPSTREAM_SESSION_KEY]

    async def open(self, accept: str = "text/event-stream") -> Any:
        return await self.session.post(
            RESPONSES_ENDPOINT,
            json=self.body,
            headers=_build_upstream_headers(self.credentials, accept=accept),
            stream=True,
            timeout=_UPSTREAM_TIMEOUT_SECONDS,
        )


async def _credentials_for(
    request: web.Request,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the account credentials, or the reason they are unavailable."""
    try:
        return await ensure_credentials(), None
    except RuntimeError as exc:
        return None, str(exc)


async def _models_for(request: web.Request) -> list[dict[str, Any]]:
    """Warm the cached catalog so model aliases and effort levels resolve.

    A cold or unreachable catalog is not fatal: callers fall back to the
    built-in defaults, which still cover the Claude aliases.
    """
    catalog: ModelCatalog = request.app[MODEL_CATALOG_KEY]
    try:
        credentials = await ensure_credentials()
    except RuntimeError:
        credentials = None
    return await catalog.get(credentials)


# --- Endpoints ---


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_models(request: web.Request) -> web.Response:
    """List the models the active account can actually call."""
    models = await _models_for(request)
    # An empty catalog means the upstream lookup failed; the built-in list is
    # better than telling clients the account has no models.
    model_ids = visible_model_ids(models) or list(FALLBACK_MODELS)
    data = [
        {
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "openai",
            "context_window": _context_window(models, model_id),
        }
        for model_id in model_ids
    ]
    return web.json_response({"object": "list", "data": data})


def _context_window(models: list[dict[str, Any]], model_id: str) -> Any:
    for model in models:
        if model["slug"] == model_id:
            return model.get("context_window")
    return None


def _estimate_anthropic_input_tokens(payload: dict[str, Any]) -> dict[str, int]:
    """Estimate Anthropic input tokens with a rough chars/4 heuristic."""
    total_chars = 0

    system = payload.get("system")
    if isinstance(system, str):
        total_chars += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                total_chars += len(block.get("text", ""))

    for message in payload.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                total_chars += len(block.get("text", ""))
            elif block.get("type") == "tool_result":
                tool_content = block.get("content", "")
                if isinstance(tool_content, str):
                    total_chars += len(tool_content)
                else:
                    total_chars += len(json.dumps(tool_content, ensure_ascii=False))

    return {"input_tokens": max(1, total_chars // 4)}


async def handle_count_tokens(request: web.Request) -> web.Response:
    """Expose Anthropic-compatible input token estimation."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    return web.json_response(_estimate_anthropic_input_tokens(body), status=200)


def _aggregate_nonstream_chat_completion(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build one OpenAI Chat Completions JSON response from streaming chunks."""
    if not chunks:
        return None

    response_id: str | None = None
    model: str | None = None
    created: int | None = None
    usage: dict[str, Any] | None = None
    role: str | None = None
    finish_reason: str | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        if response_id is None and isinstance(chunk.get("id"), str):
            response_id = chunk["id"]
        if model is None and isinstance(chunk.get("model"), str):
            model = chunk["model"]
        if created is None and isinstance(chunk.get("created"), int):
            created = chunk["created"]
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            continue
        if isinstance(choice0.get("finish_reason"), str):
            finish_reason = choice0["finish_reason"]

        delta = choice0.get("delta")
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("role"), str) and delta["role"]:
            role = delta["role"]
        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
        if isinstance(delta.get("reasoning_content"), str):
            reasoning_parts.append(delta["reasoning_content"])
        _merge_tool_call_delta(tool_calls_by_index, delta.get("tool_calls"))

    tool_calls: list[dict[str, Any]] = []
    for index in sorted(tool_calls_by_index):
        state = tool_calls_by_index[index]
        tool_calls.append(
            {
                "id": state["id"] or f"call_{index}",
                "type": state["type"] or "function",
                "function": {
                    "name": state["function"]["name"] or "",
                    "arguments": state["function"]["arguments"] or "",
                },
            }
        )

    content = "".join(content_parts)
    message: dict[str, Any] = {"role": role or "assistant"}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["content"] = content or None
        message["tool_calls"] = tool_calls
        finish_reason = finish_reason or "tool_calls"
    else:
        message["content"] = content
        finish_reason = finish_reason or "stop"

    completion: dict[str, Any] = {
        "id": response_id or "chatcmpl-proxy",
        "object": "chat.completion",
        "created": created if created is not None else int(time.time()),
        "model": model or "",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage is not None:
        completion["usage"] = usage
    return completion


def _merge_tool_call_delta(
    tool_calls_by_index: dict[int, dict[str, Any]], delta_calls: Any
) -> None:
    if not isinstance(delta_calls, list):
        return

    for call in delta_calls:
        if not isinstance(call, dict):
            continue
        index = call.get("index")
        if not isinstance(index, int):
            continue

        state = tool_calls_by_index.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if isinstance(call.get("id"), str) and call["id"]:
            state["id"] = call["id"]
        if isinstance(call.get("type"), str) and call["type"]:
            state["type"] = call["type"]

        fn_delta = call.get("function")
        if not isinstance(fn_delta, dict):
            continue
        if isinstance(fn_delta.get("name"), str) and fn_delta["name"]:
            state["function"]["name"] = fn_delta["name"]
        if isinstance(fn_delta.get("arguments"), str) and fn_delta["arguments"]:
            state["function"]["arguments"] += fn_delta["arguments"]


async def _relay_chat_completions(
    request: web.Request,
    body: dict[str, Any],
    model: str,
    client_stream: bool,
) -> web.StreamResponse | web.Response:
    """Forward a translated Chat Completions request and stream the result back."""
    credentials, auth_error = await _credentials_for(request)
    if auth_error is not None:
        return web.json_response({"error": auth_error}, status=401)

    translator = ResponseStreamTranslator(model)
    response: web.StreamResponse | None = _make_stream_response() if client_stream else None
    upstream_request = _UpstreamRequest(request, credentials, body)

    log.info(
        "Upstream chat request: model=%s, tools=%d, input_items=%d",
        body.get("model"),
        len(body.get("tools", [])),
        len(body.get("input", [])),
    )

    for attempt in range(1, _MAX_UPSTREAM_ATTEMPTS + 1):
        sent_to_client = bool(response is not None and response.prepared)
        try:
            upstream = await upstream_request.open()
            if upstream.status_code != 200:
                error_text = await _read_upstream_text(upstream)
                message, error_type = _extract_upstream_error(error_text)
                log.error("Upstream error %d: %s", upstream.status_code, message[:300])
                return web.json_response(
                    _build_error_payload(message, error_type),
                    status=upstream.status_code,
                )

            translated: list[dict[str, Any]] = []
            retry = False
            async for event in _iter_upstream_events(upstream):
                event_type = event.get("type", "")
                if event_type in {"error", "response.failed"}:
                    retriable = (
                        (not sent_to_client and not translated)
                        and attempt < _MAX_UPSTREAM_ATTEMPTS
                        and _is_retriable_upstream_event(event_type, event)
                    )
                    (log.warning if retriable else log.error)(
                        "Upstream chat event %s: %s", event_type, json.dumps(event)[:500]
                    )
                    if retriable:
                        retry = True
                        break

                for sse_line in translator.translate_event(event_type, event):
                    if client_stream and response is not None:
                        await _ensure_stream_prepared(response, request)
                        await response.write(sse_line.encode())
                        sent_to_client = True
                    else:
                        payload = _parse_sse_data_line(sse_line)
                        if payload is None or payload == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(chunk, dict) and "error" in chunk:
                            return web.json_response({"error": chunk["error"]}, status=502)
                        translated.append(chunk)

                if _stream_finished(event_type):
                    return response if response is not None else web.json_response(
                        _aggregate_nonstream_chat_completion(translated) or {}
                    )

            if retry:
                log.warning(
                    "Retrying upstream chat request after transient event (%d/%d)",
                    attempt,
                    _MAX_UPSTREAM_ATTEMPTS,
                )
                continue

            if client_stream and response is not None:
                # Upstream closed without a terminal event; close the stream so
                # the client does not wait forever.
                await _ensure_stream_prepared(response, request)
                for sse_line in translator.finish():
                    await response.write(sse_line.encode())
                return response

            completion = _aggregate_nonstream_chat_completion(translated)
            if completion is not None:
                return web.json_response(completion)
            return web.json_response(
                _build_error_payload("Missing translated completion data", "upstream_error"),
                status=502,
            )

        except Exception as exc:
            if _should_retry_upstream_error(exc, attempt, sent_to_client):
                log.warning(
                    "Retrying upstream chat request after error (%d/%d): %s",
                    attempt,
                    _MAX_UPSTREAM_ATTEMPTS,
                    exc,
                )
                continue
            log.error("Connection error on chat completions: %s", exc)
            if client_stream and response is not None and response.prepared:
                await response.write(
                    f"data: {json.dumps(_build_error_payload(str(exc), 'connection_error'))}\n\n".encode()
                )
                await response.write(b"data: [DONE]\n\n")
                return response
            return web.json_response(
                _build_error_payload(str(exc), "connection_error"), status=502
            )

    return web.json_response(
        _build_error_payload("Failed to connect to upstream", "connection_error"), status=502
    )


def _stream_finished(event_type: str) -> bool:
    return event_type in {"response.completed", "response.incomplete", "response.failed", "error"}


def _parse_sse_data_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("data: "):
        return None
    return stripped[6:]


async def handle_chat_completions(request: web.Request) -> web.StreamResponse | web.Response:
    """OpenAI Chat Completions endpoint."""
    try:
        raw_body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    models = await _models_for(request)
    model = resolve_model_name(raw_body.get("model"), models)
    body = chat_to_responses(raw_body, models)
    body["model"] = model
    return await _relay_chat_completions(request, body, model, bool(raw_body.get("stream")))


async def handle_messages(request: web.Request) -> web.StreamResponse | web.Response:
    """Expose an Anthropic-compatible /v1/messages endpoint for Claude Code."""
    try:
        raw_body = await request.json()
    except json.JSONDecodeError:
        return web.json_response(
            _build_anthropic_error_payload("Invalid JSON", "invalid_request_error"), status=400
        )

    models = await _models_for(request)
    client_stream = bool(raw_body.get("stream", False))
    body = anthropic_messages_to_responses(raw_body, models)
    response_model = raw_body.get("model") if isinstance(raw_body.get("model"), str) else body["model"]

    credentials, auth_error = await _credentials_for(request)
    if auth_error is not None:
        return web.json_response(
            _build_anthropic_error_payload(auth_error, "authentication_error"), status=401
        )

    upstream_request = _UpstreamRequest(request, credentials, body)
    response: web.StreamResponse | None = _make_stream_response() if client_stream else None

    for attempt in range(1, _MAX_UPSTREAM_ATTEMPTS + 1):
        sent_to_client = bool(response is not None and response.prepared)
        translator = AnthropicStreamTranslator(response_model)
        try:
            upstream = await upstream_request.open()
            if upstream.status_code != 200:
                error_text = await _read_upstream_text(upstream)
                message, error_type = _extract_upstream_error(error_text)
                log.error("Upstream /messages error %d: %s", upstream.status_code, message[:300])
                return web.json_response(
                    _build_anthropic_error_payload(message, error_type),
                    status=upstream.status_code,
                )

            retry = False
            async for event in _iter_upstream_events(upstream):
                event_type = event.get("type", "")
                if event_type in {"error", "response.failed"}:
                    retriable = (
                        not sent_to_client
                        and attempt < _MAX_UPSTREAM_ATTEMPTS
                        and _is_retriable_upstream_event(event_type, event)
                    )
                    (log.warning if retriable else log.error)(
                        "Upstream /messages event %s: %s", event_type, json.dumps(event)[:500]
                    )
                    if retriable:
                        retry = True
                        break

                for sse_line in translator.translate_event(event_type, event):
                    if client_stream and response is not None:
                        await _ensure_stream_prepared(response, request)
                        await response.write(sse_line.encode())
                        sent_to_client = True

                if translator.closed:
                    break

            if retry:
                log.warning(
                    "Retrying upstream /messages request after transient event (%d/%d)",
                    attempt,
                    _MAX_UPSTREAM_ATTEMPTS,
                )
                continue

            if client_stream and response is not None:
                await _ensure_stream_prepared(response, request)
                if not translator.closed:
                    # Upstream ended without a terminal event; close the message
                    # so Claude Code does not hang waiting for message_stop.
                    for sse_line in translator.finish():
                        await response.write(sse_line.encode())
                return response

            if translator.error is not None:
                error_type = str(translator.error.get("type") or "upstream_error")
                message = str(translator.error.get("message") or "Upstream request failed")
                return web.json_response(
                    _build_anthropic_error_payload(message, error_type), status=502
                )
            translator.finish()
            return web.json_response(translator.build_response())

        except Exception as exc:
            if _should_retry_upstream_error(exc, attempt, sent_to_client):
                log.warning(
                    "Retrying upstream /messages request after error (%d/%d): %s",
                    attempt,
                    _MAX_UPSTREAM_ATTEMPTS,
                    exc,
                )
                continue
            log.error("Connection error on /messages: %s", exc)
            if client_stream and response is not None and response.prepared:
                payload = {"type": "error", "error": {"type": "connection_error", "message": str(exc)}}
                await response.write(
                    f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
                )
                return response
            return web.json_response(
                _build_anthropic_error_payload(str(exc), "connection_error"), status=502
            )

    return web.json_response(
        _build_anthropic_error_payload("Failed to connect to upstream", "connection_error"),
        status=502,
    )


async def handle_responses(request: web.Request) -> web.StreamResponse | web.Response:
    """Forward OpenAI Responses API requests directly to the ChatGPT backend."""
    try:
        raw_body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    models = await _models_for(request)
    body, client_stream = _normalize_responses_body(raw_body, models)

    credentials, auth_error = await _credentials_for(request)
    if auth_error is not None:
        return web.json_response({"error": auth_error}, status=401)

    log.info(
        "Upstream /responses request: model=%s, stream=%s, input_items=%d",
        body.get("model"),
        client_stream,
        len(body.get("input", [])) if isinstance(body.get("input"), list) else 0,
    )

    upstream_request = _UpstreamRequest(request, credentials, body)
    response: web.StreamResponse | None = _make_stream_response() if client_stream else None

    for attempt in range(1, _MAX_UPSTREAM_ATTEMPTS + 1):
        sent_to_client = bool(response is not None and response.prepared)
        try:
            upstream = await upstream_request.open()
            if upstream.status_code != 200:
                error_text = await _read_upstream_text(upstream)
                try:
                    return web.json_response(json.loads(error_text), status=upstream.status_code)
                except json.JSONDecodeError:
                    return web.Response(text=error_text, status=upstream.status_code)

            if client_stream and response is not None:
                async for chunk in upstream.aiter_content():
                    await _ensure_stream_prepared(response, request)
                    await response.write(chunk if isinstance(chunk, bytes) else chunk.encode())
                    sent_to_client = True
                await _ensure_stream_prepared(response, request)
                return response

            completed: dict[str, Any] | None = None
            async for event in _iter_upstream_events(upstream):
                event_type = event.get("type", "")
                if event_type in {"response.completed", "response.incomplete"}:
                    response_obj = event.get("response")
                    if isinstance(response_obj, dict):
                        completed = response_obj
                elif event_type in {"error", "response.failed"}:
                    return web.json_response(event, status=502)

            if completed is not None:
                return web.json_response(completed, status=200)
            return web.json_response(
                _build_error_payload("Missing response.completed event", "upstream_error"),
                status=502,
            )

        except Exception as exc:
            if _should_retry_upstream_error(exc, attempt, sent_to_client):
                log.warning(
                    "Retrying upstream /responses request after error (%d/%d): %s",
                    attempt,
                    _MAX_UPSTREAM_ATTEMPTS,
                    exc,
                )
                continue
            log.error("Connection error on /responses: %s", exc)
            if client_stream and response is not None and response.prepared:
                payload = {"type": "error", "error": {"type": "connection_error", "message": str(exc)}}
                await response.write(f"data: {json.dumps(payload)}\n\n".encode())
                await response.write(b"data: [DONE]\n\n")
                return response
            return web.json_response(
                _build_error_payload(str(exc), "connection_error"), status=502
            )

    return web.json_response(
        _build_error_payload("Failed to connect to upstream", "connection_error"), status=502
    )


async def _create_upstream_session(app: web.Application) -> None:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    app[UPSTREAM_SESSION_KEY] = AsyncSession(
        impersonate="chrome",
        proxies={"https": proxy} if proxy else None,
        http_version="v1",
        curl_options={CurlOpt.TCP_KEEPALIVE: 1},
    )


async def _close_upstream_session(app: web.Application) -> None:
    await app[UPSTREAM_SESSION_KEY].close()


def create_app() -> web.Application:
    """Create the aiohttp application."""
    middlewares = [api_key_middleware]
    if _configured_api_key():
        log.info("API key authentication enabled")
    else:
        log.warning("No CODEX_PROXY_API_KEY set - running without authentication")

    app = web.Application(middlewares=middlewares)
    app[MODEL_CATALOG_KEY] = ModelCatalog()
    app.on_startup.append(_create_upstream_session)
    app.on_cleanup.append(_close_upstream_session)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/responses", handle_responses)
    app.router.add_post("/responses", handle_responses)
    app.router.add_post("/v1/messages", handle_messages)
    app.router.add_post("/messages", handle_messages)
    app.router.add_post("/v1/messages/count_tokens", handle_count_tokens)
    app.router.add_post("/messages/count_tokens", handle_count_tokens)
    return app


def run_server(host: str, port: int) -> None:
    """Start the proxy server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = create_app()
    log.info("Starting codex-proxy on %s:%d", host, port)
    log.info(
        "Endpoints: POST /v1/chat/completions, POST /v1/responses, POST /v1/messages, "
        "GET /v1/models, GET /health"
    )
    web.run_app(app, host=host, port=port, print=None)
