"""aiohttp proxy server exposing OpenAI-compatible endpoints."""

import json
import logging
import os
from typing import Any

from aiohttp import web
from curl_cffi.requests import AsyncSession

from codex_proxy.auth import ensure_credentials, extract_account_id
from codex_proxy.config import CODEX_MODELS, RESPONSES_ENDPOINT
from codex_proxy.translator import (
    ResponseStreamTranslator,
    chat_to_responses,
)

log = logging.getLogger(__name__)
_UNSUPPORTED_RESPONSES_PARAMS = {
    "max_output_tokens",
    "max_tokens",
    "max_completion_tokens",
}


def _normalize_tool_strict(value: Any) -> bool:
    """Normalize tool strictness to a boolean value."""
    return value if isinstance(value, bool) else False


def _normalize_input_content(content: Any) -> Any:
    """Normalize message content parts for ChatGPT codex upstream."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return content

    normalized: list[Any] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            normalized.append({"type": "input_text", "text": part.get("text", "")})
        else:
            normalized.append(part)
    return normalized


def _normalize_responses_input(input_value: Any) -> Any:
    """Normalize /responses input items from OpenAI-style payloads."""
    if isinstance(input_value, str):
        return [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": input_value}],
            }
        ]
    if not isinstance(input_value, list):
        return input_value

    normalized: list[Any] = []
    converted = 0
    for item in input_value:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        if item.get("type") == "message":
            converted += 1
            new_item = dict(item)
            new_item.pop("type", None)
            if "content" in new_item:
                new_item["content"] = _normalize_input_content(new_item["content"])
            normalized.append(new_item)
            continue

        new_item = dict(item)
        if "content" in new_item:
            new_item["content"] = _normalize_input_content(new_item["content"])
        normalized.append(new_item)

    if converted:
        log.info("Normalized %d OpenAI-style input message items on /responses", converted)
    return normalized


def _normalize_responses_tools(tools: Any) -> Any:
    """Accept both Chat Completions and Responses tools schema on /responses."""
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
        return {
            "type": "function",
            "name": fn.get("name", ""),
        }

    return tool_choice


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_models(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "object": "list",
            "data": CODEX_MODELS,
        }
    )


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


async def _read_upstream_text(upstream: Any) -> str:
    """Read full upstream response body as text."""
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


def _normalize_responses_body(raw_body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Normalize OpenAI-style /responses request for ChatGPT codex upstream."""
    body = dict(raw_body)
    client_stream = bool(body.get("stream"))

    dropped = _UNSUPPORTED_RESPONSES_PARAMS & body.keys()
    for key in dropped:
        body.pop(key, None)
    if dropped:
        log.warning("Dropping unsupported /responses parameters: %s", ", ".join(sorted(dropped)))

    # ChatGPT Codex responses endpoint requires instructions.
    # Add a sensible default for OpenAI-compatible clients that omit it.
    if not body.get("instructions"):
        body["instructions"] = "You are a helpful assistant."

    if body.get("store") is not False:
        body["store"] = False

    # OpenAI Responses API allows input as a plain string.
    # Normalize to ChatGPT Codex format that expects a list of input items.
    if "input" in body:
        body["input"] = _normalize_responses_input(body.get("input"))

    if "tools" in body:
        body["tools"] = _normalize_responses_tools(body.get("tools"))
    if "tool_choice" in body:
        body["tool_choice"] = _normalize_responses_tool_choice(body.get("tool_choice"))
    if body.get("tools") and "parallel_tool_calls" not in body:
        # LiteLLM <=1.81.x responses streaming chunks use a fixed tool index (0),
        # which can merge multiple calls in downstream aggregators.
        # Default to sequential tool calls unless the client explicitly opts in.
        body["parallel_tool_calls"] = False

    # ChatGPT Codex responses endpoint currently enforces stream=true.
    # For client stream=false, we aggregate upstream SSE into a final JSON response.
    body["stream"] = True
    return body, client_stream


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    """Main proxy endpoint: translate and forward to ChatGPT backend."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Get credentials (auto-refresh if expired)
    try:
        credentials = await ensure_credentials()
    except RuntimeError as e:
        return web.json_response({"error": str(e)}, status=401)

    # Translate request
    model = body.get("model", "gpt-5.1")
    responses_body = chat_to_responses(body)

    # Build headers for ChatGPT backend
    headers = _build_upstream_headers(credentials, accept="text/event-stream")

    # Forward to ChatGPT backend
    translator = ResponseStreamTranslator(model)
    response = web.StreamResponse()
    response.content_type = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    await response.prepare(request)

    tools_count = len(responses_body.get("tools", []))
    log.info(
        "Upstream request: model=%s, tools=%d, tool_choice=%s, input_items=%d",
        responses_body.get("model"),
        tools_count,
        responses_body.get("tool_choice", "N/A"),
        len(responses_body.get("input", [])),
    )
    log.debug("Upstream request body: %s", json.dumps(responses_body)[:5000])

    try:
        session: AsyncSession = request.app["upstream_session"]
        upstream = await session.post(
            RESPONSES_ENDPOINT,
            json=responses_body,
            headers=headers,
            stream=True,
            timeout=120,
        )

        if upstream.status_code != 200:
            error_text = await _read_upstream_text(upstream)
            log.error("Upstream error %d: %s", upstream.status_code, error_text[:1000])
            error_chunk = json.dumps(
                {
                    "error": {
                        "message": f"ChatGPT API error: {upstream.status_code}",
                        "type": "upstream_error",
                        "detail": error_text[:500],
                    }
                }
            )
            await response.write(f"data: {error_chunk}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        # Process SSE stream from upstream
        async for line_bytes in upstream.aiter_lines():
            line = line_bytes.decode() if isinstance(line_bytes, bytes) else line_bytes
            line = line.strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    await response.write(b"data: [DONE]\n\n")
                    return response
                try:
                    event_data = json.loads(data_str)
                    event_type = event_data.get("type", "")
                    sse_lines = translator.translate_event(event_type, event_data)
                    for sse_line in sse_lines:
                        await response.write(sse_line.encode())
                except json.JSONDecodeError:
                    log.warning("Unparseable SSE data: %s", data_str[:200])

    except Exception as e:
        log.error("Connection error: %s", e)
        error_chunk = json.dumps({"error": {"message": str(e), "type": "connection_error"}})
        await response.write(f"data: {error_chunk}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")

    return response


async def handle_responses(request: web.Request) -> web.StreamResponse | web.Response:
    """Forward OpenAI Responses API requests directly to ChatGPT backend."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    body, client_stream = _normalize_responses_body(body)

    try:
        credentials = await ensure_credentials()
    except RuntimeError as e:
        return web.json_response({"error": str(e)}, status=401)

    headers = _build_upstream_headers(
        credentials,
        accept="text/event-stream",
    )

    log.info(
        "Upstream /responses request: model=%s, stream=%s, input_items=%d",
        body.get("model"),
        client_stream,
        len(body.get("input", [])) if isinstance(body.get("input"), list) else 0,
    )
    log.debug("Upstream /responses request body: %s", json.dumps(body)[:5000])

    try:
        session: AsyncSession = request.app["upstream_session"]
        upstream = await session.post(
            RESPONSES_ENDPOINT,
            json=body,
            headers=headers,
            stream=True,
            timeout=120,
        )

        if upstream.status_code != 200:
            error_text = await _read_upstream_text(upstream)
            log.error("Upstream /responses error %d: %s", upstream.status_code, error_text[:1000])
            try:
                return web.json_response(json.loads(error_text), status=upstream.status_code)
            except json.JSONDecodeError:
                return web.Response(text=error_text, status=upstream.status_code)

        if client_stream:
            response = web.StreamResponse()
            response.content_type = "text/event-stream"
            response.headers["Cache-Control"] = "no-cache"
            response.headers["Connection"] = "keep-alive"
            response.headers["X-Accel-Buffering"] = "no"
            await response.prepare(request)

            async for chunk in upstream.aiter_content():
                await response.write(chunk if isinstance(chunk, bytes) else chunk.encode())
            return response

        completed_response: dict[str, Any] | None = None
        async for line_bytes in upstream.aiter_lines():
            line = line_bytes.decode() if isinstance(line_bytes, bytes) else line_bytes
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                event_data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = event_data.get("type", "")
            if event_type == "response.completed":
                response_obj = event_data.get("response")
                if isinstance(response_obj, dict):
                    completed_response = response_obj
            elif event_type in {"error", "response.failed"}:
                return web.json_response(event_data, status=502)

        if completed_response is not None:
            return web.json_response(completed_response, status=200)

        return web.json_response(
            {"error": {"message": "Missing response.completed event", "type": "upstream_error"}},
            status=502,
        )

    except Exception as e:
        log.error("Connection error on /responses: %s", e)
        return web.json_response(
            {"error": {"message": str(e), "type": "connection_error"}},
            status=502,
        )


async def _create_upstream_session(app: web.Application) -> None:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    app["upstream_session"] = AsyncSession(
        impersonate="chrome",
        proxies={"https": proxy} if proxy else None,
    )


async def _close_upstream_session(app: web.Application) -> None:
    session: AsyncSession = app["upstream_session"]
    await session.close()


def create_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app.on_startup.append(_create_upstream_session)
    app.on_cleanup.append(_close_upstream_session)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/responses", handle_responses)
    app.router.add_post("/responses", handle_responses)
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
        "Endpoints: POST /v1/chat/completions, POST /v1/responses, GET /v1/models, GET /health"
    )
    web.run_app(app, host=host, port=port, print=None)
