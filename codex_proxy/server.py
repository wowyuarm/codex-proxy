"""aiohttp proxy server exposing OpenAI-compatible endpoints."""

import json
import logging
import os

from aiohttp import web
from curl_cffi.requests import AsyncSession

from codex_proxy.auth import ensure_credentials, extract_account_id
from codex_proxy.config import CODEX_MODELS, RESPONSES_ENDPOINT
from codex_proxy.translator import (
    ResponseStreamTranslator,
    chat_to_responses,
)

log = logging.getLogger(__name__)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_models(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "object": "list",
            "data": CODEX_MODELS,
        }
    )


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
    access_token = credentials["access_token"]
    account_id = credentials.get("account_id") or extract_account_id(access_token)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    # Forward to ChatGPT backend
    translator = ResponseStreamTranslator(model)
    response = web.StreamResponse()
    response.content_type = "text/event-stream"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    await response.prepare(request)

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    try:
        async with AsyncSession(impersonate="chrome") as session:
            upstream = await session.post(
                RESPONSES_ENDPOINT,
                json=responses_body,
                headers=headers,
                proxies={"https": proxy} if proxy else None,
                stream=True,
                timeout=120,
            )

            if upstream.status_code != 200:
                error_text = upstream.text
                log.error("Upstream error %d: %s", upstream.status_code, error_text[:500])
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


def create_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/chat/completions", handle_chat_completions)
    return app


def run_server(host: str, port: int) -> None:
    """Start the proxy server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = create_app()
    log.info("Starting codex-proxy on %s:%d", host, port)
    log.info("Endpoints: POST /v1/chat/completions, GET /v1/models, GET /health")
    web.run_app(app, host=host, port=port, print=None)
