"""CLI entry point for codex-proxy."""

import asyncio
import time
from datetime import datetime, timezone

import click

from codex_proxy.config import DEFAULT_HOST, DEFAULT_PORT


@click.group()
def main() -> None:
    """Codex Proxy — Use ChatGPT Codex models via an OpenAI-compatible API."""


@main.command()
def login() -> None:
    """Authenticate with ChatGPT via OAuth PKCE."""
    from codex_proxy.auth import login as do_login

    asyncio.run(do_login())


@main.command()
@click.option("-h", "--host", default=DEFAULT_HOST, help="Bind address")
@click.option("-p", "--port", default=DEFAULT_PORT, type=int, help="Listen port")
def serve(host: str, port: int) -> None:
    """Start the proxy server."""
    from codex_proxy.server import run_server

    run_server(host, port)


@main.command()
def status() -> None:
    """Show current authentication status."""
    from codex_proxy.auth import load_credentials

    credentials = load_credentials()
    if credentials is None:
        click.echo("Not logged in. Run 'codex-proxy login' to authenticate.")
        return

    account_id = credentials.get("account_id", "unknown")
    expires_at = credentials.get("expires_at", 0)
    remaining = expires_at - time.time()
    expires_str = datetime.fromtimestamp(expires_at, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    click.echo(f"Account ID: {account_id}")
    click.echo(f"Expires at: {expires_str}")
    if remaining > 0:
        minutes = int(remaining // 60)
        click.echo(f"Status:     valid ({minutes} min remaining)")
    else:
        has_refresh = bool(credentials.get("refresh_token"))
        if has_refresh:
            click.echo("Status:     expired (will auto-refresh on next request)")
        else:
            click.echo("Status:     expired (no refresh token, run 'codex-proxy login')")
