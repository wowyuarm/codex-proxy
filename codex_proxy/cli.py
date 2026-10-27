"""CLI entry point for codex-proxy."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import click

from codex_proxy.accounts import (
    account_credentials_path,
    activate_account,
    find_account_record,
    list_accounts,
    load_registry,
    remove_account,
    update_account_usage,
    upsert_account,
)
from codex_proxy.config import DEFAULT_HOST, DEFAULT_PORT
from codex_proxy.usage import (
    UsageLookupError,
    fetch_usage,
    format_credits,
    format_window,
    remaining_percent,
)


@click.group()
def main() -> None:
    """Codex Proxy — Use ChatGPT Codex models via an OpenAI-compatible API."""


@main.command()
def login() -> None:
    """Authenticate with ChatGPT via OAuth PKCE."""
    from codex_proxy.auth import login as do_login

    try:
        credentials = asyncio.run(do_login())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    record = upsert_account(credentials, activate=True)
    click.echo(f"Saved account: {record['label']}")


@main.command()
@click.option("-h", "--host", default=DEFAULT_HOST, help="Bind address")
@click.option("-p", "--port", default=DEFAULT_PORT, type=int, help="Listen port")
def serve(host: str, port: int) -> None:
    """Start the proxy server."""
    from codex_proxy.server import run_server

    run_server(host, port)


async def _fetch_live_models() -> list[dict[str, Any]]:
    from codex_proxy.auth import ensure_credentials
    from codex_proxy.models import ModelCatalog

    credentials = await ensure_credentials()
    return await ModelCatalog().refresh(credentials)


@main.command()
def models() -> None:
    """List the models the active account can use."""
    try:
        catalog = asyncio.run(_fetch_live_models())
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if not catalog:
        from codex_proxy.config import FALLBACK_MODELS

        click.echo("Could not reach the upstream model list. Built-in fallback:")
        for model_id in FALLBACK_MODELS:
            click.echo(f"  {model_id}")
        return

    for entry in catalog:
        if not entry["visible"]:
            continue
        efforts = ", ".join(entry.get("reasoning_efforts") or []) or "-"
        click.echo(f"{entry['slug']}")
        click.echo(f"  name:    {entry['display_name']}")
        click.echo(f"  effort:  {efforts}")
        if entry.get("context_window"):
            click.echo(f"  context: {entry['context_window']}")
        if entry.get("description"):
            click.echo(f"  about:   {entry['description']}")


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
    click.echo("Usage:      use 'codex-proxy accounts' to see usage per account")


def _display_name(record: dict[str, Any]) -> str:
    label = record.get("label")
    email = record.get("email")
    account_id = record.get("account_id")
    if label and email and label != email:
        return f"{label} ({email})"
    return label or email or account_id or "unknown"


def _account_plan(record: dict[str, Any], usage: dict[str, Any] | None) -> str:
    if usage and isinstance(usage.get("plan_type"), str):
        return usage["plan_type"]
    cached = record.get("last_usage")
    if isinstance(cached, dict) and isinstance(cached.get("plan_type"), str):
        return cached["plan_type"]
    return "-"


def _format_compact_window(window: dict[str, Any] | None) -> str:
    remaining = remaining_percent(window)
    if remaining is None:
        return "-"
    reset_at = window.get("reset_at")
    if isinstance(reset_at, int):
        reset_str = datetime.fromtimestamp(reset_at, tz=timezone.utc).strftime("%m-%d %H:%M")
        return f"{remaining}% ({reset_str} UTC)"
    return f"{remaining}%"


async def _refresh_account_usage(
    record: dict[str, Any], active_account_id: str | None
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    from codex_proxy.auth import ensure_credentials

    account_id = record["account_id"]
    credentials = await ensure_credentials(account_credentials_path(account_id))
    refreshed = upsert_account(
        credentials,
        label=record.get("label"),
        activate=account_id == active_account_id,
    )

    try:
        usage = await fetch_usage(credentials)
    except UsageLookupError as exc:
        usage = refreshed.get("last_usage")
        return refreshed, usage, str(exc)

    update_account_usage(account_id, usage)
    latest_registry = load_registry()
    _, latest_record = find_account_record(latest_registry, account_id)
    return latest_record or refreshed, usage, None


async def _refresh_usage_rows(
    records: list[dict[str, Any]], active_account_id: str | None
) -> list[tuple[dict[str, Any], dict[str, Any] | None, str | None]]:
    return await asyncio.gather(
        *(_refresh_account_usage(record, active_account_id) for record in records)
    )


def _print_accounts_table(
    registry: dict[str, Any],
    rows: list[tuple[dict[str, Any], dict[str, Any] | None, str | None]],
) -> None:
    active_account_id = registry.get("active_account_id")
    rendered_rows = []
    for record, usage, error in rows:
        rendered_rows.append(
            {
                "marker": "*" if record.get("account_id") == active_account_id else " ",
                "account_id": record.get("account_id") or "-",
                "account": _display_name(record),
                "plan": _account_plan(record, usage),
                "five_hour": _format_compact_window((usage or {}).get("primary")),
                "weekly": _format_compact_window((usage or {}).get("secondary")),
                "error": error,
            }
        )

    id_width = max(len("ACCOUNT ID"), *(len(row["account_id"]) for row in rendered_rows))
    account_width = max(len("ACCOUNT"), *(len(row["account"]) for row in rendered_rows))
    plan_width = max(len("PLAN"), *(len(row["plan"]) for row in rendered_rows))
    five_width = max(len("5H"), *(len(row["five_hour"]) for row in rendered_rows))
    weekly_width = max(len("WEEKLY"), *(len(row["weekly"]) for row in rendered_rows))

    click.echo(
        f"  {'ACCOUNT ID'.ljust(id_width)}  {'ACCOUNT'.ljust(account_width)}  "
        f"{'PLAN'.ljust(plan_width)}  "
        f"{'5H'.ljust(five_width)}  {'WEEKLY'.ljust(weekly_width)}"
    )
    for row in rendered_rows:
        click.echo(
            f"{row['marker']} {row['account_id'].ljust(id_width)}  "
            f"{row['account'].ljust(account_width)}  {row['plan'].ljust(plan_width)}  "
            f"{row['five_hour'].ljust(five_width)}  {row['weekly'].ljust(weekly_width)}"
        )
        if row["error"]:
            click.echo(f"    usage unavailable: {row['error']}")


@main.command(name="accounts")
@click.option("--remove", "remove_id", help="Delete one saved account by account id")
def accounts_command(remove_id: str | None) -> None:
    """List stored accounts and show usage."""
    registry, accounts = list_accounts()
    if not accounts:
        click.echo("No saved accounts. Run 'codex-proxy login' first.")
        return

    if remove_id:
        removed, new_active = remove_account(remove_id)
        click.echo(f"Removed account: {_display_name(removed)}")
        if new_active:
            click.echo(f"Now active: {_display_name(new_active)}")
        registry, accounts = list_accounts()
        if not accounts:
            return

    rows = asyncio.run(_refresh_usage_rows(accounts, registry.get("active_account_id")))
    registry = load_registry()
    _print_accounts_table(registry, rows)
    click.echo("")
    click.echo("'*' marks the active account. Use 'switch <account_id>' to change it.")


@main.command()
@click.argument("query", required=False)
def switch(query: str | None) -> None:
    """Switch the active account."""
    registry, accounts = list_accounts()
    if not accounts:
        click.echo("No saved accounts. Run 'codex-proxy login' first.")
        return
    if not query:
        click.echo("Saved accounts:")
        rows = asyncio.run(_refresh_usage_rows(accounts, registry.get("active_account_id")))
        registry = load_registry()
        _print_accounts_table(registry, rows)
        return

    _, selected_record = find_account_record(registry, query)
    if selected_record is None:
        click.echo(f"No account id matches '{query}'.")
        rows = asyncio.run(_refresh_usage_rows(accounts, registry.get("active_account_id")))
        _print_accounts_table(registry, rows)
        raise click.ClickException("Use one of the account ids shown above.")

    selected = activate_account(selected_record["account_id"])
    refreshed, usage, error = asyncio.run(
        _refresh_account_usage(selected, selected["account_id"])
    )
    click.echo(f"Active account: {_display_name(refreshed)}")
    click.echo(f"Account ID: {refreshed['account_id']}")
    click.echo(f"Plan: {_account_plan(refreshed, usage)}")
    click.echo(format_window("5h", (usage or {}).get("primary")))
    click.echo(format_window("Weekly", (usage or {}).get("secondary")))
    click.echo(format_credits((usage or {}).get("credits")))
    if error:
        click.echo(f"Usage lookup warning: {error}")
