# -*- coding: utf-8 -*-
"""
core/alerts.py — Signal & Risk Alerts (Telegram + Console)
===========================================================

Sends alerts for:
- New entry/exit signals
- Risk warnings (drawdown, vol spike)
- System health issues
- Daily performance summary

Fixes #12: "No Telegram/Slack alerts for signals"

Setup:
    Add to .env:
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_telegram_config() -> tuple[Optional[str], Optional[str]]:
    """Load Telegram config from .env."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                elif line.startswith("TELEGRAM_CHAT_ID="):
                    chat_id = line.split("=", 1)[1].strip()

    return token, chat_id


def send_telegram(message: str, parse_mode: str = "Markdown") -> bool:
    """Send a Telegram message. Returns True if sent successfully."""
    token, chat_id = _load_telegram_config()
    if not token or not chat_id:
        logger.debug("Telegram not configured — alert logged to console only")
        return False

    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }, timeout=10)
        if resp.status_code == 200:
            logger.info("Telegram alert sent")
            return True
        else:
            logger.warning("Telegram send failed: %s", resp.text)
            return False
    except Exception as exc:
        logger.warning("Telegram error: %s", exc)
        return False


def alert_signal(pair: str, action: str, z_score: float, **kwargs) -> None:
    """Alert on new trading signal."""
    emoji = "🟢" if action == "ENTRY_LONG" else "🔴" if action == "ENTRY_SHORT" else "⚪"
    msg = (
        f"{emoji} *Signal: {action}*\n"
        f"Pair: `{pair}`\n"
        f"Z-Score: `{z_score:.2f}`\n"
        f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )
    for k, v in kwargs.items():
        msg += f"\n{k}: `{v}`"

    print(msg.replace("*", "").replace("`", ""))  # Console
    send_telegram(msg)


def alert_risk(warning_type: str, details: str, severity: str = "WARNING") -> None:
    """Alert on risk event."""
    emoji = {"WARNING": "⚠️", "CRITICAL": "🚨", "INFO": "ℹ️"}.get(severity, "⚠️")
    msg = (
        f"{emoji} *Risk Alert: {warning_type}*\n"
        f"Severity: `{severity}`\n"
        f"{details}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
    )
    print(msg.replace("*", "").replace("`", ""))
    send_telegram(msg)


def alert_daily_summary(
    sharpe: float,
    daily_pnl: float,
    n_active: int,
    n_trades: int,
    drawdown: float,
) -> None:
    """Send daily performance summary."""
    pnl_emoji = "📈" if daily_pnl > 0 else "📉"
    msg = (
        f"📊 *Daily Summary*\n"
        f"{pnl_emoji} PnL: `${daily_pnl:+,.0f}`\n"
        f"Sharpe: `{sharpe:.2f}`\n"
        f"Active Pairs: `{n_active}`\n"
        f"Trades Today: `{n_trades}`\n"
        f"Drawdown: `{drawdown:.1f}%`\n"
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    )
    print(msg.replace("*", "").replace("`", ""))
    send_telegram(msg)


def alert_system(component: str, status: str, details: str = "") -> None:
    """Alert on system health event."""
    emoji = "✅" if status == "OK" else "🔴"
    msg = f"{emoji} *System: {component}*\nStatus: `{status}`"
    if details:
        msg += f"\n{details}"
    print(msg.replace("*", "").replace("`", ""))
    send_telegram(msg)
