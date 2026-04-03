#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/setup_scheduler.py — Windows Task Scheduler Setup
==========================================================

Creates Windows scheduled tasks for:
1. Alpha pipeline — runs every 4 hours
2. Auto-improvement — runs every 6 hours
3. Data refresh — runs every 2 hours
4. Daily report — runs at 16:30 (after market close)

Usage:
    python scripts/setup_scheduler.py --create    # Create all tasks
    python scripts/setup_scheduler.py --delete    # Remove all tasks
    python scripts/setup_scheduler.py --status    # Check status

Fixes #13: "No scheduled cron integration"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


TASKS = [
    {
        "name": "PairsTrading_Daemon",
        "script": "scripts\\run_scheduler_daemon.py",
        "args": "",
        "schedule": "/SC ONLOGON",  # Start on user login (resilience)
        "description": "Pairs Trading: Main scheduler daemon (manages all jobs)",
    },
    {
        "name": "PairsTrading_AlphaPipeline",
        "script": "scripts\\run_full_alpha.py",
        "args": "--universe all --trials 20 --min-sharpe 0.3",
        "schedule": "/SC HOURLY /MO 4",  # Every 4 hours
        "description": "Pairs Trading: Alpha pipeline (discover, optimize, backtest)",
    },
    {
        "name": "PairsTrading_AutoImprove",
        "script": "scripts\\run_auto_improve.py",
        "args": "",
        "schedule": "/SC HOURLY /MO 6",  # Every 6 hours
        "description": "Pairs Trading: GPT auto-improvement cycle",
    },
    {
        "name": "PairsTrading_DataRefresh",
        "script": "scripts\\run_auto_improve.py",
        "args": "--cycle data",
        "schedule": "/SC HOURLY /MO 2",  # Every 2 hours
        "description": "Pairs Trading: Refresh price data from Yahoo/FMP",
    },
    {
        "name": "PairsTrading_DailyReport",
        "script": "scripts\\run_auto_improve.py",
        "args": "--cycle analysis",
        "schedule": "/SC DAILY /ST 16:30",  # 4:30 PM daily
        "description": "Pairs Trading: Daily GPT analysis report",
    },
]


def create_tasks():
    """Create all Windows scheduled tasks."""
    print("Creating scheduled tasks...")
    for task in TASKS:
        script_path = PROJECT_ROOT / task["script"]
        task_args = task["args"]
        cmd = (
            f'schtasks /CREATE /TN "{task["name"]}" '
            f'/TR "\\"{PYTHON}\\" \\"{script_path}\\" {task_args}" '
            f'{task["schedule"]} '
            f'/F'  # Force overwrite
        )
        print(f"  Creating: {task['name']}")
        print(f"    Schedule: {task['schedule']}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    ✅ Created")
        else:
            print(f"    ❌ Failed: {result.stderr.strip()}")
            print(f"    (May need admin privileges)")


def delete_tasks():
    """Delete all scheduled tasks."""
    print("Deleting scheduled tasks...")
    for task in TASKS:
        cmd = f'schtasks /DELETE /TN "{task["name"]}" /F'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Deleted: {task['name']}")
        else:
            print(f"  ⚠️ Not found: {task['name']}")


def check_status():
    """Check status of all scheduled tasks."""
    print("Task status:")
    for task in TASKS:
        cmd = f'schtasks /QUERY /TN "{task["name"]}" /FO CSV /NH'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            status = parts[2].strip('"') if len(parts) > 2 else "?"
            next_run = parts[1].strip('"') if len(parts) > 1 else "?"
            print(f"  {task['name']}: Status={status}, Next={next_run}")
        else:
            print(f"  {task['name']}: Not scheduled")


def main():
    parser = argparse.ArgumentParser(description="Windows Task Scheduler setup")
    parser.add_argument("--create", action="store_true", help="Create scheduled tasks")
    parser.add_argument("--delete", action="store_true", help="Delete scheduled tasks")
    parser.add_argument("--status", action="store_true", help="Check task status")
    args = parser.parse_args()

    if args.create:
        create_tasks()
    elif args.delete:
        delete_tasks()
    elif args.status:
        check_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
