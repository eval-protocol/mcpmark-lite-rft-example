#!/usr/bin/env python3
"""Poll a Fireworks RFT job until terminal state."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

TERMINAL_STATES = {
    "JOB_STATE_COMPLETED",
    "JOB_STATE_FAILED",
    "JOB_STATE_EARLY_STOPPED",
    "JOB_STATE_CANCELLED",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_json(url: str, api_key: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _format_line(payload: dict[str, Any]) -> str:
    state = payload.get("state", "")
    status = payload.get("status", {}) or {}
    code = status.get("code", "")
    message = (status.get("message", "") or "").replace("\n", " ").strip()
    progress = payload.get("jobProgress", {}) or {}
    percent = progress.get("percent", 0)
    epoch = progress.get("epoch", 0)
    return f"{_now_utc()} state={state} code={code} percent={percent} epoch={epoch} message={message}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a Fireworks reinforcement fine-tuning job.")
    parser.add_argument("--account", default="pyroworks", help="Fireworks account id.")
    parser.add_argument("--job-id", required=True, help="RFT job id (short id or full resource name).")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=20,
        help="Polling interval seconds.",
    )
    parser.add_argument(
        "--max-minutes",
        type=int,
        default=180,
        help="Maximum monitoring duration in minutes before timeout.",
    )
    args = parser.parse_args()

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY is not set.", file=sys.stderr)
        return 1

    job_id = args.job_id
    if "/" in job_id:
        job_resource = job_id
    else:
        job_resource = f"accounts/{args.account}/reinforcementFineTuningJobs/{job_id}"

    url = f"https://api.fireworks.ai/v1/{job_resource}"
    deadline = time.time() + (args.max_minutes * 60)

    while True:
        try:
            payload = _request_json(url, api_key)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"{_now_utc()} http_error={exc.code} detail={detail}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"{_now_utc()} request_error={exc}", file=sys.stderr)
            return 1

        print(_format_line(payload), flush=True)
        state = payload.get("state", "")
        if state in TERMINAL_STATES:
            status = payload.get("status", {}) or {}
            code = status.get("code", "")
            return 0 if state == "JOB_STATE_COMPLETED" and code == "OK" else 2

        if time.time() >= deadline:
            print(f"{_now_utc()} timeout_reached=true", file=sys.stderr)
            return 3

        time.sleep(max(1, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
