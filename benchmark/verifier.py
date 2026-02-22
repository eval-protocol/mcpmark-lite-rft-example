"""Deterministic verifier for MCPMark-lite filesystem tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _normalize_text(content: str) -> str:
    # Keep deterministic comparison while tolerating trailing whitespace differences.
    return content.replace("\r\n", "\n").rstrip() + "\n"


def run_check(task_dir: Path, check: Dict[str, Any]) -> Tuple[bool, str]:
    check_type = str(check.get("type", "")).strip().lower()
    rel_path = str(check.get("path", ""))
    target = (task_dir / rel_path).resolve()

    if check_type in {"json_equals", "text_equals", "file_contains"} and not target.exists():
        return False, f"missing file: {rel_path}"

    if check_type == "json_equals":
        expected = check.get("value")
        try:
            got = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"invalid json at {rel_path}: {exc}"
        if got == expected:
            return True, f"json_equals passed: {rel_path}"
        return False, f"json mismatch at {rel_path}: expected={expected!r} got={got!r}"

    if check_type == "text_equals":
        expected_text = _normalize_text(str(check.get("value", "")))
        got_text = _normalize_text(target.read_text(encoding="utf-8"))
        if got_text == expected_text:
            return True, f"text_equals passed: {rel_path}"
        return False, f"text mismatch at {rel_path}: expected={expected_text!r} got={got_text!r}"

    if check_type == "file_contains":
        needle = str(check.get("value", ""))
        got_text = target.read_text(encoding="utf-8")
        if needle in got_text:
            return True, f"file_contains passed: {rel_path}"
        return False, f"missing substring in {rel_path}: {needle!r}"

    return False, f"unknown check type: {check_type}"


def evaluate_task(task_dir: Path, checks: List[Dict[str, Any]]) -> Tuple[float, List[str], List[str]]:
    if not checks:
        return 0.0, ["no checks configured"], []

    passed = 0
    failures: List[str] = []
    successes: List[str] = []

    for check in checks:
        ok, reason = run_check(task_dir, check)
        if ok:
            passed += 1
            successes.append(reason)
        else:
            failures.append(reason)

    return passed / len(checks), failures, successes
