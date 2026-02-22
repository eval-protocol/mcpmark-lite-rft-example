from __future__ import annotations

import json
from pathlib import Path

from benchmark.verifier import evaluate_task, run_check


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_check_json_equals(tmp_path: Path) -> None:
    _write(tmp_path / "out.json", json.dumps({"a": 1}))
    ok, reason = run_check(tmp_path, {"type": "json_equals", "path": "out.json", "value": {"a": 1}})
    assert ok is True
    assert "passed" in reason


def test_run_check_text_equals_trailing_whitespace_tolerant(tmp_path: Path) -> None:
    _write(tmp_path / "out.txt", "hello\n")
    ok, _ = run_check(tmp_path, {"type": "text_equals", "path": "out.txt", "value": "hello"})
    assert ok is True


def test_evaluate_task_collects_failures(tmp_path: Path) -> None:
    _write(tmp_path / "a.txt", "alpha\n")
    checks = [
        {"type": "text_equals", "path": "a.txt", "value": "alpha\n"},
        {"type": "file_contains", "path": "a.txt", "value": "beta"},
    ]
    score, failures, successes = evaluate_task(tmp_path, checks)
    assert score == 0.5
    assert len(successes) == 1
    assert len(failures) == 1
