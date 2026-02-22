#!/usr/bin/env python3
"""Build an RFT-ready JSONL dataset from raw MCPMark-lite task rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.test_mcp_filesystem_rft import task_dataset_adapter


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize EvaluationRow JSONL for RFT job creation.")
    parser.add_argument("--input", default="data/tasks.jsonl", help="Raw task JSONL path.")
    parser.add_argument("--output", default="data/rft_tasks.jsonl", help="Output EvaluationRow JSONL path.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for cheap smoke runs.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    rows = _load_jsonl(input_path)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    eval_rows = task_dataset_adapter(rows)

    out_lines: list[str] = []
    for row in eval_rows:
        payload: dict[str, Any] = {
            "messages": [message.model_dump(mode="json", exclude_none=True) for message in row.messages],
        }
        if row.input_metadata is not None:
            metadata: dict[str, Any] = {"row_id": row.input_metadata.row_id}
            if row.input_metadata.dataset_info is not None:
                metadata["dataset_info"] = row.input_metadata.dataset_info
            payload["input_metadata"] = metadata
        if "messages" not in payload:
            raise ValueError("Materialized row is missing required 'messages' key.")
        out_lines.append(json.dumps(payload, ensure_ascii=False))

    _write_jsonl(output_path, out_lines)
    print(f"Wrote {len(out_lines)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
