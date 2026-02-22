"""Local filesystem MCP server for MCPMark-lite tasks.

This server is intentionally simple and deterministic:
- Every tool call requires `task_id`.
- `init_task` always resets the task workspace from seed files.
- All file access is sandboxed under TASK_WORKSPACE_ROOT/<task_id>.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_CATALOG_PATH = Path(
    (REPO_ROOT / "data" / "tasks.jsonl")
)
TASK_WORKSPACE_ROOT = Path(os.getenv("TASK_WORKSPACE_ROOT", "/tmp/mcpmark-lite-rft/workspaces"))

mcp = FastMCP("mcpmark-lite-filesystem")


def _load_tasks() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    with TASK_CATALOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            task_id = str(task["task_id"])
            tasks[task_id] = task
    if not tasks:
        raise RuntimeError(f"No tasks found in {TASK_CATALOG_PATH}")
    return tasks


TASKS = _load_tasks()
TASK_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


def _task_dir(task_id: str) -> Path:
    return (TASK_WORKSPACE_ROOT / task_id).resolve()


def _require_task(task_id: str) -> dict[str, Any]:
    task = TASKS.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    return task


def _resolve_path(task_id: str, rel_path: str) -> Path:
    base = _task_dir(task_id)
    candidate = (base / rel_path).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"Path escapes task sandbox: {rel_path}")
    return candidate


def _ensure_initialized(task_id: str) -> Path:
    base = _task_dir(task_id)
    if not base.exists():
        raise ValueError(
            f"Task workspace for '{task_id}' does not exist yet. "
            "Call init_task(task_id=...) first."
        )
    return base


@mcp.tool
async def init_task(task_id: str) -> Dict[str, Any]:
    """Reset and initialize a task workspace from seed files."""
    task = _require_task(task_id)
    base = _task_dir(task_id)

    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    seeded_files: list[str] = []
    for rel_path, content in (task.get("seed_files") or {}).items():
        target = _resolve_path(task_id, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        seeded_files.append(rel_path)

    return {
        "task_id": task_id,
        "workspace": str(base),
        "seeded_files": sorted(seeded_files),
    }


@mcp.tool
async def list_files(task_id: str, subdir: str = ".") -> Dict[str, Any]:
    """List files under a task workspace subtree."""
    _ensure_initialized(task_id)
    root = _resolve_path(task_id, subdir)
    if not root.exists():
        return {"task_id": task_id, "files": []}

    files: list[str] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(_task_dir(task_id))))

    return {"task_id": task_id, "files": sorted(files)}


@mcp.tool
async def read_file(task_id: str, path: str) -> Dict[str, Any]:
    """Read UTF-8 content from a task file."""
    _ensure_initialized(task_id)
    target = _resolve_path(task_id, path)
    if not target.exists() or not target.is_file():
        raise ValueError(f"File not found: {path}")

    return {
        "task_id": task_id,
        "path": path,
        "content": target.read_text(encoding="utf-8"),
    }


@mcp.tool
async def write_file(task_id: str, path: str, content: str) -> Dict[str, Any]:
    """Write UTF-8 content to a task file (overwrite)."""
    _ensure_initialized(task_id)
    target = _resolve_path(task_id, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return {
        "task_id": task_id,
        "path": path,
        "bytes_written": len(content.encode("utf-8")),
    }


@mcp.tool
async def append_file(task_id: str, path: str, content: str) -> Dict[str, Any]:
    """Append UTF-8 content to a task file."""
    _ensure_initialized(task_id)
    target = _resolve_path(task_id, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(content)

    return {
        "task_id": task_id,
        "path": path,
        "bytes_appended": len(content.encode("utf-8")),
    }


@mcp.tool
async def get_task_prompt(task_id: str) -> Dict[str, Any]:
    """Return the canonical user prompt for a task."""
    task = _require_task(task_id)
    return {"task_id": task_id, "user_prompt": task.get("user_prompt", "")}


if __name__ == "__main__":
    mcp.run()
