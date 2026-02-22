"""MCPMark-lite filesystem benchmark for RFT-ready tool-calling evals."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message, MetricResult
from eval_protocol.pytest import AgentRolloutProcessor, evaluation_test

from benchmark.verifier import evaluate_task

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.getenv("TASK_WORKSPACE_ROOT", "/tmp/mcpmark-lite-rft/workspaces")).resolve()
MAX_TOKENS = int(os.getenv("MCP_AGENT_MAX_TOKENS", "512"))
ROLLOUT_STEPS = int(os.getenv("MCP_AGENT_STEPS", "10"))
PASSED_THRESHOLD = float(os.getenv("MCP_PASSED_THRESHOLD", "0.0"))
MAX_CONCURRENCY = int(os.getenv("MCP_MAX_CONCURRENT_ROLLOUTS", "2"))

DEFAULT_MODEL = "fireworks_ai/accounts/fireworks/models/qwen3-8b"
SYSTEM_PROMPT = (
    "You are a precise filesystem automation agent. "
    "You must use tools instead of guessing file contents. "
    "Always call init_task(task_id=...) before any other task tools. "
    "When finished, respond with TASK_COMPLETE."
)


def task_dataset_adapter(rows: List[Dict[str, Any]]) -> List[EvaluationRow]:
    eval_rows: List[EvaluationRow] = []

    for row in rows:
        task_id = str(row["task_id"])
        checks = row.get("checks", [])
        min_tool_calls = int(row.get("min_tool_calls", 3))

        # Keep each run deterministic by cleaning task workspace before rollout.
        task_dir = WORKSPACE_ROOT / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)

        user_prompt = (
            f"TASK_ID: {task_id}\n"
            f"{row['user_prompt']}\n"
            "Use tools to complete the task and then reply with TASK_COMPLETE."
        )

        eval_rows.append(
            EvaluationRow(
                messages=[
                    Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ],
                input_metadata=InputMetadata(
                    row_id=task_id,
                    dataset_info={
                        "task_id": task_id,
                        "checks": checks,
                        "min_tool_calls": min_tool_calls,
                    },
                ),
            )
        )

    return eval_rows


@evaluation_test(
    input_dataset=["data/tasks.jsonl"],
    dataset_adapter=task_dataset_adapter,
    rollout_processor=AgentRolloutProcessor(),
    completion_params=[
        {
            "model": os.getenv("MCP_AGENT_MODEL", DEFAULT_MODEL),
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
        }
    ],
    mcp_config_path="mcp_config/task_files_stdio.json",
    mode="pointwise",
    passed_threshold=PASSED_THRESHOLD,
    steps=ROLLOUT_STEPS,
    max_concurrent_rollouts=MAX_CONCURRENCY,
)
def test_mcpmark_lite_filesystem(row: EvaluationRow) -> EvaluationRow:
    dataset_info = (row.input_metadata.dataset_info or {}) if row.input_metadata else {}
    task_id = str(dataset_info.get("task_id", row.input_metadata.row_id if row.input_metadata else "unknown"))
    checks = dataset_info.get("checks", []) if isinstance(dataset_info, dict) else []
    min_tool_calls = int(dataset_info.get("min_tool_calls", 3)) if isinstance(dataset_info, dict) else 3

    task_dir = WORKSPACE_ROOT / task_id
    verifier_score, failures, successes = evaluate_task(task_dir, checks)

    tool_call_count = sum(1 for message in row.messages if message.role == "tool")
    tool_call_ratio = min(1.0, tool_call_count / float(max(1, min_tool_calls)))
    min_tool_calls_met = tool_call_count >= min_tool_calls

    final_score = verifier_score * (1.0 if min_tool_calls_met else 0.5)

    if failures:
        reason = f"Task {task_id} failed checks: " + " | ".join(failures)
    else:
        reason = f"Task {task_id} passed all deterministic checks."

    row.evaluation_result = EvaluateResult(
        score=final_score,
        reason=reason,
        metrics={
            "verifier_score": MetricResult(
                score=verifier_score,
                reason=f"Deterministic checks passed: {len(successes)}/{len(checks)}",
                is_score_valid=True,
                data={"failures": failures, "successes": successes},
            ),
            "tool_call_count": MetricResult(
                score=tool_call_ratio,
                reason=(
                    f"Observed {tool_call_count} tool calls, minimum target {min_tool_calls}."
                ),
                is_score_valid=True,
                data={"raw_count": tool_call_count, "min_required": min_tool_calls},
            ),
            "min_tool_calls_met": MetricResult(
                score=1.0 if min_tool_calls_met else 0.0,
                reason="Minimum tool-call threshold met." if min_tool_calls_met else "Tool-call threshold missed.",
                is_score_valid=True,
            ),
        },
        is_score_valid=True,
    )

    return row
