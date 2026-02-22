# MCPMark-lite RFT Example

Standalone, Docker-packaged, multi-turn tool-calling benchmark that is compatible with `eval-protocol` and Fireworks RFT.

This repo is a pragmatic subset inspired by MCP benchmark design:
- real tool execution against mutable state (filesystem)
- deterministic verifier over post-rollout environment state
- low infra overhead (single local FastMCP server over stdio)

## Why this subset
For an end-to-end RFT example, the cleanest path is filesystem-only MCP tasks:
- `MCPMark` emphasizes verifier-driven realism and notes filesystem tasks can run with zero API-key setup in a quickstart path.
- `MCP-Universe` is valuable but includes optional internet/API-dependent domains and a broader server matrix.
- `MCP-Bench` is comprehensive but setup-heavy (multiple provider keys + Docker stack + richer harness requirements).

References:
- [MCPMark repo](https://github.com/microsoft/mcpmark)
- [MCP-Universe repo](https://github.com/LeapLabTHU/MCP-Universe)
- [MCP-Bench repo](https://github.com/SageSELab/mcp-bench)
- [MCP-AgentBench paper](https://arxiv.org/abs/2508.05715)

## What is included
- `mcp_server/task_files_server.py`: local FastMCP server with task-scoped filesystem tools.
- `data/tasks.jsonl`: 8 deterministic multi-turn tasks.
- `benchmark/test_mcp_filesystem_rft.py`: `@evaluation_test` benchmark using `AgentRolloutProcessor` + deterministic verifier.
- `benchmark/verifier.py`: strict file-state checks (`json_equals`, `text_equals`, `file_contains`).
- `Dockerfile`: standalone runnable container.

## Tooling pattern
Each rollout is expected to:
1. call `init_task(task_id)`
2. use `list_files` / `read_file`
3. produce required output files with `write_file`
4. append completion marker in checklist via `append_file`

Reward is computed from real filesystem state, not just assistant text.

## Local setup
```bash
uv sync
```

Set Fireworks auth:
```bash
export FIREWORKS_API_KEY=...
```

Optional (default is a small qwen model):
```bash
export MCP_AGENT_MODEL=fireworks_ai/accounts/fireworks/models/qwen3-8b
```

Optional low-cost knobs:
```bash
export MCP_AGENT_STEPS=8
export MCP_AGENT_MAX_TOKENS=512
export MCP_MAX_CONCURRENT_ROLLOUTS=1
```

## Run benchmark
```bash
uv run pytest benchmark/test_mcp_filesystem_rft.py::test_mcpmark_lite_filesystem -q -s
```

Small smoke run:
```bash
EP_MAX_DATASET_ROWS=1 MCP_AGENT_STEPS=6 MCP_AGENT_MAX_TOKENS=512 uv run pytest benchmark/test_mcp_filesystem_rft.py::test_mcpmark_lite_filesystem -q -s
```

## Docker run
```bash
docker build -t mcpmark-lite-rft .
docker run --rm -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" mcpmark-lite-rft
```

## Fireworks RFT flow
Use a known evaluator id for this test:
- `test-mcp-filesystem-rft-test-mcpmark-lite-filesystem`

Create RFT (base model required):
```bash
uv run ep create rft \
  --evaluator test-mcp-filesystem-rft-test-mcpmark-lite-filesystem \
  --base-model accounts/fireworks/models/qwen3-8b \
  --yes \
  --ignore-docker \
  --skip-validation
```

Notes:
- In this `python-sdk` branch, `create rft` auto-detects JSONL input dataset from `@evaluation_test(input_dataset=[...])` in many cases.
- If auto-detection fails in your environment, create/upload dataset first and rerun with `--dataset <dataset_id>`.

## Benchmark design constraints
- Deterministic checks make reward stable for RL.
- Task-scoped sandboxes prevent cross-row contamination.
- No external APIs required by default, which keeps rollout generation cost and failure modes low.
