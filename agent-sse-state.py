#!/usr/bin/env python3
"""External-python agent_benchmark profile: runs sse_agent (SKILL.state
replication, see README.md) non-interactively against a free, unauthenticated
model endpoint.

Calling convention matches agent_workflow's external-python profile
(see ../agentknit/secondguess_agent.py and ~/.local/share/agent-workflow/agents
for sibling examples): benchmark.py invokes this script with the process cwd
already set to the isolated task working directory, and args
``--non-interactive "<task prompt>"``.

Usage from agent_benchmark:
    python benchmark.py /path/to/agent-sse-state.py tasks/<task-name>
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import agentknit  # noqa: E402

from sse_agent import CODING_TOOL_CATALOG, make_coding_step_fn, run_episode  # noqa: E402

MODEL = "/home/martin/bin/deepseek-v4-flash-completions.py"
ENDPOINT = "run:///home/martin/bin/deepseek-v4-flash-completions.py"
MAX_STEPS = int(os.environ.get("SSE_AGENT_MAX_STEPS", "15"))
STEP_SLEEP_S = float(os.environ.get("SSE_AGENT_SLEEP_S", "0.3"))


def _on_step(record) -> None:  # type: ignore[no-untyped-def]
    extra = f" parse_error={record.parse_error}" if record.parse_error else ""
    print(
        f"[step {record.step}] tool={record.action.get('tool')} "
        f"prompt_tokens={record.prompt_tokens} completion_tokens={record.completion_tokens}{extra}",
        flush=True,
    )


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--non-interactive":
        args = args[1:]
    if not args:
        print("usage: agent-sse-state.py [--non-interactive] <task prompt>", file=sys.stderr)
        sys.exit(2)
    task_prompt = " ".join(args)

    client = agentknit.create_client({"model": MODEL, "endpoint": ENDPOINT})
    step_fn = make_coding_step_fn(MAX_STEPS)

    result = run_episode(
        client,
        MODEL,
        task_spec=task_prompt,
        tool_catalog=CODING_TOOL_CATALOG,
        initial_observation="(episode start — no actions taken yet)",
        step_fn=step_fn,
        mode="structured_state",
        max_steps=MAX_STEPS,
        sleep_s=STEP_SLEEP_S,
        on_step=_on_step,
    )

    summary = {
        "mode": result.mode,
        "stopped_reason": result.stopped_reason,
        "final_reply": result.final_reply,
        "final_state": result.final_state,
        "num_steps": len(result.steps),
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_completion_tokens": result.total_completion_tokens,
    }
    print(f"\n=== sse-agent finished: {result.stopped_reason} ===")
    print("SSE_SUMMARY: " + json.dumps(summary))


if __name__ == "__main__":
    main()
