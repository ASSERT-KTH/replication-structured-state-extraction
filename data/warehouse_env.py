#!/usr/bin/env python3
"""Small long-horizon scaling experiment, modelled on the paper's warehouse
long-horizon benchmark (arxiv 2608.26263v2, Table 1): a scripted stream of
inventory events is fed to the agent one at a time, and after each event the
agent must report the running totals. There is no real environment beyond
this scripted ledger — the point is purely to measure how prompt size and
answer accuracy behave as the number of steps T grows, under two memory
strategies:

  * ``structured_state`` — the sse_agent runtime in this repo: bounded prompt,
    a persisted JSON ledger, reasoning discarded every turn.
  * ``conversational``   — the same model/task/tool contract, but with the
    full transcript kept and resent every turn (the paper's "Prompt/ReAct"
    baseline).

Both runs at a given horizon share the same event sequence (same seed), so
the only variable is the memory mechanism. Usage (prompt_tokens per call) is
read from the provider's own accounting, not estimated.

Usage:
    python data/warehouse_env.py --horizons 5 10 20 --n-shelves 12 --out data/warehouse_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agentknit  # noqa: E402

from sse_agent import Mode, StepRecord, run_episode  # noqa: E402

MODEL = "/home/martin/bin/deepseek-v4-flash-completions.py"
ENDPOINT = "run:///home/martin/bin/deepseek-v4-flash-completions.py"

TASK_SPEC = (
    "You are tracking inventory for a warehouse with a fixed set of numbered "
    "shelves. Each turn you are told about one event: a shelf was STOREd into "
    "(now occupied), a shelf was SHIPped out of (now empty), or nothing "
    "happened (WAIT). You must always report the CURRENT totals after "
    "incorporating the latest event: how many shelves are occupied right now, "
    "and how many items have been shipped in total since the start."
)

REPORT_TOOL_CATALOG = (
    '- report {"occupied_count": int, "shipped_total": int}  '
    "-- the only action; call it every turn with your current totals"
)


class WarehouseEnv:
    """Scripted ground-truth ledger the agent must keep track of."""

    def __init__(self, n_shelves: int, horizon: int, seed: int) -> None:
        self.rng = random.Random(seed)
        self.n_shelves = n_shelves
        self.horizon = horizon
        self.occupied = {i: False for i in range(n_shelves)}
        self.shipped = 0
        self.gt_log: list[tuple[int, int]] = []

    def _apply_random_event(self) -> str:
        empty = [i for i, occ in self.occupied.items() if not occ]
        full = [i for i, occ in self.occupied.items() if occ]
        choices = (["STORE"] if empty else []) + (["SHIP"] if full else []) + ["WAIT"]
        kind = self.rng.choice(choices)
        if kind == "STORE":
            shelf = self.rng.choice(empty)
            self.occupied[shelf] = True
            description = f"STORE shelf_{shelf:03d}"
        elif kind == "SHIP":
            shelf = self.rng.choice(full)
            self.occupied[shelf] = False
            self.shipped += 1
            description = f"SHIP shelf_{shelf:03d}"
        else:
            description = "WAIT (no change)"
        self.gt_log.append((sum(self.occupied.values()), self.shipped))
        return description

    def _render(self, event_no: int, description: str) -> str:
        return (
            f"Event #{event_no}: {description}\n"
            'Report the CURRENT totals as action {"tool": "report", '
            '"args": {"occupied_count": <int>, "shipped_total": <int>}}.'
        )

    def initial_observation(self) -> str:
        description = self._apply_random_event()
        return self._render(1, description)

    def step_fn(self, t: int, action: dict[str, Any], patch: dict[str, Any]) -> tuple[str, bool]:
        next_index = t + 1
        if next_index >= self.horizon:
            return "", True
        description = self._apply_random_event()
        return self._render(next_index + 1, description), False


def score(steps: list[StepRecord], gt_log: list[tuple[int, int]]) -> dict[str, Any]:
    assert len(steps) == len(gt_log), (len(steps), len(gt_log))
    correct = 0
    per_step = []
    for step, (gt_occ, gt_shipped) in zip(steps, gt_log):
        args = step.action.get("args") or {}
        reported_occ = args.get("occupied_count")
        reported_shipped = args.get("shipped_total")
        is_correct = reported_occ == gt_occ and reported_shipped == gt_shipped
        correct += int(is_correct)
        per_step.append(
            {
                "step": step.step,
                "gt_occupied": gt_occ,
                "gt_shipped": gt_shipped,
                "reported_occupied": reported_occ,
                "reported_shipped": reported_shipped,
                "correct": is_correct,
                "prompt_tokens": step.prompt_tokens,
                "completion_tokens": step.completion_tokens,
                "prompt_chars": step.prompt_chars,
                "parse_error": step.parse_error,
            }
        )
    return {"accuracy": correct / len(steps) if steps else 0.0, "per_step": per_step}


def run_one(client: Any, horizon: int, n_shelves: int, mode: Mode, seed: int,
            sleep_s: float) -> dict[str, Any]:
    env = WarehouseEnv(n_shelves, horizon, seed=seed)
    initial_observation = env.initial_observation()

    def on_step(record: StepRecord) -> None:
        extra = f" parse_error={record.parse_error}" if record.parse_error else ""
        print(
            f"  [{mode} T={horizon} step {record.step}] "
            f"prompt_tokens={record.prompt_tokens} completion_tokens={record.completion_tokens}{extra}",
            flush=True,
        )

    result = run_episode(
        client,
        MODEL,
        task_spec=TASK_SPEC,
        tool_catalog=REPORT_TOOL_CATALOG,
        initial_observation=initial_observation,
        step_fn=env.step_fn,
        mode=mode,
        max_steps=horizon,
        sleep_s=sleep_s,
        on_step=on_step,
    )
    scored = score(result.steps, env.gt_log)
    cumulative_prompt_tokens = 0
    cumulative_series = []
    for step in result.steps:
        cumulative_prompt_tokens += step.prompt_tokens
        cumulative_series.append(cumulative_prompt_tokens)

    return {
        "mode": mode,
        "horizon": horizon,
        "n_shelves": n_shelves,
        "seed": seed,
        "stopped_reason": result.stopped_reason,
        "accuracy": scored["accuracy"],
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_completion_tokens": result.total_completion_tokens,
        "total_tokens": result.total_tokens,
        "final_prompt_tokens_this_step": result.steps[-1].prompt_tokens if result.steps else None,
        "cumulative_prompt_tokens_series": cumulative_series,
        "per_step": scored["per_step"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--n-shelves", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.2, help="delay between LLM calls")
    parser.add_argument("--out", default=str(Path(__file__).parent / "warehouse_results.json"))
    parser.add_argument("--csv-out", default=str(Path(__file__).parent / "warehouse_results.csv"))
    args = parser.parse_args()

    client = agentknit.create_client({"model": MODEL, "endpoint": ENDPOINT})

    all_results = []
    for horizon in args.horizons:
        for mode in ("structured_state", "conversational"):
            print(f"\n=== running mode={mode} horizon={horizon} ===", flush=True)
            all_results.append(run_one(client, horizon, args.n_shelves, mode, args.seed, args.sleep))

    Path(args.out).write_text(json.dumps(all_results, indent=2))

    with open(args.csv_out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "horizon", "accuracy", "total_prompt_tokens",
                          "total_completion_tokens", "final_prompt_tokens_this_step"])
        for row in all_results:
            writer.writerow([row["mode"], row["horizon"], f"{row['accuracy']:.3f}",
                              row["total_prompt_tokens"], row["total_completion_tokens"],
                              row["final_prompt_tokens_this_step"]])

    print(f"\nWrote {args.out} and {args.csv_out}")
    print(f"{'mode':<18}{'T':>5}{'accuracy':>10}{'total_prompt_tok':>18}{'last_step_prompt_tok':>22}")
    for row in all_results:
        print(f"{row['mode']:<18}{row['horizon']:>5}{row['accuracy']:>10.2f}"
              f"{row['total_prompt_tokens']:>18}{row['final_prompt_tokens_this_step']:>22}")


if __name__ == "__main__":
    main()
