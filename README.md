# sse-agent

A small-scale replication of the bounded-state agent runtime described in
["SKILL.state: Scalable Long-Horizon Agent Skills"](https://arxiv.org/html/2608.26263v2),
built on top of [agentknit](https://github.com/monperrus/agentknit) and run
against [agent_benchmark](https://github.com/monperrus/agent_benchmark).

## The paper, in short

Standard tool-using agents keep an append-only conversation transcript: every
tool call and every result is appended to the history and resent, in full,
on every subsequent model call. Prompt size then grows with the number of
steps taken, and cumulative token spend across an episode grows
quadratically in the number of steps.

The paper's fix is architectural: replace the growing transcript with a
small, explicit, mutable state object. On each step the model is shown only
three things — a fixed task/tool specification, its own current state, and
the latest observation — and asked to produce (a) reasoning that is
discarded immediately, and (b) a patch to merge onto its state plus the next
action. Because the prompt no longer includes history, its size stays
roughly constant as the episode gets longer, and cumulative token spend
grows linearly instead of quadratically. The tradeoff is that anything the
model doesn't explicitly write into the patch is gone on the next turn — the
state has to be a sufficient summary of everything future decisions depend
on.

## What's implemented here

`sse_agent/` is a small runtime, generic over the environment, implementing
exactly that loop on top of agentknit's model-connection layer
(`agentknit.create_client`, auth resolution, `run://` subprocess support):

- `merge_state()` — the patch operator: recursive dict merge, `null` deletes
  a key, matching the paper's "dictionary merge with null-deletion".
- `parse_response()` — splits a reply into (discarded) reasoning and a JSON
  block carrying `state_patch` + `action`; degrades to a no-op with a
  recorded `parse_error` on malformed output rather than crashing the loop.
- `run_episode()` — the step loop. Each turn sends exactly
  `[system_prompt, user_message]` where `user_message` is `Sigma_t` (current
  state) + `O_t` (latest observation) — nothing else. A `mode="conversational"`
  branch runs the *same* task/tools/model through a plain growing transcript
  instead, so the two memory strategies can be A/B'd with everything else
  held fixed.
- `sse_agent/_tools.py` — a coding-task action space (`read_file`,
  `write_file`, `str_replace`, `exec_shell`, `finish`) dispatched straight to
  `agentknit.tool_library`, exposed to the model as a plain-text catalog
  rather than a native function-calling schema (so the tool schema itself
  doesn't leak into every turn's token count the way it would with
  structured tool calls).

### Deviations from the paper (and why)

This is a proof-of-concept at a much smaller scale than the paper, run
against a single model (DeepSeek V4 Flash, via the official DeepSeek API
through a local OpenAI-compatible bridge script) rather than the paper's
model lineup or its InterCode-CTF / tau-bench benchmarks:

- **Action protocol.** The paper's runtime is environment-agnostic; here the
  action is always a JSON `{"tool": ..., "args": ...}` blob parsed out of
  plain text, not a native tool call. This keeps every call symmetric
  between the two `mode`s and keeps the tool schema out of the token count.
- **Scale.** The synthetic long-horizon experiment below uses horizons of
  5–30 steps and a 12-shelf warehouse, not the paper's T up to 200 over 500
  shelves — chosen to keep the real-API-call budget for this replication
  small while still making the O(T) vs. O(T²) trend visible.
- **No multi-model sweep, no InterCode-CTF/tau-bench run, no explicit
  state-recovery (Table 3) experiment.** Out of scope for this pass; the
  code is generic enough (`step_fn` callback) that another environment could
  be plugged in later.

## Repository layout

```
sse_agent/                  the runtime (core loop + coding-task tools)
tests/test_sse_agent.py     unit tests (state merge, response parsing, loop mechanics)
agent-sse-state.py          agent_benchmark external-python profile entry point
data/warehouse_env.py       synthetic long-horizon scaling experiment (paper's Table 1, downsized)
data/warehouse_results.{json,csv}   measured results from that experiment
data/agent_benchmark_results/       results copied from a real agent_benchmark run
```

## Running it

Unit tests (no network calls):

```bash
pip install -e . --group dev
pytest
```

### Long-horizon scaling experiment

Reruns the O(T) vs. O(T²) comparison from scratch (real API calls):

```bash
python data/warehouse_env.py --horizons 5 10 20 30 --n-shelves 12
```

### Against agent_benchmark

`agent-sse-state.py` follows `agent_workflow`'s external-python profile
convention (`--non-interactive "<prompt>"`, cwd already set to the isolated
task directory), so it can be pointed at directly:

```bash
cd ../agent_benchmark
python benchmark.py /path/to/agent-sse-state.py tasks/<task-name>
```

## Results

### Long-horizon scaling (`data/warehouse_env.py`)

A scripted stream of `STORE` / `SHIP` / `WAIT` inventory events is fed to
the agent one at a time; after each event it must report the running
occupied-shelf count and total shipped count. Both modes see the identical
event sequence at a given horizon T (same seed) — the only variable is
whether the agent gets a bounded state or the full transcript.

| mode | T | accuracy | total prompt tokens | prompt tokens on the *last* step |
|---|---:|---:|---:|---:|
| structured_state | 5  | 1.00 | 2,182  | 440   |
| conversational    | 5  | 1.00 | 2,275  | 609   |
| structured_state | 10 | 1.00 | 4,462  | 460   |
| conversational    | 10 | 1.00 | 6,475  | 994   |
| structured_state | 20 | 1.00 | 8,782  | 440   |
| conversational    | 20 | 1.00 | 24,637 | 2,174 |
| structured_state | 30 | 1.00 | 13,182 | 440   |
| conversational    | 30 | 1.00 | 43,761 | 2,749 |

The qualitative claim replicates cleanly, and this run scored perfectly
(1.00) in both modes at every horizon, so the token-growth curves aren't
confounded by accuracy differences: the structured-state agent's per-step
prompt size is flat (~440–460 tokens, independent of T — bounded by the
12-shelf state, not by history length), while the conversational baseline's
per-step prompt grows linearly with T (609 → 2,749 tokens over the same
range). Cumulative spend reflects that directly: 6× the horizon (5→30)
produces a ~19× increase in total prompt tokens for the conversational
baseline, versus ~6× for structured-state — i.e. close to the paper's
predicted O(T²) vs. O(T) split at this scale, and cleaner than an earlier
pass of this same experiment run against Claude Haiku 4.5 (kept in git
history), where a small-model parsing slip at T=5 briefly dented
structured-state's accuracy without affecting the token-growth trend.

### agent_benchmark tasks (`data/agent_benchmark_results/`)

Four representative tasks run via `agent-sse-state.py` through
`agent_benchmark`'s real runner (isolated working directory, real oracle
checks):

| task | outcome | steps | elapsed | total tokens (prompt/completion) |
|---|---|---:|---:|---:|
| tell-the-date | pass | 5 | 17.0s | 2,004 / 1,266 |
| count-files-in-dir | pass | 7 | 32.5s | 2,822 / 2,427 |
| csv-counting | **fail** | 6 | 37.5s | 2,575 / 3,454 |
| buggy-script-fix | pass | 15 | 84.5s | 13,578 / 7,897 |

(An earlier pass of these same four tasks against Claude Haiku 4.5 — kept in
git history — went 3 pass / 1 fail with the failure on a different task and
far fewer tokens; DeepSeek V4 Flash reasons more verbosely per step here,
which shows up directly as higher completion-token counts.)

The `csv-counting` failure is a genuine off-by-format error, not an
infrastructure problem: the agent computed the average of `10, 20, 30, 40,
50, 60` correctly but wrote `35` instead of `35.0` to `average_output.txt`,
and the oracle's regex requires the literal substring `35.0`. `buggy-script-fix`
took all 15 of its step budget and never called `finish`, but a `cat` of the
generated `summary.json` in its last tool call shows the correct values, so
the oracle passed anyway. Full stdout (including every step's token usage)
is preserved in `data/agent_benchmark_results/`.

These four tasks are short (3–6 tool calls), so they don't exercise the
long-horizon regime the paper — or the scaling experiment above — is about;
they're included mainly to show the runtime working end-to-end inside a
real benchmark harness with a real correctness oracle, not as a token-cost
comparison.

## License

MIT — see `LICENSE`.
