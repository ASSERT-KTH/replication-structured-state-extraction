"""Bounded structured-state agent runtime.

Replicates the "SKILL.state" execution model described in
arxiv.org/html/2608.26263v2: instead of an append-only conversation history,
the agent carries a small mutable JSON state (`Sigma`) across steps. Each
turn the model receives only three things — the immutable task/tool
specification (`P`), the current state (`Sigma_t`), and the latest
observation (`O_t`) — generates free-text reasoning plus a JSON block
containing a state patch and an action, and the reasoning is discarded
after the patch is applied. This keeps the prompt sent on every turn
bounded, independent of how many steps have elapsed, instead of growing
with the full transcript.

A `mode="conversational"` runtime is also provided in the same loop so the
two memory strategies can be compared step-for-step against the same model,
tools and task — the only variable that changes is what the model is shown.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

Mode = Literal["structured_state", "conversational"]

# The model is asked to always answer with reasoning followed by exactly one
# fenced ```json block. This regex grabs the *last* such block, in case the
# model's reasoning happens to contain an earlier code fence.
_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
# Fallback for models that forget the fence: grab the last brace-delimited blob.
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def merge_state(state: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply *patch* onto *state* (Sigma_{t+1} = Sigma_t (+) Delta Sigma_t).

    A ``null`` value deletes the key. A nested object merges recursively
    into an existing nested object at the same key; any other value type
    simply overwrites.
    """
    result = dict(state)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_state(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class ParsedStep:
    reasoning: str
    state_patch: dict[str, Any]
    action: dict[str, Any]
    parse_error: str | None = None


def parse_response(text: str) -> ParsedStep:
    """Split a model reply into discarded reasoning, a state patch and an action.

    Tolerates the model forgetting the ```json fence or omitting one of the
    two keys; a genuinely unparsable reply becomes a no-op action with
    ``parse_error`` set, mirroring the JSON-formatting-error failure mode
    the paper reports for weaker open-weight models.
    """
    text = text or ""
    match = _JSON_FENCE_RE.search(text)
    raw = match.group(1) if match else None
    if raw is None:
        candidates = _BARE_JSON_RE.findall(text)
        raw = candidates[-1] if candidates else None
    if raw is None:
        return ParsedStep(text, {}, {"tool": "noop"}, parse_error="no_json_block_found")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ParsedStep(text, {}, {"tool": "noop"}, parse_error=f"json_decode_error: {exc}")
    if not isinstance(payload, dict):
        return ParsedStep(text, {}, {"tool": "noop"}, parse_error="json_not_object")

    state_patch = payload.get("state_patch")
    if not isinstance(state_patch, dict):
        state_patch = {}
    action = payload.get("action")
    if not isinstance(action, dict) or "tool" not in action:
        action = {"tool": "noop"}

    reasoning = text[: match.start()] if match else ""
    return ParsedStep(reasoning, state_patch, action)


@dataclass
class StepRecord:
    step: int
    prompt_tokens: int
    completion_tokens: int
    prompt_chars: int
    action: dict[str, Any]
    state_patch: dict[str, Any] | None
    observation_preview: str
    parse_error: str | None = None


@dataclass
class EpisodeResult:
    mode: Mode
    steps: list[StepRecord] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    final_reply: str | None = None
    stopped_reason: str = "max_steps"

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


# step_fn(step_index, action, state_patch) -> (next_observation_text, done)
ObservationFn = Callable[[int, dict[str, Any], dict[str, Any]], tuple[str, bool]]

STRUCTURED_STATE_CONTRACT = (
    "You are an execution agent. You do NOT see the conversation history — "
    "each turn you are given only your own persisted state and the latest "
    "observation; everything else has been discarded.\n\n"
    "Every reply MUST have exactly this shape:\n"
    "1. Free-text reasoning (scratch thinking; discarded after this turn).\n"
    "2. A single fenced block:\n"
    '```json\n{"state_patch": {...}, "action": {"tool": "...", "args": {...}}}\n```\n\n'
    "`state_patch` is merged onto your current state: existing keys survive "
    "unless the patch overwrites them, a null value DELETES that key, and "
    "nested objects merge recursively. Persist everything you will need on "
    "the next turn (plans, counters, findings) in state_patch — nothing else "
    "survives to the next turn."
)

CONVERSATIONAL_CONTRACT = (
    "You are an execution agent with access to the full conversation so far. "
    "Every reply MUST have exactly this shape:\n"
    "1. Free-text reasoning.\n"
    "2. A single fenced block:\n"
    '```json\n{"action": {"tool": "...", "args": {...}}}\n```'
)


def _call_llm(client: Any, model: str, messages: list[dict[str, str]],
              *, retries: int = 3, backoff_s: float = 2.0) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return client.chat.completions.create(model=model, messages=messages, temperature=0)
        except Exception as exc:  # noqa: BLE001 - backends raise plain RuntimeError/HTTPError
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_exc}") from last_exc


def run_episode(
    client: Any,
    model: str,
    *,
    task_spec: str,
    tool_catalog: str,
    initial_observation: str,
    step_fn: ObservationFn,
    mode: Mode = "structured_state",
    max_steps: int = 30,
    sleep_s: float = 0.0,
    on_step: Callable[[StepRecord], None] | None = None,
) -> EpisodeResult:
    """Run Algorithm 1 (or its conversational counterpart) for up to *max_steps* turns.

    ``step_fn`` is supplied by the caller and owns the environment: given the
    step index and the parsed action/state-patch it returns the next
    observation text and whether the episode is finished. For a coding task
    this dispatches a real tool; for the synthetic long-horizon benchmark it
    just advances a scripted event generator.
    """
    contract = STRUCTURED_STATE_CONTRACT if mode == "structured_state" else CONVERSATIONAL_CONTRACT
    system_prompt = f"{contract}\n\n## Task\n{task_spec}\n\n## Tools\n{tool_catalog}"

    state: dict[str, Any] = {}
    observation = initial_observation
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    result = EpisodeResult(mode=mode)

    for t in range(max_steps):
        if mode == "structured_state":
            user_content = (
                f"## Current state (Sigma)\n```json\n{json.dumps(state, indent=2)}\n```\n\n"
                f"## Latest observation\n{observation}\n"
            )
            messages = [history[0], {"role": "user", "content": user_content}]
        else:
            history.append({"role": "user", "content": observation})
            messages = list(history)

        response = _call_llm(client, model, messages)
        text = response.choices[0].message.content or ""
        parsed = parse_response(text)

        if mode == "structured_state":
            state = merge_state(state, parsed.state_patch)
        else:
            history.append({"role": "assistant", "content": text})

        usage = response.usage
        record = StepRecord(
            step=t,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            prompt_chars=sum(len(m["content"]) for m in messages),
            action=parsed.action,
            state_patch=parsed.state_patch if mode == "structured_state" else None,
            observation_preview=observation[:300],
            parse_error=parsed.parse_error,
        )
        result.steps.append(record)
        if on_step is not None:
            on_step(record)

        if parsed.action.get("tool") == "finish":
            result.final_reply = (parsed.action.get("args") or {}).get("summary")
            result.stopped_reason = "finish"
            break

        observation, done = step_fn(t, parsed.action, parsed.state_patch)
        if done:
            result.stopped_reason = "env_done"
            break
        if sleep_s:
            time.sleep(sleep_s)

    result.final_state = state
    return result
