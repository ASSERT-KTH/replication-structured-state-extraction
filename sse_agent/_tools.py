"""Coding-task tool dispatch: thin wrapper around agentknit's tool_library.

Actions are plain JSON (``{"tool": "read_file", "args": {"path": "..."}}``)
rather than native function-calling — the model is never shown a
tool-calling schema, only the text catalog below, so nothing outside
``(P, Sigma_t, O_t)`` leaks into the prompt.
"""

from __future__ import annotations

from typing import Any

from agentknit.tool_library import t_read, t_run, t_update, t_write

from ._core import ObservationFn

CODING_TOOL_CATALOG = (
    '- read_file {"path": str}\n'
    '- write_file {"path": str, "content": str}\n'
    '- str_replace {"path": str, "old_str": str, "new_str": str}\n'
    '- exec_shell {"command": str}\n'
    '- finish {"summary": str}  -- call once the task is complete; ends the episode'
)

_OBSERVATION_CHAR_LIMIT = 4000


def make_coding_step_fn(max_steps: int) -> ObservationFn:
    """Return a ``step_fn`` that dispatches one coding-tool action per step."""

    def step_fn(t: int, action: dict[str, Any], _patch: dict[str, Any]) -> tuple[str, bool]:
        tool = action.get("tool")
        args = action.get("args") or {}
        try:
            if tool == "read_file":
                text, _meta = t_read(args.get("path", ""))
            elif tool == "write_file":
                text, _meta = t_write(args.get("path", ""), args.get("content", ""))
            elif tool == "str_replace":
                text, _meta = t_update(
                    path=args.get("path", ""),
                    old=args.get("old_str", ""),
                    new=args.get("new_str", ""),
                )
            elif tool == "exec_shell":
                text, _meta = t_run(args.get("command", ""))
            elif tool == "noop":
                text = "No action taken — reply must include a valid action JSON block."
            else:
                text = f"ERROR: unknown tool {tool!r}. Valid tools: read_file, write_file, str_replace, exec_shell, finish."
        except Exception as exc:  # noqa: BLE001 - surface any tool failure as an observation
            text = f"ERROR: {exc}"

        observation = text[:_OBSERVATION_CHAR_LIMIT]
        done = (t + 1) >= max_steps
        if done:
            observation += "\n\n(step budget exhausted — call finish now)"
        return observation, done

    return step_fn
