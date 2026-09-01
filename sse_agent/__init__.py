"""sse_agent — a replication of the SKILL.state bounded-state agent runtime
(arxiv.org/html/2608.26263v2), built on top of agentknit.

See ``README.md`` for the paper summary, design notes and how to reproduce
the benchmark results in ``data/``.
"""

from __future__ import annotations

from ._core import (
    CONVERSATIONAL_CONTRACT,
    STRUCTURED_STATE_CONTRACT,
    EpisodeResult,
    Mode,
    ObservationFn,
    ParsedStep,
    StepRecord,
    merge_state,
    parse_response,
    run_episode,
)
from ._tools import CODING_TOOL_CATALOG, make_coding_step_fn

__all__ = [
    "CODING_TOOL_CATALOG",
    "CONVERSATIONAL_CONTRACT",
    "STRUCTURED_STATE_CONTRACT",
    "EpisodeResult",
    "Mode",
    "ObservationFn",
    "ParsedStep",
    "StepRecord",
    "make_coding_step_fn",
    "merge_state",
    "parse_response",
    "run_episode",
]

__version__ = "0.1.0"
