from __future__ import annotations

from sse_agent import EpisodeResult, StepRecord, merge_state, parse_response, run_episode


def test_merge_state_overwrites_and_adds() -> None:
    state = {"a": 1, "b": 2}
    patch = {"b": 3, "c": 4}
    assert merge_state(state, patch) == {"a": 1, "b": 3, "c": 4}


def test_merge_state_null_deletes_key() -> None:
    state = {"a": 1, "b": 2}
    assert merge_state(state, {"b": None}) == {"a": 1}


def test_merge_state_deletes_missing_key_is_noop() -> None:
    assert merge_state({"a": 1}, {"z": None}) == {"a": 1}


def test_merge_state_merges_nested_dicts_recursively() -> None:
    state = {"shelves": {"1": True, "2": False}}
    patch = {"shelves": {"2": True, "3": True}}
    assert merge_state(state, patch) == {"shelves": {"1": True, "2": True, "3": True}}


def test_merge_state_nested_null_deletes_nested_key() -> None:
    state = {"shelves": {"1": True, "2": False}}
    patch = {"shelves": {"1": None}}
    assert merge_state(state, patch) == {"shelves": {"2": False}}


def test_merge_state_does_not_mutate_inputs() -> None:
    state = {"a": {"x": 1}}
    patch = {"a": {"y": 2}}
    merge_state(state, patch)
    assert state == {"a": {"x": 1}}
    assert patch == {"a": {"y": 2}}


def test_parse_response_extracts_fenced_json() -> None:
    text = (
        "I should store the value.\n"
        '```json\n{"state_patch": {"count": 1}, "action": {"tool": "noop", "args": {}}}\n```'
    )
    parsed = parse_response(text)
    assert parsed.state_patch == {"count": 1}
    assert parsed.action == {"tool": "noop", "args": {}}
    assert parsed.parse_error is None
    assert "should store" in parsed.reasoning


def test_parse_response_falls_back_to_bare_json() -> None:
    text = 'reasoning...\n{"state_patch": {}, "action": {"tool": "finish", "args": {}}}'
    parsed = parse_response(text)
    assert parsed.action == {"tool": "finish", "args": {}}
    assert parsed.parse_error is None


def test_parse_response_reports_error_when_no_json_found() -> None:
    parsed = parse_response("just some prose, no json at all")
    assert parsed.action == {"tool": "noop"}
    assert parsed.state_patch == {}
    assert parsed.parse_error == "no_json_block_found"


def test_parse_response_reports_error_on_malformed_json() -> None:
    parsed = parse_response('```json\n{"state_patch": {, "action": {}}\n```')
    assert parsed.parse_error is not None
    assert parsed.action == {"tool": "noop"}


def test_parse_response_defaults_missing_action_to_noop() -> None:
    parsed = parse_response('```json\n{"state_patch": {"k": 1}}\n```')
    assert parsed.action == {"tool": "noop"}
    assert parsed.state_patch == {"k": 1}


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeCompletions:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def create(self, *, model: str, messages: list[dict[str, str]], temperature: float = 0) -> _FakeResponse:
        self.calls.append(messages)
        content = self._replies.pop(0)
        # Fixed, deterministic token counts so tests don't depend on a tokenizer.
        return _FakeResponse(content, prompt_tokens=100, completion_tokens=20)


class _FakeChat:
    def __init__(self, replies: list[str]) -> None:
        self.completions = _FakeCompletions(replies)


class _FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.chat = _FakeChat(replies)


def _reply(patch: dict, action: dict) -> str:
    import json

    return f'reasoning\n```json\n{json.dumps({"state_patch": patch, "action": action})}\n```'


def test_run_episode_structured_state_persists_patches_across_steps() -> None:
    replies = [
        _reply({"count": 1}, {"tool": "step"}),
        _reply({"count": 2}, {"tool": "step"}),
        _reply({}, {"tool": "finish", "args": {"summary": "done"}}),
    ]
    client = _FakeClient(replies)

    def step_fn(t: int, action: dict, patch: dict) -> tuple[str, bool]:
        return f"observation for step {t}", False

    result: EpisodeResult = run_episode(
        client,
        "fake-model",
        task_spec="do a thing",
        tool_catalog="- step {}",
        initial_observation="start",
        step_fn=step_fn,
        mode="structured_state",
        max_steps=10,
    )

    assert result.stopped_reason == "finish"
    assert result.final_reply == "done"
    assert result.final_state == {"count": 2}
    assert len(result.steps) == 3
    # Bounded prompt: every call only ever sends [system, user] — 2 messages.
    assert all(len(call) == 2 for call in client.chat.completions.calls)


def test_run_episode_conversational_grows_history_each_step() -> None:
    replies = [_reply({}, {"tool": "step"}) for _ in range(3)]
    client = _FakeClient(replies)

    def step_fn(t: int, action: dict, patch: dict) -> tuple[str, bool]:
        done = t >= 2
        return f"observation for step {t}", done

    run_episode(
        client,
        "fake-model",
        task_spec="do a thing",
        tool_catalog="- step {}",
        initial_observation="start",
        step_fn=step_fn,
        mode="conversational",
        max_steps=10,
    )

    call_lengths = [len(call) for call in client.chat.completions.calls]
    # system + growing (user, assistant, user, assistant, ...) pairs.
    assert call_lengths == sorted(call_lengths)
    assert call_lengths[0] < call_lengths[-1]


def test_run_episode_respects_max_steps_without_finish() -> None:
    replies = [_reply({}, {"tool": "step"}) for _ in range(5)]
    client = _FakeClient(replies)

    def step_fn(t: int, action: dict, patch: dict) -> tuple[str, bool]:
        return "obs", False

    result = run_episode(
        client,
        "fake-model",
        task_spec="t",
        tool_catalog="c",
        initial_observation="start",
        step_fn=step_fn,
        mode="structured_state",
        max_steps=5,
    )
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 5


def test_step_record_and_episode_result_token_totals() -> None:
    steps = [
        StepRecord(0, prompt_tokens=10, completion_tokens=1, prompt_chars=5, action={}, state_patch=None, observation_preview=""),
        StepRecord(1, prompt_tokens=12, completion_tokens=2, prompt_chars=5, action={}, state_patch=None, observation_preview=""),
    ]
    result = EpisodeResult(mode="structured_state", steps=steps)
    assert result.total_prompt_tokens == 22
    assert result.total_completion_tokens == 3
    assert result.total_tokens == 25
