"""Unit tests for ChaosPlugin (tool hooks and model hooks)."""

import copy
import json
import logging
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from strands import Agent
from strands.hooks import BeforeModelCallEvent
from strands.models.model import Model

from strands_evals.chaos import ChaosCase, ChaosPlugin
from strands_evals.chaos._context import _current_chaos_case
from strands_evals.chaos.effects import (
    Confabulation,
    EmptyResponse,
    FullRefusal,
    MalformedJson,
    NetworkError,
    SuccessFraming,
    Timeout,
    TruncateFields,
)
from strands_evals.chaos.plugin import _CHAOS_STATE_KEY, _MALFORMED_OUTPUT_APPLIED


@pytest.fixture
def chaos_plugin():
    return ChaosPlugin()


@pytest.fixture
def before_event():
    """Create a mock BeforeToolCallEvent."""
    event = MagicMock()
    event.tool_use = {"name": "search_tool"}
    event.cancel_tool = None
    return event


@pytest.fixture
def after_event():
    """Create a mock AfterToolCallEvent with list content."""
    event = MagicMock()
    event.tool_use = {"name": "search_tool"}
    event.result = {
        "content": [{"text": json.dumps({"title": "Long Title Here", "count": 42})}],
        "status": "success",
        "toolUseId": "tool-123",
    }
    return event


@pytest.fixture
def activate_case():
    """Factory that activates a ChaosCase for the duration of the test.

    Accepts either a full ChaosCase or a list of model effects, which is wrapped as
    effects={"model_effects": {"*": effects}}. The ContextVar is restored to its
    pre-test state on teardown.
    """
    tokens = []

    def _activate(effects_or_case: list | ChaosCase) -> ChaosCase:
        if isinstance(effects_or_case, ChaosCase):
            case = effects_or_case
        else:
            case = ChaosCase(
                name="test_case",
                input="test input",
                effects={"model_effects": {"*": effects_or_case}},
            )
        token = _current_chaos_case.set(case)
        if not tokens:
            tokens.append(token)
        return case

    yield _activate
    if tokens:
        _current_chaos_case.reset(tokens[0])


@pytest.fixture
def message_added_event():
    """Factory for a mock MessageAddedEvent."""

    def _make(message: dict, dynamic_tools: dict | None = None) -> MagicMock:
        """Build the event.

        Args:
            message: The message dict.
            dynamic_tools: Optional dict of dynamic tool names -> tools (structured-output tools).
                If None, defaults to empty dict (no structured-output tools registered).
        """
        event = MagicMock()
        event.message = message
        event.agent.tool_registry.dynamic_tools = dynamic_tools or {}
        return event

    return _make


def _final_assistant_message(text: str = "The answer is 42.") -> dict:
    """An end_turn assistant message with text content only (no toolUse)."""
    return {
        "role": "assistant",
        "content": [{"text": text}],
    }


def _tooluse_assistant_message() -> dict:
    """A tool_use assistant message containing a toolUse block."""
    return {
        "role": "assistant",
        "content": [
            {"text": "Let me search for that."},
            {"toolUse": {"toolUseId": "tu_1", "name": "search", "input": {"query": "test"}}},
        ],
    }


def _user_message() -> dict:
    """A user message."""
    return {
        "role": "user",
        "content": [{"text": "Hello, what is 2+2?"}],
    }


def _tool_result_message() -> dict:
    """A tool result message."""
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "tu_1", "status": "success", "content": [{"text": "4"}]}}],
    }


class _StructuredOutput(BaseModel):
    """Structured output model for the agent-loop regression."""

    answer: str


class _ScriptedModel(Model):
    """Emits one valid structured-output toolUse per call and counts calls.

    The chaos plugin supplies the failure, so the model itself never needs to
    produce an invalid payload.
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.call_count = 0

    def get_config(self):
        return {}

    def update_config(self, **model_config):
        pass

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": output_model(answer="direct")}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.call_count += 1
        yield {"messageStart": {"role": "assistant"}}
        yield {
            "contentBlockStart": {"start": {"toolUse": {"name": self.tool_name, "toolUseId": f"tu_{self.call_count}"}}}
        }
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"answer": "ok"}'}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 1},
            }
        }


class TestChaosPluginBeforeToolCall:
    """Tests for the before_tool_call hook."""

    def test_no_case_active_does_nothing(self, chaos_plugin, before_event):
        token = _current_chaos_case.set(None)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool is None
        finally:
            _current_chaos_case.reset(token)

    def test_case_without_matching_tool_does_nothing(self, chaos_plugin, before_event):
        case = ChaosCase(
            name="other_tool_fails",
            input="test",
            effects={"tool_effects": {"other_tool": [Timeout()]}},
        )
        token = _current_chaos_case.set(case)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool is None
        finally:
            _current_chaos_case.reset(token)

    def test_pre_hook_effect_cancels_tool(self, chaos_plugin, before_event):
        case = ChaosCase(
            name="search_timeout",
            input="test",
            effects={"tool_effects": {"search_tool": [Timeout()]}},
        )
        token = _current_chaos_case.set(case)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool == "Tool call timed out"
        finally:
            _current_chaos_case.reset(token)

    def test_post_hook_effect_does_not_cancel_tool(self, chaos_plugin, before_event):
        case = ChaosCase(
            name="search_truncated",
            input="test",
            effects={"tool_effects": {"search_tool": [TruncateFields(max_length=5)]}},
        )
        token = _current_chaos_case.set(case)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool is None
        finally:
            _current_chaos_case.reset(token)

    def test_multiple_pre_hook_effects(self, chaos_plugin, before_event):
        """Multiple effects per tool should be rejected."""
        with pytest.raises(ValueError, match="only 1 is allowed"):
            ChaosCase(
                name="multi_pre",
                input="test",
                effects={
                    "tool_effects": {
                        "search_tool": [
                            Timeout(),
                            NetworkError(),
                        ]
                    }
                },
            )


class TestChaosPluginAfterToolCall:
    """Tests for the after_tool_call hook."""

    def test_no_case_active_does_nothing(self, chaos_plugin, after_event):
        token = _current_chaos_case.set(None)
        try:
            original_content = after_event.result["content"][0]["text"]
            chaos_plugin.after_tool_call(after_event)
            assert after_event.result["content"][0]["text"] == original_content
        finally:
            _current_chaos_case.reset(token)

    def test_case_without_matching_tool_does_nothing(self, chaos_plugin, after_event):
        case = ChaosCase(
            name="other_tool",
            input="test",
            effects={"tool_effects": {"other_tool": [TruncateFields(max_length=3)]}},
        )
        token = _current_chaos_case.set(case)
        try:
            original_content = after_event.result["content"][0]["text"]
            chaos_plugin.after_tool_call(after_event)
            assert after_event.result["content"][0]["text"] == original_content
        finally:
            _current_chaos_case.reset(token)

    def test_post_hook_corrupts_json_text_blocks(self, chaos_plugin, after_event):
        case = ChaosCase(
            name="truncate",
            input="test",
            effects={"tool_effects": {"search_tool": [TruncateFields(max_length=3)]}},
        )
        token = _current_chaos_case.set(case)
        try:
            chaos_plugin.after_tool_call(after_event)
            corrupted = json.loads(after_event.result["content"][0]["text"])
            assert corrupted["title"] == "Lon"
            assert corrupted["count"] == 42  # non-string preserved
        finally:
            _current_chaos_case.reset(token)

    def test_pre_hook_effect_ignored_in_after_hook(self, chaos_plugin, after_event):
        case = ChaosCase(
            name="pre_only",
            input="test",
            effects={"tool_effects": {"search_tool": [Timeout()]}},
        )
        token = _current_chaos_case.set(case)
        try:
            original_content = after_event.result["content"][0]["text"]
            chaos_plugin.after_tool_call(after_event)
            assert after_event.result["content"][0]["text"] == original_content
        finally:
            _current_chaos_case.reset(token)

    def test_none_result_is_skipped(self, chaos_plugin):
        event = MagicMock()
        event.tool_use = {"name": "search_tool"}
        event.result = None

        case = ChaosCase(
            name="truncate",
            input="test",
            effects={"tool_effects": {"search_tool": [TruncateFields(max_length=3)]}},
        )
        token = _current_chaos_case.set(case)
        try:
            chaos_plugin.after_tool_call(event)  # Should not raise
        finally:
            _current_chaos_case.reset(token)

    def test_plain_text_truncation(self, chaos_plugin):
        """Test that plain (non-JSON) text blocks get truncated if effect has max_length."""
        event = MagicMock()
        event.tool_use = {"name": "search_tool"}
        event.result = {
            "content": [{"text": "This is plain text, not JSON"}],
            "status": "success",
            "toolUseId": "tool-456",
        }

        case = ChaosCase(
            name="truncate",
            input="test",
            effects={"tool_effects": {"search_tool": [TruncateFields(max_length=4)]}},
        )
        token = _current_chaos_case.set(case)
        try:
            chaos_plugin.after_tool_call(event)
            assert event.result["content"][0]["text"] == "This"
        finally:
            _current_chaos_case.reset(token)


class TestEmptyResponsePreHook:
    """EmptyResponse is a pre-hook effect — cancels model call with single space."""

    def test_empty_response_cancels_with_single_space(self, chaos_plugin, activate_case):
        """before_model_invocation sets event.cancel to ' ' (single space)."""
        activate_case([EmptyResponse()])
        event = BeforeModelCallEvent(agent=MagicMock())

        chaos_plugin.before_model_invocation(event)

        assert event.cancel == " "

    def test_empty_response_model_not_called(self, chaos_plugin, activate_case, message_added_event):
        """When EmptyResponse fires as pre-hook, post-hook does not apply effects."""
        activate_case([EmptyResponse()])

        # Pre-hook fires
        pre_event = BeforeModelCallEvent(agent=MagicMock())
        chaos_plugin.before_model_invocation(pre_event)
        assert pre_event.cancel == " "

        # SDK builds cancel message, MessageAddedEvent fires
        cancel_message = {"role": "assistant", "content": [{"text": " "}]}
        post_event = message_added_event(cancel_message)
        chaos_plugin.after_model_invocation(post_event)

        # Content should be unchanged (pre effects skip post processing)
        assert cancel_message["content"] == [{"text": " "}]


class TestFullRefusalPreHook:
    """FullRefusal is a pre-hook effect — cancels model call with refusal text."""

    def test_full_refusal_cancels_model_call(self, chaos_plugin, activate_case):
        """before_model_invocation sets event.cancel to a refusal template."""
        activate_case([FullRefusal()])
        event = BeforeModelCallEvent(agent=MagicMock())

        chaos_plugin.before_model_invocation(event)

        assert event.cancel in FullRefusal._REFUSAL_TEMPLATES

    def test_full_refusal_produces_single_turn(self, chaos_plugin, activate_case, message_added_event):
        """FullRefusal cancels model call, SDK builds cancel message, run ends."""
        activate_case([FullRefusal()])

        # Step 1: before_model_invocation fires
        pre_event = BeforeModelCallEvent(agent=MagicMock())
        chaos_plugin.before_model_invocation(pre_event)
        cancel_text = pre_event.cancel
        assert cancel_text in FullRefusal._REFUSAL_TEMPLATES

        # Step 2: SDK builds the cancel message and fires MessageAddedEvent
        cancel_message = {"role": "assistant", "content": [{"text": cancel_text}]}
        post_event = message_added_event(cancel_message)
        chaos_plugin.after_model_invocation(post_event)

        # Step 3: verify the cancel message is unchanged (not double-corrupted)
        assert cancel_message["content"] == [{"text": cancel_text}]
        assert len(cancel_message["content"]) == 1


class TestStructuredOutputFailureInjection:
    """MalformedJson injects one structured-output parse failure per invocation."""

    _EXPECTED_MESSAGE = "Structured output was malformed and could not be parsed. Please produce a corrected response."

    def _tool_event(self, tool_type, invocation_state=None):
        event = MagicMock()
        event.tool_use = {"name": "MyModel"}
        event.selected_tool = MagicMock(tool_type=tool_type)
        event.invocation_state = {} if invocation_state is None else invocation_state
        event.cancel_tool = False
        return event

    def test_structured_output_attempt_is_failed(self, chaos_plugin, activate_case, caplog):
        """The first structured-output attempt is cancelled with the parse-failure message."""
        activate_case([MalformedJson()])
        event = self._tool_event("structured_output")

        with caplog.at_level(logging.INFO):
            chaos_plugin.before_tool_call(event)

        assert event.cancel_tool == self._EXPECTED_MESSAGE
        assert "injected structured output parse failure" in caplog.text

    def test_injection_is_one_shot_per_invocation(self, chaos_plugin, activate_case):
        """A second attempt in the same invocation passes through so the agent can recover."""
        activate_case([MalformedJson()])
        invocation_state = {}

        first = self._tool_event("structured_output", invocation_state)
        chaos_plugin.before_tool_call(first)
        assert first.cancel_tool == self._EXPECTED_MESSAGE

        second = self._tool_event("structured_output", invocation_state)
        chaos_plugin.before_tool_call(second)
        assert second.cancel_tool is False

    def test_ordinary_tool_is_unaffected(self, chaos_plugin, activate_case):
        """A non-structured-output tool is not cancelled by MalformedJson."""
        activate_case([MalformedJson()])
        event = self._tool_event("function")

        chaos_plugin.before_tool_call(event)

        assert event.cancel_tool is False
        assert _CHAOS_STATE_KEY not in event.invocation_state

    def test_no_malformed_json_configured_writes_no_state(self, chaos_plugin, activate_case):
        """Without MalformedJson the structured-output tool runs and no marker is written."""
        activate_case([Confabulation()])
        event = self._tool_event("structured_output")

        chaos_plugin.before_tool_call(event)

        assert event.cancel_tool is False
        assert _CHAOS_STATE_KEY not in event.invocation_state


class TestToolUseMessagesNeverCorrupted:
    """after_model_invocation leaves every message carrying a toolUse block untouched."""

    def test_structured_output_tooluse_untouched(self, chaos_plugin, activate_case, message_added_event, caplog):
        """A structured-output toolUse message is not corrupted and nothing is logged."""
        activate_case([MalformedJson(), SuccessFraming()])
        message = {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "so_1", "name": "MyModel", "input": {"field1": "value1"}}},
            ],
        }
        original_content = copy.deepcopy(message["content"])
        event = message_added_event(message)

        with caplog.at_level(logging.INFO):
            chaos_plugin.after_model_invocation(event)

        assert message["content"] == original_content
        assert "applied model output chaos" not in caplog.text

    def test_ordinary_tooluse_untouched(self, chaos_plugin, activate_case, message_added_event):
        """An ordinary mid-turn toolUse message is not corrupted."""
        activate_case([MalformedJson()])
        message = _tooluse_assistant_message()
        original_content = copy.deepcopy(message["content"])
        event = message_added_event(message)

        chaos_plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def test_text_only_response_still_corrupted(self, chaos_plugin, activate_case, message_added_event):
        """Final text responses remain corruptible."""
        activate_case([MalformedJson()])
        message = _final_assistant_message('{"key": "value", "nested": {"a": 1}}')
        event = message_added_event(message)

        chaos_plugin.after_model_invocation(event)

        assert message["content"][0]["text"] != '{"key": "value", "nested": {"a": 1}}'


class TestInvocationStateCleanup:
    """The per-invocation chaos marker is cleared at the invocation boundary."""

    def test_marker_and_empty_parent_are_removed(self, chaos_plugin):
        """A consumed marker is popped along with the now-empty chaos namespace."""
        invocation_state = {_CHAOS_STATE_KEY: {_MALFORMED_OUTPUT_APPLIED: True}}
        event = MagicMock()
        event.invocation_state = invocation_state

        chaos_plugin.after_invocation(event)

        assert _CHAOS_STATE_KEY not in invocation_state

    def test_parent_retained_when_other_chaos_state_remains(self, chaos_plugin):
        """Unrelated chaos state keeps the namespace alive."""
        invocation_state = {_CHAOS_STATE_KEY: {_MALFORMED_OUTPUT_APPLIED: True, "other": 1}}
        event = MagicMock()
        event.invocation_state = invocation_state

        chaos_plugin.after_invocation(event)

        assert invocation_state[_CHAOS_STATE_KEY] == {"other": 1}

    def test_absent_or_non_dict_state_is_ignored(self, chaos_plugin):
        """Missing or malformed chaos state is left untouched without raising."""
        empty = {}
        event = MagicMock()
        event.invocation_state = empty
        chaos_plugin.after_invocation(event)
        assert empty == {}

        non_dict = {_CHAOS_STATE_KEY: "unexpected"}
        event = MagicMock()
        event.invocation_state = non_dict
        chaos_plugin.after_invocation(event)
        assert non_dict == {_CHAOS_STATE_KEY: "unexpected"}


class TestStructuredOutputAgentLoop:
    """Regression through the real SDK loop: injection fires once per invocation."""

    async def test_reused_invocation_state_still_injects(self, chaos_plugin):
        """Two invocations sharing one state dict each get exactly one injected failure."""
        # The async test body runs in its own context copy, so the case is set and reset
        # inline rather than via the activate_case fixture (whose teardown runs outside it).
        case = ChaosCase(
            name="test_case",
            input="test input",
            effects={"model_effects": {"*": [MalformedJson()]}},
        )
        token = _current_chaos_case.set(case)
        try:
            model = _ScriptedModel(tool_name=_StructuredOutput.__name__)
            agent = Agent(model=model, plugins=[chaos_plugin], callback_handler=None)
            shared_state = {}

            first = await agent.invoke_async(
                "first", invocation_state=shared_state, structured_output_model=_StructuredOutput
            )
            calls_after_first = model.call_count

            second = await agent.invoke_async(
                "second", invocation_state=shared_state, structured_output_model=_StructuredOutput
            )
        finally:
            _current_chaos_case.reset(token)

        # one failed attempt plus the corrected retry, per invocation
        assert calls_after_first == 2
        assert model.call_count == 4
        # the caller still receives validated structured output both times
        assert first.structured_output == _StructuredOutput(answer="ok")
        assert second.structured_output == _StructuredOutput(answer="ok")
        # the boundary cleanup ran, so the marker cannot leak into a later invocation
        assert _CHAOS_STATE_KEY not in shared_state


class TestPostEffectsOnText:
    """Post effects (Confabulation, MalformedJson-on-text, SuccessFraming) work."""

    def test_confabulation_injects_template(self, chaos_plugin, activate_case, message_added_event):
        """Confabulation injects fabricated citations into text content."""
        activate_case([Confabulation()])
        original_text = "The weather is sunny. It is warm outside. Birds are singing."
        message = _final_assistant_message(original_text)
        event = message_added_event(message)

        chaos_plugin.after_model_invocation(event)

        result_text = message["content"][0]["text"]
        assert result_text != original_text
        assert "sunny" in result_text or "warm" in result_text

    def test_malformed_json_on_text(self, chaos_plugin, activate_case, message_added_event):
        """MalformedJson truncates JSON-like text content."""
        activate_case([MalformedJson()])
        message = _final_assistant_message('{"key": "value", "nested": {"a": 1}}')
        event = message_added_event(message)

        chaos_plugin.after_model_invocation(event)

        result_text = message["content"][0]["text"]
        assert result_text != '{"key": "value", "nested": {"a": 1}}'
        assert len(result_text) < len('{"key": "value", "nested": {"a": 1}}')

    def test_success_framing_prepends_prefix(self, chaos_plugin, activate_case, message_added_event):
        """SuccessFraming prepends a confident prefix to text content."""
        activate_case([SuccessFraming()])
        message = _final_assistant_message("Here is the result.")
        event = message_added_event(message)

        chaos_plugin.after_model_invocation(event)

        result_text = message["content"][0]["text"]
        has_prefix = any(result_text.startswith(p) for p in SuccessFraming._SUCCESS_PREFIXES)
        assert has_prefix
        assert "Here is the result." in result_text

    def test_confabulation_plus_success_framing(self, chaos_plugin, activate_case, message_added_event):
        """Confabulation + SuccessFraming compose: citation injected, then prefix prepended."""
        activate_case([Confabulation(), SuccessFraming()])
        original_text = "The weather is sunny. It is warm outside. Birds are singing."
        message = _final_assistant_message(original_text)
        event = message_added_event(message)

        chaos_plugin.after_model_invocation(event)

        result_text = message["content"][0]["text"]
        has_prefix = any(result_text.startswith(p) for p in SuccessFraming._SUCCESS_PREFIXES)
        assert has_prefix


class TestMixedPrePostCase:
    """Mixed pre+post effects: pre wins, post does NOT double-corrupt."""

    def test_full_refusal_plus_malformed_json(self, chaos_plugin, activate_case, message_added_event):
        """FullRefusal (pre) + MalformedJson (post): pre cancels, post skipped."""
        activate_case([FullRefusal(), MalformedJson()])

        pre_event = BeforeModelCallEvent(agent=MagicMock())
        chaos_plugin.before_model_invocation(pre_event)
        cancel_text = pre_event.cancel
        assert cancel_text in FullRefusal._REFUSAL_TEMPLATES

        cancel_message = {"role": "assistant", "content": [{"text": cancel_text}]}
        post_event = message_added_event(cancel_message)
        chaos_plugin.after_model_invocation(post_event)

        # Post effect (MalformedJson) should NOT have corrupted the content
        assert cancel_message["content"] == [{"text": cancel_text}]
        assert len(cancel_message["content"]) == 1

    def test_empty_response_plus_success_framing(self, chaos_plugin, activate_case, message_added_event):
        """EmptyResponse (pre) + SuccessFraming (post): pre cancels, post skipped."""
        activate_case([EmptyResponse(), SuccessFraming()])

        pre_event = BeforeModelCallEvent(agent=MagicMock())
        chaos_plugin.before_model_invocation(pre_event)
        assert pre_event.cancel == " "

        cancel_message = {"role": "assistant", "content": [{"text": " "}]}
        post_event = message_added_event(cancel_message)
        chaos_plugin.after_model_invocation(post_event)

        # SuccessFraming (post) should NOT have been applied
        assert cancel_message["content"] == [{"text": " "}]


class TestGuardRoleFiltering:
    """User and tool result messages are NOT corrupted."""

    def test_user_message_not_corrupted(self, chaos_plugin, activate_case, message_added_event):
        activate_case([Confabulation()])
        message = _user_message()
        original_content = copy.deepcopy(message["content"])
        event = message_added_event(message)

        chaos_plugin.after_model_invocation(event)

        assert message["content"] == original_content

    def test_tool_result_message_not_corrupted(self, chaos_plugin, activate_case, message_added_event):
        activate_case([Confabulation()])
        message = _tool_result_message()
        original_content = copy.deepcopy(message["content"])
        event = message_added_event(message)

        chaos_plugin.after_model_invocation(event)

        assert message["content"] == original_content


class TestPassthrough:
    """No corruption when no model_effects is set."""

    def test_no_config_passes_through(self, chaos_plugin, message_added_event):
        token = _current_chaos_case.set(None)
        try:
            message = _final_assistant_message("Hello world")
            original_content = copy.deepcopy(message["content"])
            event = message_added_event(message)

            chaos_plugin.after_model_invocation(event)

            assert message["content"] == original_content
        finally:
            _current_chaos_case.reset(token)

    def test_empty_effects_passes_through(self, chaos_plugin, activate_case, message_added_event):
        """ChaosCase with empty effects dict does not corrupt."""
        activate_case(ChaosCase(name="baseline", input="test", effects={}))
        message = _final_assistant_message("Hello world")
        original_content = copy.deepcopy(message["content"])
        event = message_added_event(message)

        chaos_plugin.after_model_invocation(event)

        assert message["content"] == original_content


class TestSelectPreModelEffect:
    """_select_pre_model_effect returns the single pre effect regardless of position."""

    def test_no_effects_returns_none(self, chaos_plugin, activate_case):
        activate_case([])
        assert chaos_plugin._select_pre_model_effect() is None

    def test_post_only_returns_none(self, chaos_plugin, activate_case):
        activate_case([MalformedJson(), SuccessFraming()])
        assert chaos_plugin._select_pre_model_effect() is None

    def test_single_pre_returned(self, chaos_plugin, activate_case):
        pre = FullRefusal()
        activate_case([pre])
        assert chaos_plugin._select_pre_model_effect() is pre

    def test_pre_after_posts_returned(self, chaos_plugin, activate_case):
        pre = EmptyResponse()
        activate_case([MalformedJson(), Confabulation(), pre])
        assert chaos_plugin._select_pre_model_effect() is pre
