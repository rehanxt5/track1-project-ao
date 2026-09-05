from metaagent.analyzer import Diagnosis, FailureMode
from metaagent.optimizer import (
    REASONING_OFF_MARKER,
    STRICT_JSON_DIRECTIVE,
    optimize,
)
from metaagent.spec.models import AgentSpec, MemoryKind
from metaagent.spec.verify import verify_spec
from tests.fakes import SAMPLE_TOOLS, valid_spec_dict


def _spec(**kwargs) -> AgentSpec:
    return AgentSpec.model_validate(valid_spec_dict(**kwargs))


def _diagnosis(dominant_mode, **overrides) -> Diagnosis:
    defaults = dict(
        domain="d", split="dev", total_runs=3, failed_runs=3,
        distribution={dominant_mode: 3} if dominant_mode else {},
        dominant_mode=dominant_mode, instances=[], interpretation="synthetic diagnosis for tests",
        mean_score=0.2, pass_rate=0.0, mean_output_tokens=100.0, mean_latency_ms=500.0, mean_steps=1.0,
    )
    defaults.update(overrides)
    return Diagnosis(**defaults)


def test_optimize_always_returns_a_verifiable_spec_for_every_failure_mode():
    spec = _spec()
    for mode in list(FailureMode) + [None]:
        diagnosis = _diagnosis(mode.value if mode else None)
        result = optimize(spec, diagnosis, SAMPLE_TOOLS)
        verification = verify_spec(result.spec, SAMPLE_TOOLS)
        assert verification.ok, f"mode={mode}: {verification.errors}"
        assert result.mutations, f"mode={mode}: expected at least one recorded mutation"
        assert result.rationale


def test_optimize_never_mutates_the_input_spec_in_place():
    spec = _spec()
    original_prompt = spec.roles[0].system_prompt
    diagnosis = _diagnosis(FailureMode.MALFORMED_OUTPUT.value)
    optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert spec.roles[0].system_prompt == original_prompt


def test_malformed_output_adds_strict_json_directive_to_prompt():
    spec = _spec()
    diagnosis = _diagnosis(FailureMode.MALFORMED_OUTPUT.value)
    result = optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert STRICT_JSON_DIRECTIVE.strip() in result.spec.roles[0].system_prompt
    assert result.mutations[0].axis == "prompt"
    assert "malformed_output" in result.rationale


def test_malformed_output_is_idempotent_second_pass_falls_back():
    spec = _spec()
    diagnosis = _diagnosis(FailureMode.MALFORMED_OUTPUT.value)
    first = optimize(spec, diagnosis, SAMPLE_TOOLS)
    second = optimize(first.spec, diagnosis, SAMPLE_TOOLS)
    # directive already present -> falls through to the safe fallback mutation
    assert second.mutations[0].axis == "orchestration"
    assert second.spec.orchestration.max_steps == first.spec.orchestration.max_steps + 1


def test_wrong_tool_chosen_grants_an_unused_available_tool():
    spec = _spec(tools=["search"])
    diagnosis = _diagnosis(FailureMode.WRONG_TOOL_CHOSEN.value)
    result = optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert result.mutations[0].axis == "tools"
    assert "calculator" in result.spec.roles[0].tools
    assert "search" in result.spec.roles[0].tools


def test_wrong_tool_chosen_falls_back_to_prompt_when_all_tools_already_granted():
    spec = _spec(tools=["search", "calculator"])  # every SAMPLE_TOOLS entry already granted
    diagnosis = _diagnosis(FailureMode.WRONG_TOOL_CHOSEN.value)
    result = optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert result.mutations[0].axis == "prompt"


def test_tool_call_errored_increases_retry_budget():
    spec = _spec()
    assert spec.roles[0].retry.max_retries == 1
    diagnosis = _diagnosis(FailureMode.TOOL_CALL_ERRORED.value)
    result = optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert result.mutations[0].axis == "orchestration"
    assert result.spec.roles[0].retry.max_retries == 2
    assert result.spec.roles[0].retry.backoff == "exponential"


def test_missing_context_upgrades_memory_kind():
    spec = _spec()
    assert spec.roles[0].memory.kind == MemoryKind.SCRATCHPAD
    diagnosis = _diagnosis(FailureMode.MISSING_CONTEXT.value)
    result = optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert result.mutations[0].axis == "memory"
    assert result.spec.roles[0].memory.kind == MemoryKind.SUMMARY_BUFFER


def test_instruction_ignored_adds_ordering_directive():
    spec = _spec()
    diagnosis = _diagnosis(FailureMode.INSTRUCTION_IGNORED.value)
    result = optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert result.mutations[0].axis == "prompt"
    assert "order" in result.spec.roles[0].system_prompt.lower()


def test_low_quality_output_enables_critic():
    spec = _spec()
    assert spec.roles[0].critic is None
    diagnosis = _diagnosis(FailureMode.LOW_QUALITY_OUTPUT.value)
    result = optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert result.spec.roles[0].critic is not None
    assert result.spec.roles[0].critic.enabled is True
    assert result.spec.roles[0].critic.system_prompt


def test_unclassified_falls_back_to_safe_step_budget_nudge():
    spec = _spec()
    before = spec.orchestration.max_steps
    diagnosis = _diagnosis(None)
    result = optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert result.spec.orchestration.max_steps == before + 1


def test_reasoning_budget_exhausted_turns_reasoning_off_and_shrinks_tokens():
    spec = _spec()
    before_tokens = spec.roles[0].max_tokens
    diagnosis = _diagnosis(
        FailureMode.REASONING_BUDGET_EXHAUSTED.value, mean_output_tokens=900.0, mean_latency_ms=4000.0,
    )
    result = optimize(spec, diagnosis, SAMPLE_TOOLS)
    assert result.mutations[0].axis == "reasoning"
    assert REASONING_OFF_MARKER in result.spec.roles[0].system_prompt
    assert result.spec.roles[0].max_tokens < before_tokens
    assert "70x" in result.rationale


def test_reasoning_toggles_back_on_when_still_failing_with_reasoning_off():
    spec = _spec()
    off_diagnosis = _diagnosis(FailureMode.REASONING_BUDGET_EXHAUSTED.value)
    turned_off = optimize(spec, off_diagnosis, SAMPLE_TOOLS)
    assert REASONING_OFF_MARKER in turned_off.spec.roles[0].system_prompt
    shrunk_tokens = turned_off.spec.roles[0].max_tokens

    low_quality_diagnosis = _diagnosis(FailureMode.LOW_QUALITY_OUTPUT.value)
    turned_on = optimize(turned_off.spec, low_quality_diagnosis, SAMPLE_TOOLS)
    assert REASONING_OFF_MARKER not in turned_on.spec.roles[0].system_prompt
    assert turned_on.spec.roles[0].max_tokens > shrunk_tokens
