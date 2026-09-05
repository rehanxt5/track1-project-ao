from metaagent.spec.models import AgentSpec
from metaagent.spec.verify import verify_spec
from tests.fakes import SAMPLE_TOOLS, valid_spec_dict


def test_valid_spec_passes():
    spec = AgentSpec.model_validate(valid_spec_dict())
    result = verify_spec(spec, SAMPLE_TOOLS)
    assert result.ok
    assert result.errors == []


def test_rejects_unknown_tool():
    data = valid_spec_dict(tools=["search", "time_machine"])
    spec = AgentSpec.model_validate(data)
    result = verify_spec(spec, SAMPLE_TOOLS)
    assert not result.ok
    assert any(e.code == "unknown_tool" for e in result.errors)


def test_rejects_whitespace_only_prompt():
    data = valid_spec_dict()
    data["roles"][0]["system_prompt"] = "   "
    spec = AgentSpec.model_validate(data)  # passes pydantic's min_length
    result = verify_spec(spec, SAMPLE_TOOLS)
    assert not result.ok
    assert any(e.code == "empty_prompt" for e in result.errors)


def test_rejects_invalid_entry_role():
    data = valid_spec_dict()
    data["orchestration"]["entry_role"] = "ghost"
    spec = AgentSpec.model_validate(data)
    result = verify_spec(spec, SAMPLE_TOOLS)
    assert not result.ok
    assert any(e.code == "invalid_entry_role" for e in result.errors)


def test_rejects_step_referencing_unknown_role():
    data = valid_spec_dict()
    data["orchestration"]["steps"] = [{"role": "ghost", "next": []}]
    spec = AgentSpec.model_validate(data)
    result = verify_spec(spec, SAMPLE_TOOLS)
    assert not result.ok
    assert any(e.code == "unknown_role_in_step" for e in result.errors)


def test_rejects_orphaned_role():
    data = valid_spec_dict()
    data["roles"].append(
        {
            "name": "unreachable",
            "system_prompt": "Never called.",
            "tools": [],
        }
    )
    # entry_role is "solver"; no step ever hands off to "unreachable".
    data["orchestration"]["steps"] = [{"role": "solver", "next": []}]
    spec = AgentSpec.model_validate(data)
    result = verify_spec(spec, SAMPLE_TOOLS)
    assert not result.ok
    assert any(e.code == "orphaned_role" for e in result.errors)


def test_rejects_role_max_tokens_exceeding_budget():
    data = valid_spec_dict()
    data["orchestration"]["budget_tokens"] = 100
    data["roles"][0]["max_tokens"] = 1024
    spec = AgentSpec.model_validate(data)
    result = verify_spec(spec, SAMPLE_TOOLS)
    assert not result.ok
    assert any(e.code == "budget_too_small" for e in result.errors)


def test_reachable_multi_role_graph_passes():
    data = valid_spec_dict()
    data["roles"][0]["name"] = "planner"
    data["orchestration"]["entry_role"] = "planner"
    data["roles"].append(
        {
            "name": "executor",
            "system_prompt": "Execute the plan.",
            "tools": ["search"],
        }
    )
    data["orchestration"]["topology"] = "supervisor_workers"
    data["orchestration"]["steps"] = [
        {"role": "planner", "next": ["executor"]},
        {"role": "executor", "next": []},
    ]
    spec = AgentSpec.model_validate(data)
    result = verify_spec(spec, SAMPLE_TOOLS)
    assert result.ok
