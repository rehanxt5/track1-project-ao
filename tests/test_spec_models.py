import pytest
from pydantic import ValidationError

from metaagent.spec.models import AgentSpec
from tests.fakes import valid_spec_dict


def test_round_trip():
    spec = AgentSpec.model_validate(valid_spec_dict())
    dumped = spec.model_dump_json()
    restored = AgentSpec.model_validate_json(dumped)
    assert restored == spec


def test_minimal_valid_spec_has_expected_shape():
    spec = AgentSpec.model_validate(valid_spec_dict())
    assert spec.version == "1.0"
    assert spec.design_philosophy.value == "react"
    assert spec.orchestration.topology.value == "single_agent"
    assert len(spec.roles) == 1
    assert spec.roles[0].tools == ["search", "calculator"]


def test_blank_system_prompt_rejected():
    data = valid_spec_dict()
    data["roles"][0]["system_prompt"] = ""
    with pytest.raises(ValidationError):
        AgentSpec.model_validate(data)


def test_critic_enabled_requires_prompt():
    data = valid_spec_dict()
    data["roles"][0]["critic"] = {"enabled": True, "system_prompt": ""}
    with pytest.raises(ValidationError):
        AgentSpec.model_validate(data)


def test_critic_enabled_with_prompt_is_valid():
    data = valid_spec_dict()
    data["roles"][0]["critic"] = {"enabled": True, "system_prompt": "Review the answer."}
    spec = AgentSpec.model_validate(data)
    assert spec.roles[0].critic.enabled is True


def test_max_steps_out_of_bounds_rejected():
    data = valid_spec_dict()
    data["orchestration"]["max_steps"] = 0
    with pytest.raises(ValidationError):
        AgentSpec.model_validate(data)

    data["orchestration"]["max_steps"] = 201
    with pytest.raises(ValidationError):
        AgentSpec.model_validate(data)


def test_duplicate_role_names_rejected():
    data = valid_spec_dict()
    data["roles"].append(dict(data["roles"][0]))
    with pytest.raises(ValidationError):
        AgentSpec.model_validate(data)


def test_extra_fields_forbidden():
    data = valid_spec_dict()
    data["unexpected_field"] = "surprise"
    with pytest.raises(ValidationError):
        AgentSpec.model_validate(data)


def test_at_least_one_role_required():
    data = valid_spec_dict()
    data["roles"] = []
    with pytest.raises(ValidationError):
        AgentSpec.model_validate(data)
