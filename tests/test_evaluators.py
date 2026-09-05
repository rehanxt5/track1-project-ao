import json

import pytest

from metaagent.evaluation import load_evaluator, load_tasks, load_tools

DOMAINS = ["structured_extraction", "multi_step_tools", "sql_generation"]


# ---------------------------------------------------------------------------
# Dev/test split hygiene -- the most important property of this substrate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_split_requires_explicit_argument(domain):
    with pytest.raises(TypeError):
        load_tasks(domain)  # no default split -- must be impossible to call without one


@pytest.mark.parametrize("domain", DOMAINS)
def test_split_rejects_unknown_value(domain):
    with pytest.raises(ValueError):
        load_tasks(domain, "train")


@pytest.mark.parametrize("domain", DOMAINS)
def test_dev_and_test_are_disjoint(domain):
    dev_tasks = load_tasks(domain, "dev")
    test_tasks = load_tasks(domain, "test")

    dev_ids = {t["id"] for t in dev_tasks}
    test_ids = {t["id"] for t in test_tasks}
    assert not (dev_ids & test_ids), "dev and test task ids overlap"

    # ids alone aren't enough -- check the actual task content doesn't repeat
    # across the split (would defeat the purpose of held-out data).
    dev_payloads = {json.dumps(t, sort_keys=True) for t in dev_tasks}
    test_payloads = {json.dumps(t, sort_keys=True) for t in test_tasks}
    assert not (dev_payloads & test_payloads), "identical task appears in both dev and test"


@pytest.mark.parametrize("domain", DOMAINS)
def test_task_counts_in_range(domain):
    for split in ("dev", "test"):
        tasks = load_tasks(domain, split)
        assert 20 <= len(tasks) <= 30, f"{domain}/{split} has {len(tasks)} tasks"


@pytest.mark.parametrize("domain", DOMAINS)
def test_load_tools_returns_declarative_list(domain):
    tools = load_tools(domain)
    assert isinstance(tools, list)
    assert len(tools) >= 1
    for tool in tools:
        assert {"name", "description", "parameters"} <= tool.keys()


# ---------------------------------------------------------------------------
# structured_extraction
# ---------------------------------------------------------------------------


def test_structured_extraction_good_output_passes():
    evaluator = load_evaluator("structured_extraction")
    task = load_tasks("structured_extraction", "dev")[0]
    output = json.dumps(task["gold"])

    result = evaluator.evaluate(task, output)

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["parse_error"] is None
    assert result.details["correct_count"] == result.details["total_fields"]


def test_structured_extraction_bad_output_fails():
    evaluator = load_evaluator("structured_extraction")
    task = load_tasks("structured_extraction", "dev")[0]
    bad = dict(task["gold"])
    bad["issue_type"] = "not_a_real_issue_type"
    bad["priority"] = "not_a_real_priority"
    bad.pop("customer_name", None)

    result = evaluator.evaluate(task, json.dumps(bad))

    assert result.passed is False
    assert result.score < 1.0
    assert "customer_name" in result.details["missing_fields"]
    assert result.details["fields"]["issue_type"]["correct"] is False


def test_structured_extraction_unparseable_output_scores_zero():
    evaluator = load_evaluator("structured_extraction")
    task = load_tasks("structured_extraction", "dev")[0]

    result = evaluator.evaluate(task, "not json at all, sorry")

    assert result.score == 0.0
    assert result.passed is False
    assert result.details["parse_error"] is not None


# ---------------------------------------------------------------------------
# multi_step_tools
# ---------------------------------------------------------------------------


def test_multi_step_tools_good_output_passes():
    evaluator = load_evaluator("multi_step_tools")
    task = load_tasks("multi_step_tools", "dev")[0]
    gold = task["gold"]
    output = json.dumps({"tool_calls": gold["required_calls"], "answer": gold["answer"]})

    result = evaluator.evaluate(task, output)

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["missing_calls"] == []
    assert result.details["order_violations"] == []


def test_multi_step_tools_bad_output_fails_on_answer_and_tools():
    evaluator = load_evaluator("multi_step_tools")
    task = load_tasks("multi_step_tools", "dev")[0]
    gold = task["gold"]
    bogus_answer = "definitely wrong" if not isinstance(gold["answer"], (int, float)) else -999999
    output = json.dumps({"tool_calls": [], "answer": bogus_answer})

    result = evaluator.evaluate(task, output)

    assert result.passed is False
    assert result.details["answer_score"] == 0.0
    assert result.details["matched_required_calls"] == 0
    assert result.score < 0.5


def test_multi_step_tools_out_of_order_calls_violate_constraints():
    evaluator = load_evaluator("multi_step_tools")
    task = next(
        t for t in load_tasks("multi_step_tools", "dev") if t["gold"]["order_constraints"]
    )
    gold = task["gold"]
    reversed_calls = list(reversed(gold["required_calls"]))
    output = json.dumps({"tool_calls": reversed_calls, "answer": gold["answer"]})

    result = evaluator.evaluate(task, output)

    assert result.details["order_violations"] != []
    assert result.passed is False


# ---------------------------------------------------------------------------
# sql_generation
# ---------------------------------------------------------------------------


def test_sql_generation_good_output_passes():
    evaluator = load_evaluator("sql_generation")
    task = load_tasks("sql_generation", "dev")[0]
    output = json.dumps({"sql": task["gold"]["sql"]})

    result = evaluator.evaluate(task, output)

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["execution_error"] is None
    assert result.details["missing_rows"] == []
    assert result.details["extra_rows"] == []


def test_sql_generation_wrong_query_fails():
    evaluator = load_evaluator("sql_generation")
    task = load_tasks("sql_generation", "dev")[0]
    output = json.dumps({"sql": "SELECT 1 WHERE 1=0"})

    result = evaluator.evaluate(task, output)

    assert result.passed is False
    assert result.score < 1.0


def test_sql_generation_invalid_sql_reports_execution_error():
    evaluator = load_evaluator("sql_generation")
    task = load_tasks("sql_generation", "dev")[0]
    output = json.dumps({"sql": "SELEKT garbage FROM nowhere"})

    result = evaluator.evaluate(task, output)

    assert result.passed is False
    assert result.score == 0.0
    assert result.details["execution_error"] is not None


def test_sql_generation_rejects_non_select():
    evaluator = load_evaluator("sql_generation")
    task = load_tasks("sql_generation", "dev")[0]
    output = json.dumps({"sql": "DELETE FROM customers"})

    result = evaluator.evaluate(task, output)

    assert result.passed is False
    assert result.score == 0.0
    assert "SELECT" in result.details["execution_error"]
