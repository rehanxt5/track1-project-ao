from metaagent.archive import Archive
from tests.fakes import valid_spec_dict


def test_cold_archive_is_empty(tmp_path):
    archive = Archive(tmp_path / "cold.jsonl")
    assert archive.load() == []
    assert archive.retrieve_similar("do anything", []) == []


def test_add_and_load(tmp_path):
    archive = Archive(tmp_path / "archive.jsonl")
    archive.add(
        goal="Book a flight",
        domain="travel",
        spec=valid_spec_dict(),
        scores={"accuracy": 0.8},
        failure_modes=["timed out waiting for tool"],
    )
    archive.add(
        goal="Cancel a flight",
        domain="travel",
        spec=valid_spec_dict(),
        scores={"accuracy": 0.5},
    )
    entries = archive.load()
    assert len(entries) == 2
    assert entries[0].goal == "Book a flight"
    assert entries[0].failure_modes == ["timed out waiting for tool"]
    assert entries[1].failure_modes == []


def test_reset_clears_entries(tmp_path):
    archive = Archive(tmp_path / "archive.jsonl")
    archive.add(goal="g", domain="d", spec=valid_spec_dict(), scores={})
    assert len(archive.load()) == 1
    archive.reset()
    assert archive.load() == []


def test_reset_on_nonexistent_file_is_safe(tmp_path):
    archive = Archive(tmp_path / "never_created.jsonl")
    archive.reset()
    assert archive.load() == []


def test_snapshot_and_restore(tmp_path):
    warm = Archive(tmp_path / "warm.jsonl")
    warm.add(goal="seed goal", domain="finance", spec=valid_spec_dict(), scores={"accuracy": 0.9})
    snapshot_path = tmp_path / "warm_snapshot.jsonl"
    warm.snapshot(snapshot_path)

    # Mutate the live archive after taking the snapshot.
    warm.add(goal="second goal", domain="finance", spec=valid_spec_dict(), scores={})
    assert len(warm.load()) == 2

    restored = Archive.from_snapshot(snapshot_path, tmp_path / "restored.jsonl")
    entries = restored.load()
    assert len(entries) == 1
    assert entries[0].goal == "seed goal"


def test_retrieve_similar_ranks_goal_and_tool_overlap(tmp_path):
    archive = Archive(tmp_path / "archive.jsonl")
    archive.add(
        goal="Summarize quarterly earnings reports",
        domain="finance",
        spec=valid_spec_dict(tools=["search"]),
        scores={},
        entry_id="finance-1",
    )
    archive.add(
        goal="Book a table at a restaurant",
        domain="dining",
        spec=valid_spec_dict(tools=["calculator"]),
        scores={},
        entry_id="dining-1",
    )
    results = archive.retrieve_similar(
        "Summarize quarterly earnings for a company", tools=["search"], k=2
    )
    assert results[0].id == "finance-1"


def test_retrieve_similar_respects_k(tmp_path):
    archive = Archive(tmp_path / "archive.jsonl")
    for i in range(5):
        archive.add(goal=f"goal {i}", domain="d", spec=valid_spec_dict(), scores={})
    results = archive.retrieve_similar("goal 0", k=2)
    assert len(results) == 2
