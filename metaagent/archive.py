"""Append-only JSONL archive of every AgentSpec ever tried, with its
scores and failure modes, plus cheap similarity retrieval.

This is what the architect's RETRIEVE step reads and what the improve
stage appends to after every eval run. It also has to power the
headline experiment: a COLD archive (empty) vs a WARM archive (seeded
from other domains) run through the same goal, so `reset`, `snapshot`
and `from_snapshot` exist specifically to make that A/B repeatable.

No embeddings, no network calls: retrieval is lexical overlap on the
goal text plus tool-name overlap. That's enough to find "similar past
attempts" for prompting, and it keeps the archive usable fully offline.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

DEFAULT_ARCHIVE_PATH = Path("archive.jsonl")


@dataclass
class ArchiveEntry:
    id: str
    goal: str
    domain: str
    spec: dict[str, Any]
    scores: dict[str, float]
    failure_modes: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchiveEntry":
        return cls(
            id=d["id"],
            goal=d["goal"],
            domain=d.get("domain", ""),
            spec=d["spec"],
            scores=d.get("scores", {}),
            failure_modes=d.get("failure_modes", []),
            timestamp=d.get("timestamp", 0.0),
        )


ToolLike = Union[str, dict[str, Any]]


def _tool_names(tools: list[ToolLike]) -> set[str]:
    names = set()
    for t in tools:
        if isinstance(t, str):
            names.add(t)
        elif isinstance(t, dict) and "name" in t:
            names.add(t["name"])
    return names


def _entry_tool_names(entry: ArchiveEntry) -> set[str]:
    names: set[str] = set()
    for role in entry.spec.get("roles", []):
        names.update(role.get("tools", []))
    return names


def _similarity(goal: str, tool_names: set[str], entry: ArchiveEntry) -> float:
    goal_tokens = set(goal.lower().split())
    entry_tokens = set(entry.goal.lower().split())
    goal_union = goal_tokens | entry_tokens
    goal_overlap = len(goal_tokens & entry_tokens) / len(goal_union) if goal_union else 0.0

    entry_tools = _entry_tool_names(entry)
    tool_union = tool_names | entry_tools
    tool_overlap = len(tool_names & entry_tools) / len(tool_union) if tool_union else 0.0

    return 0.6 * goal_overlap + 0.4 * tool_overlap


class Archive:
    """Append-only JSONL store. Safe to point at a path that doesn't exist
    yet -- that's the cold-start case."""

    def __init__(self, path: Union[str, Path] = DEFAULT_ARCHIVE_PATH):
        self.path = Path(path)

    def add(
        self,
        *,
        goal: str,
        domain: str,
        spec: dict[str, Any],
        scores: dict[str, float],
        failure_modes: Optional[list[str]] = None,
        entry_id: Optional[str] = None,
    ) -> ArchiveEntry:
        entry = ArchiveEntry(
            id=entry_id or str(uuid.uuid4()),
            goal=goal,
            domain=domain,
            spec=spec,
            scores=scores,
            failure_modes=failure_modes or [],
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(entry.to_json() + "\n")
        return entry

    def load(self) -> list[ArchiveEntry]:
        if not self.path.exists():
            return []
        entries = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(ArchiveEntry.from_dict(json.loads(line)))
        return entries

    def reset(self) -> None:
        """Truncate the archive back to cold-empty."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def snapshot(self, dest: Union[str, Path]) -> Path:
        """Copy the current archive state to `dest` for later restore."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            shutil.copyfile(self.path, dest)
        else:
            dest.write_text("")
        return dest

    @classmethod
    def from_snapshot(cls, src: Union[str, Path], path: Union[str, Path]) -> "Archive":
        """Restore an archive at `path` from a previously taken `snapshot`."""
        src = Path(src)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)
        return cls(path)

    def retrieve_similar(
        self,
        goal: str,
        tools: Optional[list[ToolLike]] = None,
        k: int = 3,
    ) -> list[ArchiveEntry]:
        entries = self.load()
        if not entries:
            return []
        tool_names = _tool_names(tools or [])
        scored = sorted(
            entries,
            key=lambda e: _similarity(goal, tool_names, e),
            reverse=True,
        )
        return scored[:k]
