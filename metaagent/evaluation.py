"""
Shared evaluation interface for the Track 1 evaluation substrate.

===========================================================================
DEV / TEST SPLIT -- READ THIS BEFORE TOUCHING load_tasks
===========================================================================
Track 1 is judged on how agents handle tasks they have never seen. The
optimizer loop may read and tune against domains/<name>/tasks/dev/*.json.
domains/<name>/tasks/test/*.json is HELD OUT: nothing that feeds back into
the optimizer -- prompts, few-shot examples, tool tweaks, "retry until the
grader is happy" loops -- may read test-split tasks or their gold labels.
Every score reported in the final report MUST come from split="test".

load_tasks() below takes `split` as a REQUIRED argument with NO default,
on purpose. A default of "dev" is exactly the kind of mistake that lets
contamination happen silently -- someone forgets the argument, the loop
tunes on test data, and every number in the report becomes suspect. Do
not add one.
===========================================================================
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"
VALID_SPLITS = ("dev", "test")


@dataclass
class Score:
    score: float          # 0.0 to 1.0
    passed: bool
    details: dict         # per-field or per-check breakdown, used by the failure analyzer


class Evaluator(Protocol):
    def evaluate(self, task: dict, output: str) -> Score: ...


def _domain_dir(domain: str) -> Path:
    d = DOMAINS_DIR / domain
    if not d.is_dir():
        raise ValueError(f"unknown domain {domain!r}; looked for a directory at {d}")
    return d


def load_tasks(domain: str, split: str) -> list[dict]:
    """Load tasks for one domain/split. `split` has no default -- see module docstring."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS!r}, got {split!r}")

    task_dir = _domain_dir(domain) / "tasks" / split
    if not task_dir.is_dir():
        raise ValueError(f"no {split!r} split found for domain {domain!r} at {task_dir}")

    tasks = []
    for path in sorted(task_dir.glob("*.json")):
        with path.open() as f:
            task = json.load(f)
        task.setdefault("id", path.stem)
        tasks.append(task)

    if not tasks:
        raise ValueError(f"no tasks found for domain={domain!r} split={split!r} in {task_dir}")
    return tasks


def load_evaluator(domain: str) -> Evaluator:
    """Load the Evaluator instance from domains/<domain>/evaluator.py (module-level EVALUATOR)."""
    path = _domain_dir(domain) / "evaluator.py"
    spec = importlib.util.spec_from_file_location(f"domains.{domain}.evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EVALUATOR


def load_tools(domain: str) -> list[dict]:
    """Load the declarative tool list (name/description/JSON-schema params) for a domain."""
    path = _domain_dir(domain) / "tools.json"
    with path.open() as f:
        return json.load(f)
