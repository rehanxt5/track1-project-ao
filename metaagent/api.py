"""FastAPI backend for the Studio (Layer 1) and Optimization (Layer 2) views.

This module owns the web-facing wiring only. It never talks to LLM
providers directly (that stays inside metaagent.gateway) and never invents
new architect/archive/spec behaviour -- it calls the public entry points
those modules already expose (`architect.generate`, `Archive.*`,
`verify_spec`, `load_tools`) and reads their results for display.

GATEWAY_MODE=fake path: `metaagent.gateway`'s own fake mode only fabricates
schema-shaped JSON when the caller passes a `response_format["schema"]`,
but the architect's synthesize/repair calls only ever pass
`{"type": "json_object"}`. To keep "fake mode works end to end" true for
the Studio without touching architect.py or gateway.py, this module
injects its own deterministic `complete_fn` (see `_fake_studio_complete`)
that reads the same prompts architect.py builds and returns valid,
verify-passing AgentSpec JSON directly -- offline, no keys, no network.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from metaagent import architect
from metaagent.archive import Archive, ArchiveEntry
from metaagent.archive import _similarity as _archive_similarity  # noqa: PLC0415
from metaagent.archive import _tool_names as _archive_tool_names  # noqa: PLC0415
from metaagent.architect import PHILOSOPHY_LIBRARY
from metaagent.architect import _select_philosophies  # noqa: PLC0415
from metaagent.config import gateway_mode
from metaagent.evaluation import DOMAINS_DIR, load_tools
from metaagent.spec.verify import verify_spec

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
ARCHIVE_PATH = Path("archive.jsonl")
RESULTS_DIR = Path("results")

app = FastAPI(title="Meta-Agent Studio")


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------

def _goal_summary(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return " ".join(lines[:4])[:400]


def _list_domains() -> list[dict[str, Any]]:
    domains = []
    if not DOMAINS_DIR.is_dir():
        return domains
    for d in sorted(DOMAINS_DIR.iterdir()):
        goal_path, tools_path = d / "goal.md", d / "tools.json"
        if not (d.is_dir() and goal_path.exists() and tools_path.exists()):
            continue
        domains.append(
            {
                "id": d.name,
                "label": d.name.replace("_", " "),
                "goal_hint": _goal_summary(goal_path.read_text()),
                "tools": load_tools(d.name),
            }
        )
    return domains


@app.get("/api/domains")
async def get_domains() -> JSONResponse:
    return JSONResponse({"domains": _list_domains()})


# ---------------------------------------------------------------------------
# Studio: prompt -> candidate specs, streamed stage by stage over SSE
# ---------------------------------------------------------------------------

def _resolve_tools(domain: Optional[str], tools_json: Optional[str]) -> tuple[list[dict], Optional[str]]:
    if domain:
        try:
            return load_tools(domain), None
        except (ValueError, FileNotFoundError) as e:
            return [], str(e)
    if tools_json:
        try:
            parsed = json.loads(tools_json)
        except json.JSONDecodeError:
            return [], "custom tools must be valid JSON"
        if not isinstance(parsed, list) or not all(isinstance(t, dict) and t.get("name") for t in parsed):
            return [], "custom tools must be a list of objects with at least a 'name'"
        return parsed, None
    return [], "choose a domain or provide at least one custom tool"


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _retrieved_payload(entries: list[ArchiveEntry], goal: str, tools: list[dict]) -> list[dict]:
    tool_names = _archive_tool_names(tools)
    return [
        {
            "id": e.id,
            "goal": e.goal,
            "domain": e.domain,
            "design_philosophy": e.spec.get("design_philosophy"),
            "accuracy": e.scores.get("accuracy"),
            "similarity": round(_archive_similarity(goal, tool_names, e), 3),
            "failure_modes": e.failure_modes,
        }
        for e in entries
    ]


def _select_payload(n_seeds: int, retrieved: list[ArchiveEntry]) -> list[dict]:
    poor = {
        e.spec.get("design_philosophy")
        for e in retrieved
        if e.spec.get("design_philosophy") and e.scores.get("accuracy", 1.0) < 0.4
    }
    philosophies = _select_philosophies(n_seeds, retrieved)
    return [
        {
            "name": p.value,
            "topology": PHILOSOPHY_LIBRARY[p.value]["topology"],
            "summary": PHILOSOPHY_LIBRARY[p.value]["summary"],
            "deprioritized": p.value in poor,
        }
        for p in philosophies
    ]


def _spec_payload(spec) -> dict:
    return spec.model_dump(mode="json")


async def _generate_stream(goal: str, tools: list[dict], n_seeds: int):
    goal = goal.strip()
    if not goal:
        yield _sse({"stage": "error", "message": "goal must not be blank"})
        return

    archive = Archive(ARCHIVE_PATH)

    try:
        yield _sse({"stage": "perceive", "goal": goal, "tools": tools})
        await asyncio.sleep(0.15)

        retrieved = archive.retrieve_similar(goal, tools, k=3)
        yield _sse({"stage": "retrieve", "retrieved": _retrieved_payload(retrieved, goal, tools)})
        await asyncio.sleep(0.15)

        yield _sse({"stage": "select", "philosophies": _select_payload(n_seeds, retrieved)})
        await asyncio.sleep(0.15)

        yield _sse({"stage": "synthesize", "status": "in_progress"})
        complete_fn = _fake_studio_complete if gateway_mode() == "fake" else None
        specs = await architect.generate(
            goal, tools, n_seeds=n_seeds, archive=archive, complete_fn=complete_fn
        )

        verify_payload = []
        for spec in specs:
            result = verify_spec(spec, tools)
            verify_payload.append(
                {
                    "design_philosophy": spec.design_philosophy.value,
                    "ok": result.ok,
                    "errors": [e.message for e in result.errors],
                }
            )
        yield _sse({"stage": "verify", "results": verify_payload})
        await asyncio.sleep(0.1)

        yield _sse({"stage": "done", "specs": [_spec_payload(s) for s in specs]})
    except Exception as e:  # architect.ArchitectError, ValueError, GatewayError, ...
        yield _sse({"stage": "error", "message": str(e)})


@app.get("/api/generate/stream")
async def generate_stream(
    goal: str = Query(...),
    domain: Optional[str] = Query(None),
    tools: Optional[str] = Query(None),
    n_seeds: int = Query(3, ge=2, le=3),
) -> StreamingResponse:
    resolved_tools, err = _resolve_tools(domain, tools)

    async def _error_only(message: str):
        yield _sse({"stage": "error", "message": message})

    body = _error_only(err) if err else _generate_stream(goal, resolved_tools, n_seeds)
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Fake-mode synthesis: deterministic, always verify-passing, offline
# ---------------------------------------------------------------------------

@dataclass
class _FakeCompletion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    provider: str


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _extract_goal_and_tools(user_content: str) -> tuple[str, list[str]]:
    m = re.search(r"Goal: (.*?)\n\nAvailable tools:\n(.*?)\n\nRequired seeds:", user_content, re.DOTALL)
    if not m:
        return "", []
    goal = m.group(1).strip()
    tool_names = re.findall(r"^- ([^:\n]+):", m.group(2), re.MULTILINE)
    return goal, tool_names


def _fake_spec_for_philosophy(philosophy: str, goal: str, tools: list[str]) -> dict:
    topology = PHILOSOPHY_LIBRARY.get(philosophy, PHILOSOPHY_LIBRARY["react"])["topology"]
    goal_snippet = goal or "the stated goal"
    retry = {"max_retries": 1, "backoff": "none", "on_failure": "abort"}
    memory = {"kind": "scratchpad", "max_items": 20}

    if philosophy == "plan_execute":
        roles = [
            {
                "name": "planner",
                "system_prompt": f"Decompose the goal into an ordered list of concrete tasks: {goal_snippet}",
                "tools": [],
                "memory": memory,
                "retry": retry,
                "max_tokens": 512,
                "temperature": 0.3,
            },
            {
                "name": "executor",
                "system_prompt": (
                    f"Carry out each planned task in order, using the available tools, "
                    f"to satisfy: {goal_snippet}"
                ),
                "tools": tools,
                "memory": memory,
                "retry": retry,
                "max_tokens": 1024,
                "temperature": 0.2,
            },
        ]
        entry_role, steps = "planner", [
            {"role": "planner", "next": ["executor"]},
            {"role": "executor", "next": []},
        ]
    elif philosophy == "plan_critic_reflect":
        roles = [
            {
                "name": "planner",
                "system_prompt": f"Draft a solution for: {goal_snippet}",
                "tools": tools,
                "memory": memory,
                "retry": retry,
                "max_tokens": 1024,
                "temperature": 0.3,
            },
            {
                "name": "critic",
                "system_prompt": (
                    f"Review the draft against the goal and list concrete gaps before it "
                    f"is finalized: {goal_snippet}"
                ),
                "tools": [],
                "memory": memory,
                "retry": retry,
                "max_tokens": 512,
                "temperature": 0.2,
            },
        ]
        entry_role, steps = "planner", [
            {"role": "planner", "next": ["critic"]},
            {"role": "critic", "next": []},
        ]
    else:  # react (and unrecognized fallback)
        philosophy = "react"
        roles = [
            {
                "name": "solver",
                "system_prompt": (
                    f"Interleave Thought/Action/Observation using the available tools until "
                    f"the goal is satisfied: {goal_snippet}"
                ),
                "tools": tools,
                "memory": memory,
                "retry": retry,
                "max_tokens": 1024,
                "temperature": 0.2,
            }
        ]
        entry_role, steps = "solver", []

    return {
        "version": "1.0",
        "goal": goal or "unspecified goal",
        "design_philosophy": philosophy,
        "orchestration": {
            "topology": topology,
            "entry_role": entry_role,
            "steps": steps,
            "max_steps": 10,
            "budget_tokens": 20000,
        },
        "roles": roles,
    }


async def _fake_studio_complete(
    layer: str,
    messages: list[dict[str, str]],
    *,
    response_format: Optional[dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> _FakeCompletion:
    start = time.perf_counter()
    system = messages[0]["content"] if messages else ""
    user = messages[-1]["content"] if messages else ""

    if "VERIFY/repair" in system:
        m = re.search(r"design_philosophy=(\w+)", system)
        philosophy = m.group(1) if m else "react"
        gm = re.search(r'"goal":\s*"([^"]*)"', user)
        goal = gm.group(1) if gm else ""
        spec = _fake_spec_for_philosophy(philosophy, goal, [])
        text = json.dumps(spec)
    else:
        goal, tools = _extract_goal_and_tools(user)
        philosophies = re.findall(r"design_philosophy=(\w+)", user) or ["react"]
        specs = [_fake_spec_for_philosophy(p, goal, tools) for p in philosophies]
        text = json.dumps({"seeds": specs})

    await asyncio.sleep(0)
    latency_ms = (time.perf_counter() - start) * 1000 + 90.0
    return _FakeCompletion(
        text=text,
        input_tokens=_estimate_tokens(user),
        output_tokens=_estimate_tokens(text),
        latency_ms=latency_ms,
        model="fake-studio",
        provider="fake",
    )


# ---------------------------------------------------------------------------
# Optimization: read whatever metaagent/loop.py has appended to results/
# ---------------------------------------------------------------------------

def _get(d: Optional[dict], *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _normalize_record(r: dict) -> dict:
    dev = r.get("dev") if isinstance(r.get("dev"), dict) else None
    test = r.get("test") if isinstance(r.get("test"), dict) else None
    tokens = r.get("tokens") if isinstance(r.get("tokens"), dict) else None
    return {
        "iteration": r.get("iteration"),
        "spec_version": _get(r, "spec_version", "version"),
        "dev_mean_score": _get(dev, "mean_score") if dev else _get(r, "dev_mean_score"),
        "test_mean_score": _get(test, "mean_score") if test else _get(r, "test_mean_score"),
        "pass_rate": (_get(test, "pass_rate") if test else None)
        or (_get(dev, "pass_rate") if dev else None)
        or _get(r, "pass_rate"),
        "score_stddev": (_get(test, "score_stddev") if test else None)
        or (_get(dev, "score_stddev") if dev else None)
        or _get(r, "score_stddev"),
        "input_tokens": (_get(tokens, "input", "input_tokens") if tokens else None) or _get(r, "input_tokens"),
        "output_tokens": (_get(tokens, "output", "output_tokens") if tokens else None) or _get(r, "output_tokens"),
        "latency_ms": _get(r, "latency_ms", "latency"),
        "mutation_rationale": _get(r, "mutation_rationale", "rationale", "reason"),
        "spec": r.get("spec"),
    }


def _load_result_records() -> list[dict]:
    if not RESULTS_DIR.is_dir():
        return []
    raw: list[dict] = []
    for f in sorted(RESULTS_DIR.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    raw.sort(key=lambda r: r.get("iteration", 0))
    return [_normalize_record(r) for r in raw]


@app.get("/api/results")
async def get_results() -> JSONResponse:
    records = _load_result_records()
    return JSONResponse({"empty": not records, "iterations": records})


# Static frontend last, so it never shadows the /api/* routes above.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
