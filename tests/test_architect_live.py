"""Live smoke test: exercises architect.generate() against the real
TensorMux API. Fake-mode tests can't catch provider-shaped failures (wrong
max_tokens, reasoning-model quirks) since they never leave the process --
that gap is exactly what let the GENERATE stage ship 100% broken against
the live API. This test is the guard against that gap recurring.

Skipped by default: it needs GATEWAY_MODE=live plus a real META_API_KEY (see
.env.example / `ln -sf ~/.ao/secrets/metaagent.env .env`). CI has neither, so
the normal suite stays green and key-free; run locally with a live .env to
exercise it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metaagent import config
from metaagent.architect import generate

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ["structured_extraction", "multi_step_tools", "sql_generation"]


def _live_keys_available() -> bool:
    if config.gateway_mode() != "live":
        return False
    meta = config.get_layer_config("meta")
    return bool(meta.api_key and meta.base_url)


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _live_keys_available(),
        reason="no live API key configured (GATEWAY_MODE=live + META_*/WORKER_* env vars required)",
    ),
]


def _load_domain(name: str) -> tuple[str, list[dict]]:
    domain_dir = REPO_ROOT / "domains" / name
    goal = (domain_dir / "goal.md").read_text()
    tools = json.loads((domain_dir / "tools.json").read_text())
    return goal, tools


@pytest.mark.parametrize("domain", DOMAINS)
async def test_live_generate_returns_verified_seeds(domain):
    goal, tools = _load_domain(domain)
    n_seeds = 3

    specs = await generate(goal, tools, n_seeds=n_seeds)

    assert len(specs) == n_seeds
    for spec in specs:
        used_tools = {t for role in spec.roles for t in role.tools}
        available = {t["name"] for t in tools}
        assert used_tools <= available
