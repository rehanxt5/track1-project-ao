"""Environment-driven configuration for LLM providers.

Every provider (meta / worker / fallback) is configured purely through env
vars so swapping an OpenAI-compatible endpoint never requires a code change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

VALID_LAYERS = ("meta", "worker")

_LAYER_PREFIX = {
    "meta": "META",
    "worker": "WORKER",
}

FIXTURES_DIR = Path(os.environ.get("GATEWAY_FIXTURES_DIR", "metaagent/_fixtures"))


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_key: str
    model: str


def _provider_from_env(prefix: str) -> ProviderConfig:
    return ProviderConfig(
        base_url=os.environ.get(f"{prefix}_BASE_URL", ""),
        api_key=os.environ.get(f"{prefix}_API_KEY", ""),
        model=os.environ.get(f"{prefix}_MODEL", ""),
    )


def get_layer_config(layer: str) -> ProviderConfig:
    """Return the provider config for the "meta" or "worker" layer."""
    prefix = _LAYER_PREFIX.get(layer)
    if prefix is None:
        raise ValueError(f"Unknown layer {layer!r}; expected one of {VALID_LAYERS}")
    return _provider_from_env(prefix)


def get_fallback_config() -> ProviderConfig:
    return _provider_from_env("FALLBACK")


def gateway_mode() -> str:
    """One of "live" (default), "fake", "record", or "replay"."""
    return os.environ.get("GATEWAY_MODE", "live").lower()


def architect_reasoning_default() -> bool | str:
    """Whether the architect's SYNTHESIZE/repair calls request chain-of-
    thought by default, overridable per-call via generate(reasoning=...).

    Measured live against glm-4-7-flash/TensorMux, 3 trials each, on
    structured_extraction/generate(n_seeds=3):
      reasoning=True:  avg 98.7s/call, avg 2923 reasoning tokens burned,
                        33% of trials verified all seeds on the first try
                        (the rest needed a repair round-trip).
      reasoning=False: avg 31.8s/call (~3x faster), 0 reasoning tokens,
                        67% first-try verification rate, comparable seed
                        yield (2.7 vs 3.0 avg valid seeds across trials).
    Reasoning bought slower, not-obviously-better synthesis in this
    comparison, so it defaults off. Set ARCHITECT_REASONING=true (or a
    reasoning_effort string like "low") to turn it back on for
    experimentation; see PR description for the full comparison.
    """
    raw = os.environ.get("ARCHITECT_REASONING", "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if lowered in ("0", "false", "off", "none"):
        return False
    if lowered in ("1", "true", "on"):
        return True
    return raw
