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
