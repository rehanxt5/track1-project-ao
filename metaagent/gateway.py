"""The model gateway: the only module in the codebase that knows about LLM
providers. Every other module calls `complete()` and never touches httpx or
provider-specific request/response shapes directly.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import (
    FIXTURES_DIR,
    ProviderConfig,
    gateway_mode,
    get_fallback_config,
    get_layer_config,
)

logger = logging.getLogger("metaagent.gateway")

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 0.5
JSON_REPAIR_RETRIES = 2
REQUEST_TIMEOUT_SECONDS = 60.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Reasoning models (e.g. glm-4-7-flash) spend an unbounded number of tokens on
# hidden chain-of-thought before the answer even starts. A low max_tokens (or
# an omitted one, left to the provider's own tiny default) truncates them
# mid-thought: content stays null and complete() would silently return "".
#
# Measured live against glm-4-7-flash/TensorMux on architect.generate(n_seeds=3)
# (SYNTHESIZE: reasoning + a 3-seed multi-role JSON spec in one completion):
#   max_tokens=2048  -> FAILS: truncated at 2130 reasoning tokens, zero output
#   max_tokens=8192  -> ok: 1705 reasoning + 3227 output tokens
#   max_tokens=16384 -> ok: 3420 reasoning + 1154 output tokens
# 8192 is the floor below which even a plain short answer risks truncation,
# since reasoning alone ran past 2048 and past 3400 tokens in these runs
# before any output began. Callers with larger expected output (e.g. the
# architect's synthesis/repair calls, which pass their own explicit
# max_tokens sized for reasoning + a multi-role spec) should not rely on
# this default; see architect.py's SYNTHESIS_REASONING_RESERVE. See the
# empty-content+finish_reason="length" check below for the case where even
# this floor isn't enough -- that raises GatewayReasoningBudgetError so
# callers can retry with a bigger budget instead of getting silent "".
DEFAULT_MAX_TOKENS = 8192

_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    provider: str
    reasoning: str = ""
    reasoning_tokens: int = 0


class GatewayError(Exception):
    """Raised when a provider (and its fallback) both fail to produce a
    usable completion."""


class GatewayRequestError(GatewayError):
    """Raised for deterministic, request-shaped failures: reasoning-budget
    exhaustion, JSON-schema repair exhaustion, non-retryable 4xx. A different
    provider cannot fix these by being retried, so `complete()` raises them
    immediately without attempting the fallback provider."""


class GatewayReasoningBudgetError(GatewayRequestError):
    """Raised specifically when a reasoning model exhausted max_tokens on
    chain-of-thought before emitting an answer (finish_reason="length",
    empty content). Distinct from other GatewayRequestError cases so a
    caller that can afford it (e.g. the architect's synthesis/repair calls)
    can catch this one specifically and retry once with a larger budget
    instead of giving up."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def complete(
    layer: str,
    messages: list[dict],
    *,
    response_format: dict | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    reasoning: bool | str | None = None,
) -> Completion:
    mode = gateway_mode()
    if mode == "fake":
        return await _fake_complete(layer, messages, response_format)

    provider = get_layer_config(layer)
    payload = _build_payload(provider.model, messages, response_format, max_tokens, temperature, reasoning)

    try:
        return await _complete_with_provider(provider, payload, messages, response_format, mode)
    except GatewayRequestError:
        # Deterministic, request-shaped failure: a different provider would
        # fail the same way, so don't burn tokens/latency on a fallback call.
        raise
    except GatewayError as primary_exc:
        fallback = get_fallback_config()
        logger.warning(
            "gateway fallback triggered: layer=%s primary=%s fallback=%s reason=%s",
            layer,
            provider.base_url,
            fallback.base_url,
            primary_exc,
        )
        fallback_payload = _build_payload(fallback.model, messages, response_format, max_tokens, temperature, reasoning)
        try:
            return await _complete_with_provider(fallback, fallback_payload, messages, response_format, mode)
        except GatewayError as fallback_exc:
            raise GatewayError(
                f"primary provider failed ({primary_exc}) and fallback provider also failed ({fallback_exc})"
            ) from fallback_exc


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

def _build_payload(
    model: str,
    messages: list[dict],
    response_format: dict | None,
    max_tokens: int | None,
    temperature: float | None,
    reasoning: bool | str | None = None,
) -> dict:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if response_format is not None:
        payload["response_format"] = response_format
    payload["max_tokens"] = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning is not None:
        # `reasoning` is the optimizer-facing lever for the accuracy/cost/speed
        # tradeoff: reasoning models burn ~70x more tokens thinking than
        # answering, so being able to dial this per role matters. We send both
        # OpenAI-style `reasoning_effort` and vLLM-style `chat_template_kwargs`
        # since different OpenAI-compatible backends honor one or the other.
        if reasoning is False or reasoning == "none":
            payload["reasoning_effort"] = "none"
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif reasoning is True:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            payload["reasoning_effort"] = reasoning
            payload["chat_template_kwargs"] = {"enable_thinking": True}
    return payload


def _extract_schema(response_format: dict | None) -> dict | None:
    if not response_format:
        return None
    if "schema" in response_format:
        return response_format["schema"]
    return response_format.get("json_schema", {}).get("schema")


# ---------------------------------------------------------------------------
# HTTP transport (single provider, with retry + backoff)
# ---------------------------------------------------------------------------

def _new_http_client() -> httpx.AsyncClient:
    """Overridable factory so tests can inject an httpx.MockTransport and
    exercise the real request/retry code path with zero network access."""
    return httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)


async def _call_provider_with_retries(provider: ProviderConfig, payload: dict) -> tuple[dict, float]:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        start = time.perf_counter()
        try:
            async with _new_http_client() as client:
                resp = await client.post(
                    f"{provider.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {provider.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            latency_ms = (time.perf_counter() - start) * 1000
            if resp.status_code in _RETRYABLE_STATUS:
                last_exc = GatewayError(
                    f"HTTP {resp.status_code} from {provider.base_url}: {resp.text[:200]}"
                )
                logger.warning(
                    "gateway retryable status %s from %s (attempt %d/%d)",
                    resp.status_code,
                    provider.base_url,
                    attempt + 1,
                    MAX_RETRIES,
                )
            elif resp.status_code >= 400:
                raise GatewayRequestError(
                    f"HTTP {resp.status_code} from {provider.base_url}: {resp.text[:200]}"
                )
            else:
                return resp.json(), latency_ms
        except httpx.TimeoutException as exc:
            last_exc = exc
            logger.warning(
                "gateway timeout calling %s (attempt %d/%d)",
                provider.base_url,
                attempt + 1,
                MAX_RETRIES,
            )
        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(BASE_BACKOFF_SECONDS * (2**attempt))
    raise GatewayError(f"exhausted retries against {provider.base_url}: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Provider-level completion: dispatches live/record/replay + JSON repair loop
# ---------------------------------------------------------------------------

async def _complete_with_provider(
    provider: ProviderConfig,
    payload: dict,
    messages: list[dict],
    response_format: dict | None,
    mode: str,
) -> Completion:
    schema = _extract_schema(response_format)
    attempt_messages = list(messages)
    text_out = ""
    reasoning_out = ""
    data: dict = {}
    latency_ms = 0.0

    for json_attempt in range(JSON_REPAIR_RETRIES + 1):
        current_payload = dict(payload)
        current_payload["messages"] = attempt_messages

        if mode == "replay":
            data, latency_ms = _replay_response(provider, current_payload)
        else:
            data, latency_ms = await _call_provider_with_retries(provider, current_payload)
            if mode == "record":
                _record_response(provider, current_payload, data)

        text, reasoning, finish_reason, usage = _extract_text_and_usage(data)
        reasoning_out = reasoning

        if not text and finish_reason == "length":
            raise GatewayReasoningBudgetError(
                "reasoning budget exhausted: provider truncated the response "
                f"(finish_reason=length) before emitting an answer "
                f"(reasoning_tokens={usage.get('reasoning_tokens', 0)}); "
                "raise max_tokens or pass reasoning=\"none\"/False for this role"
            )

        if response_format is None:
            text_out = text
            break

        parsed, error = _try_parse_json(text, schema)
        if error is None:
            text_out = json.dumps(parsed)
            break

        logger.warning(
            "gateway JSON validation failed (attempt %d/%d): %s",
            json_attempt + 1,
            JSON_REPAIR_RETRIES + 1,
            error,
        )
        if json_attempt >= JSON_REPAIR_RETRIES:
            raise GatewayRequestError(f"model returned invalid JSON after {JSON_REPAIR_RETRIES + 1} attempts: {error}")

        attempt_messages = attempt_messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    f"Your last response was not valid JSON matching the required schema "
                    f"({error}). Reply again with ONLY valid JSON matching the schema, no other text."
                ),
            },
        ]

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    reasoning_tokens = usage.get("reasoning_tokens", 0)
    estimated = False
    if input_tokens is None:
        input_tokens = _messages_token_estimate(messages)
        estimated = True
    if output_tokens is None:
        output_tokens = _estimate_tokens(text_out)
        estimated = True

    provider_label = _provider_name(provider.base_url)
    if estimated:
        provider_label = f"{provider_label}:estimated"

    return Completion(
        text=text_out,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model=provider.model,
        provider=provider_label,
        reasoning=reasoning_out,
        reasoning_tokens=reasoning_tokens,
    )


def _extract_text_and_usage(data: dict) -> tuple[str, str, str | None, dict]:
    """Pull answer text, chain-of-thought, finish reason, and usage out of an
    OpenAI-compatible response. Reasoning models put chain-of-thought in
    `message.reasoning` and leave `message.content` null until the answer
    starts, so `content` must never be assumed present."""
    choice = data["choices"][0]
    message = choice.get("message", {})
    text = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    finish_reason = choice.get("finish_reason")

    usage_raw = data.get("usage") or {}
    usage: dict[str, int] = {}
    if "prompt_tokens" in usage_raw:
        usage["input_tokens"] = usage_raw["prompt_tokens"]

    completion_tokens = usage_raw.get("completion_tokens")
    details = usage_raw.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    if reasoning_tokens is None:
        # Provider didn't break out reasoning tokens explicitly (TensorMux
        # doesn't); estimate from the reasoning text so cost accounting can
        # still separate thinking tokens from answer tokens.
        reasoning_tokens = _estimate_tokens(reasoning)
    usage["reasoning_tokens"] = reasoning_tokens

    if completion_tokens is not None:
        usage["output_tokens"] = max(completion_tokens - reasoning_tokens, 0)

    return text, reasoning, finish_reason, usage


def _provider_name(base_url: str) -> str:
    try:
        host = httpx.URL(base_url).host
        return host or base_url
    except Exception:
        return base_url


# ---------------------------------------------------------------------------
# Token estimation (fallback when a provider omits usage data)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic, used only when a provider omits usage."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _messages_token_estimate(messages: list[dict]) -> int:
    return sum(_estimate_tokens(m.get("content") or "") for m in messages)


# ---------------------------------------------------------------------------
# JSON-schema validation + repair
# ---------------------------------------------------------------------------

def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _try_parse_json(text: str, schema: dict | None) -> tuple[Any, str | None]:
    try:
        parsed = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if schema is not None:
        error = _validate_schema(parsed, schema)
        if error:
            return None, error
    return parsed, None


def _validate_schema(value: Any, schema: dict) -> str | None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return f"expected object, got {type(value).__name__}"
        for key in schema.get("required", []):
            if key not in value:
                return f"missing required field {key!r}"
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                sub_error = _validate_schema(value[key], subschema)
                if sub_error:
                    return f"field {key!r}: {sub_error}"
        return None
    if expected_type in _TYPE_MAP:
        if expected_type == "integer" and isinstance(value, bool):
            return "expected integer, got boolean"
        if not isinstance(value, _TYPE_MAP[expected_type]):
            return f"expected {expected_type}, got {type(value).__name__}"
        return None
    return None  # unknown/unspecified type: accept as-is


# ---------------------------------------------------------------------------
# Record / replay fixtures (offline testing without burning tokens or keys)
# ---------------------------------------------------------------------------

def _fixture_key(provider: ProviderConfig, payload: dict) -> str:
    blob = json.dumps({"base_url": provider.base_url, "payload": payload}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _fixture_path(key: str) -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR / f"{key}.json"


def _record_response(provider: ProviderConfig, payload: dict, response_data: dict) -> None:
    key = _fixture_key(provider, payload)
    _fixture_path(key).write_text(json.dumps({"request": payload, "response": response_data}, indent=2))


def _replay_response(provider: ProviderConfig, payload: dict) -> tuple[dict, float]:
    key = _fixture_key(provider, payload)
    path = _fixture_path(key)
    if not path.exists():
        raise GatewayError(f"no recorded fixture for request (key={key}); run in record mode first")
    data = json.loads(path.read_text())
    return data["response"], 0.0


# ---------------------------------------------------------------------------
# Fake mode: deterministic, offline, no network, no keys
# ---------------------------------------------------------------------------

async def _fake_complete(layer: str, messages: list[dict], response_format: dict | None) -> Completion:
    start = time.perf_counter()
    provider = get_layer_config(layer)
    model = provider.model or f"fake-{layer}"
    schema = _extract_schema(response_format)

    if schema is not None:
        text = json.dumps(_fake_value_for_schema(schema))
    else:
        last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        text = f"[fake:{model}] {last_user[:80]}"

    await asyncio.sleep(0)
    latency_ms = (time.perf_counter() - start) * 1000
    return Completion(
        text=text,
        input_tokens=_messages_token_estimate(messages),
        output_tokens=_estimate_tokens(text),
        latency_ms=latency_ms,
        model=model,
        provider="fake",
    )


def _fake_value_for_schema(schema: dict) -> Any:
    schema_type = schema.get("type", "object")
    if schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties.keys()))
        return {key: _fake_value_for_schema(properties.get(key, {"type": "string"})) for key in required}
    if schema_type == "string":
        return "fake"
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        return [_fake_value_for_schema(schema.get("items", {"type": "string"}))]
    return None
