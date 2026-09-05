import json

import httpx
import pytest

from metaagent import gateway
from metaagent.gateway import GatewayError, complete

MESSAGES = [{"role": "user", "content": "hello there"}]


def _openai_response(content: str, usage: dict | None = None) -> dict:
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _json_response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def _configure_live_env(monkeypatch, *, gateway_mode: str = "live") -> None:
    monkeypatch.setenv("GATEWAY_MODE", gateway_mode)
    monkeypatch.setenv("META_BASE_URL", "https://meta.example.com/v1")
    monkeypatch.setenv("META_API_KEY", "meta-key")
    monkeypatch.setenv("META_MODEL", "meta-model")
    monkeypatch.setenv("WORKER_BASE_URL", "https://worker.example.com/v1")
    monkeypatch.setenv("WORKER_API_KEY", "worker-key")
    monkeypatch.setenv("WORKER_MODEL", "worker-model")
    monkeypatch.setenv("FALLBACK_BASE_URL", "https://fallback.example.com/v1")
    monkeypatch.setenv("FALLBACK_API_KEY", "fallback-key")
    monkeypatch.setenv("FALLBACK_MODEL", "fallback-model")


def _install_transport(monkeypatch, handler) -> None:
    monkeypatch.setattr(gateway, "_new_http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    # Keep retry backoff out of the way of test runtime.
    monkeypatch.setattr(gateway, "BASE_BACKOFF_SECONDS", 0.0)


# ---------------------------------------------------------------------------
# Fake mode: zero network, zero keys
# ---------------------------------------------------------------------------

async def test_fake_mode_requires_no_network_or_keys(monkeypatch):
    monkeypatch.delenv("META_BASE_URL", raising=False)
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("GATEWAY_MODE", "fake")

    def _boom(*args, **kwargs):
        raise AssertionError("fake mode must not touch the network")

    monkeypatch.setattr(gateway, "_new_http_client", _boom)

    result = await complete("meta", MESSAGES)

    assert result.provider == "fake"
    assert result.text
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.latency_ms >= 0


async def test_fake_mode_honors_json_schema(monkeypatch):
    monkeypatch.setenv("GATEWAY_MODE", "fake")
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    }

    result = await complete("worker", MESSAGES, response_format={"schema": schema})

    parsed = json.loads(result.text)
    assert "answer" in parsed


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

async def test_token_accounting_uses_real_provider_usage(monkeypatch):
    _configure_live_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, _openai_response("hi", usage={"prompt_tokens": 7, "completion_tokens": 3}))

    _install_transport(monkeypatch, handler)

    result = await complete("worker", MESSAGES)

    assert result.input_tokens == 7
    assert result.output_tokens == 3
    assert result.provider == "worker.example.com"
    assert result.latency_ms >= 0


async def test_token_accounting_falls_back_to_estimate_when_usage_missing(monkeypatch):
    _configure_live_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, _openai_response("a reply with no usage block"))

    _install_transport(monkeypatch, handler)

    result = await complete("worker", MESSAGES)

    assert result.input_tokens == gateway._messages_token_estimate(MESSAGES)
    assert result.output_tokens == gateway._estimate_tokens("a reply with no usage block")
    assert result.provider.endswith(":estimated")


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

async def test_retries_on_5xx_then_succeeds(monkeypatch):
    _configure_live_env(monkeypatch)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            return _json_response(503, {"error": "server busy"})
        return _json_response(200, _openai_response("ok", usage={"prompt_tokens": 1, "completion_tokens": 1}))

    _install_transport(monkeypatch, handler)

    result = await complete("worker", MESSAGES)

    assert calls["count"] == 2
    assert result.text == "ok"


async def test_retries_on_timeout_then_succeeds(monkeypatch):
    _configure_live_env(monkeypatch)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            raise httpx.ReadTimeout("timed out", request=request)
        return _json_response(200, _openai_response("ok", usage={"prompt_tokens": 1, "completion_tokens": 1}))

    _install_transport(monkeypatch, handler)

    result = await complete("worker", MESSAGES)

    assert calls["count"] == 2
    assert result.text == "ok"


async def test_retry_exhaustion_raises_gateway_error_without_fallback_configured(monkeypatch):
    _configure_live_env(monkeypatch)
    monkeypatch.setenv("FALLBACK_BASE_URL", "https://worker.example.com/v1")
    monkeypatch.setenv("FALLBACK_API_KEY", "worker-key")
    monkeypatch.setenv("FALLBACK_MODEL", "worker-model")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _json_response(503, {"error": "still busy"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GatewayError):
        await complete("worker", MESSAGES)

    # MAX_RETRIES attempts against primary, then MAX_RETRIES attempts against fallback.
    assert calls["count"] == gateway.MAX_RETRIES * 2


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

async def test_falls_back_to_fallback_provider_after_primary_exhausts_retries(monkeypatch):
    _configure_live_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "worker.example.com":
            return _json_response(500, {"error": "primary down"})
        assert request.url.host == "fallback.example.com"
        return _json_response(200, _openai_response("from fallback", usage={"prompt_tokens": 2, "completion_tokens": 2}))

    _install_transport(monkeypatch, handler)

    result = await complete("worker", MESSAGES)

    assert result.text == "from fallback"
    assert result.model == "fallback-model"
    assert result.provider == "fallback.example.com"


async def test_fallback_logs_event(monkeypatch, caplog):
    _configure_live_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "worker.example.com":
            return _json_response(500, {"error": "primary down"})
        return _json_response(200, _openai_response("ok", usage={"prompt_tokens": 1, "completion_tokens": 1}))

    _install_transport(monkeypatch, handler)

    with caplog.at_level("WARNING", logger="metaagent.gateway"):
        await complete("worker", MESSAGES)

    assert any("fallback triggered" in record.message for record in caplog.records)


async def test_both_primary_and_fallback_failing_raises(monkeypatch):
    _configure_live_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(500, {"error": "down"})

    _install_transport(monkeypatch, handler)

    with pytest.raises(GatewayError, match="primary provider failed"):
        await complete("worker", MESSAGES)


# ---------------------------------------------------------------------------
# JSON-schema validate-and-retry (repair) loop
# ---------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


async def test_json_repair_succeeds_on_second_attempt(monkeypatch):
    _configure_live_env(monkeypatch)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return _json_response(200, _openai_response("not json at all"))
        return _json_response(
            200,
            _openai_response(json.dumps({"answer": "42"}), usage={"prompt_tokens": 1, "completion_tokens": 1}),
        )

    _install_transport(monkeypatch, handler)

    result = await complete("worker", MESSAGES, response_format={"schema": SCHEMA})

    assert calls["count"] == 2
    assert json.loads(result.text) == {"answer": "42"}


async def test_json_repair_validates_required_fields(monkeypatch):
    _configure_live_env(monkeypatch)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            # Valid JSON, but missing the required "answer" field.
            return _json_response(200, _openai_response(json.dumps({"wrong_key": "x"})))
        return _json_response(
            200,
            _openai_response(json.dumps({"answer": "42"}), usage={"prompt_tokens": 1, "completion_tokens": 1}),
        )

    _install_transport(monkeypatch, handler)

    result = await complete("worker", MESSAGES, response_format={"schema": SCHEMA})

    assert calls["count"] == 2
    assert json.loads(result.text) == {"answer": "42"}


async def test_json_repair_exhausted_raises(monkeypatch):
    _configure_live_env(monkeypatch)
    monkeypatch.setenv("FALLBACK_BASE_URL", "https://worker.example.com/v1")
    monkeypatch.setenv("FALLBACK_API_KEY", "worker-key")
    monkeypatch.setenv("FALLBACK_MODEL", "worker-model")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _json_response(200, _openai_response("still not json"))

    _install_transport(monkeypatch, handler)

    with pytest.raises(GatewayError):
        await complete("worker", MESSAGES, response_format={"schema": SCHEMA})

    # (JSON_REPAIR_RETRIES + 1) attempts against primary, then again against fallback.
    assert calls["count"] == (gateway.JSON_REPAIR_RETRIES + 1) * 2


async def test_json_repair_strips_markdown_code_fence(monkeypatch):
    _configure_live_env(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        fenced = "```json\n" + json.dumps({"answer": "42"}) + "\n```"
        return _json_response(200, _openai_response(fenced, usage={"prompt_tokens": 1, "completion_tokens": 1}))

    _install_transport(monkeypatch, handler)

    result = await complete("worker", MESSAGES, response_format={"schema": SCHEMA})

    assert json.loads(result.text) == {"answer": "42"}


# ---------------------------------------------------------------------------
# Record / replay
# ---------------------------------------------------------------------------

async def test_record_then_replay_roundtrip(monkeypatch, tmp_path):
    _configure_live_env(monkeypatch, gateway_mode="record")
    monkeypatch.setattr(gateway, "FIXTURES_DIR", tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, _openai_response("recorded answer", usage={"prompt_tokens": 5, "completion_tokens": 5}))

    _install_transport(monkeypatch, handler)

    recorded = await complete("worker", MESSAGES)
    assert recorded.text == "recorded answer"
    assert any(tmp_path.iterdir())

    monkeypatch.setenv("GATEWAY_MODE", "replay")

    def _boom(*args, **kwargs):
        raise AssertionError("replay mode must not touch the network")

    monkeypatch.setattr(gateway, "_new_http_client", _boom)

    replayed = await complete("worker", MESSAGES)

    assert replayed.text == "recorded answer"
    assert replayed.input_tokens == 5
    assert replayed.output_tokens == 5


async def test_replay_without_fixture_raises(monkeypatch, tmp_path):
    _configure_live_env(monkeypatch, gateway_mode="replay")
    monkeypatch.setattr(gateway, "FIXTURES_DIR", tmp_path)

    with pytest.raises(GatewayError, match="no recorded fixture"):
        await complete("worker", MESSAGES)
