import asyncio
import time
from types import SimpleNamespace

import pytest

from ursa.util import health
from ursa.util.health import run_probe_with_deadline
from ursa.util.secrets import SecretReference


class _ConfigValue(SimpleNamespace):
    def model_copy(self, *, update):
        return type(self)(**{**vars(self), **update})


def _block() -> None:
    time.sleep(60)


async def _block_async() -> None:
    await asyncio.sleep(60)


@pytest.mark.parametrize("probe", [_block, _block_async])
def test_probe_deadline_terminates_blocked_work(probe):
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out after 0.05 seconds"):
        run_probe_with_deadline(probe, timeout=0.05)
    assert time.monotonic() - started < 3


def test_embedding_probe_distinguishes_model_capability_from_authentication():
    def config_with_error(error):
        def embed_query(_text):
            raise error

        return SimpleNamespace(
            init_embedding=lambda: SimpleNamespace(embed_query=embed_query)
        )

    mixed_provider_error = RuntimeError(
        "Error code: 401 - This model does not support embeddings"
    )
    with pytest.raises(
        ValueError,
        match=r"^The selected model does not support embeddings\.$",
    ) as capability:
        health.probe_embedding_model(config_with_error(mixed_provider_error))
    assert capability.value.__context__ is None

    authentication_error = RuntimeError("401 Unauthorized")
    with pytest.raises(RuntimeError, match="401 Unauthorized") as captured:
        health.probe_embedding_model(config_with_error(authentication_error))
    assert captured.value is authentication_error


def test_config_health_runs_every_probe_and_collects_failures(monkeypatch):
    calls = []

    def run(probe, *args, timeout):
        calls.append((probe.__name__, args, timeout))
        failing = (
            getattr(args[0], "tag", None) == "bad-provider"
            or probe is health.probe_chat_config
            and args[-1] == "tool"
            or probe is health.probe_embedding_model
            or probe is health.probe_mcp_server
            and args[0] == "bad-mcp"
        )
        if failing:
            raise RuntimeError("private service detail")
        return "ok"

    monkeypatch.setattr(health, "run_probe_with_deadline", run)
    relative_key = SecretReference(keyring=True)
    model = _ConfigValue(
        api_key=relative_key,
        inference_provider="good",
        model_provider="openai",
    )
    config = SimpleNamespace(
        resolve=lambda: config,
        inference_providers={
            "good": _ConfigValue(api_key=relative_key, tag="provider"),
            "bad": _ConfigValue(api_key=None, tag="bad-provider"),
        },
        llm_model=model,
        emb_model=model,
        mcp_servers={"good": "server", "bad-mcp": "server"},
    )

    failures = health.check_config_health(config, timeout=7)

    assert list(failures) == [
        ("inference_providers", "bad"),
        ("llm_model", "tool"),
        ("emb_model",),
        ("mcp_servers", "bad-mcp"),
    ]
    assert len(calls) == 8
    assert {call[2] for call in calls} == {7}
    provider = calls[0][1][0]
    chat = calls[2][1][0]
    assert provider.api_key.keyring == "good"
    assert chat.api_key.keyring == "good"
