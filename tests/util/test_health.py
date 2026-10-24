import asyncio
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ursa.util.health import probe_embedding_model, run_probe_with_deadline


def _block():
    time.sleep(60)


async def _block_async():
    await asyncio.sleep(60)


@pytest.mark.parametrize("probe", [_block, _block_async])
def test_probe_deadline_terminates_blocked_work(probe):
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out after 0.05 seconds"):
        run_probe_with_deadline(probe, timeout=0.05)
    assert time.monotonic() - started < 3


def _embedding_failure(error):
    return SimpleNamespace(
        init_embedding=lambda: SimpleNamespace(
            embed_query=lambda _: (_ for _ in ()).throw(error)
        )
    )


def test_embedding_probe_normalizes_provider_errors():
    cases = [
        (
            "401: model does not support embeddings",
            "does not support embeddings",
        ),
        ("401", "model may not support embeddings"),
        ("403 Forbidden", "model may not support embeddings"),
        ("Unauthorized", "model may not support embeddings"),
    ]
    for detail, message in cases:
        with pytest.raises(ValueError, match=message) as captured:
            probe_embedding_model(_embedding_failure(RuntimeError(detail)))
        assert captured.value.__context__ is None

    error = RuntimeError("rate limited")
    with pytest.raises(RuntimeError, match="rate limited") as captured:
        probe_embedding_model(_embedding_failure(error))
    assert captured.value is error


@pytest.mark.skipif(
    os.name != "posix", reason="invalid pass_fds is POSIX-specific"
)
def test_probe_starts_with_textual_invalid_stderr():
    script = """import sys
from ursa.util.health import run_probe_with_deadline
class Stderr:
 def write(self, text): return len(text)
 def flush(self): pass
 def fileno(self): return -1
sys.stderr = Stderr()
print(run_probe_with_deadline(str, 'connected', timeout=5))"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert (result.returncode, result.stdout.strip()) == (0, "connected")
