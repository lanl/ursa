import asyncio
import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from ursa.util.health import probe_embedding_model, run_probe_with_deadline


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


def test_embedding_probe_normalizes_capability_and_authentication_errors():
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
        probe_embedding_model(config_with_error(mixed_provider_error))
    assert capability.value.__context__ is None

    for detail in ("401", "403 Forbidden", "Unauthorized"):
        with pytest.raises(
            ValueError,
            match=(
                "The selected model may not support embeddings, or the API "
                "key may not have access to it"
            ),
        ) as ambiguous:
            probe_embedding_model(config_with_error(RuntimeError(detail)))
        assert ambiguous.value.__context__ is None

    unrelated_error = RuntimeError("rate limited")
    with pytest.raises(RuntimeError, match="rate limited") as captured:
        probe_embedding_model(config_with_error(unrelated_error))
    assert captured.value is unrelated_error


@pytest.mark.skipif(
    os.name != "posix", reason="invalid pass_fds is POSIX-specific"
)
def test_probe_starts_with_textual_invalid_stderr():
    script = """
        import sys
        from ursa.util.health import run_probe_with_deadline

        class TextualStderr:
            def write(self, text): return len(text)
            def flush(self): pass
            def fileno(self): return -1

        sys.stderr = TextualStderr()
        print(run_probe_with_deadline(str, "connected", timeout=5))
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "connected"
