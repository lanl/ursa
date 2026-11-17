"""Sandbox execution must tolerate output that is not valid UTF-8."""

import sys

from ursa.sandbox.base import NoSandbox


def test_no_sandbox_replaces_undecodable_output(tmp_path):
    cmd = (
        f"{sys.executable} -c "
        "\"import sys; sys.stdout.buffer.write(b'ok \\xff\\xfe end')\""
    )

    result = NoSandbox(cwd=tmp_path).execute(cmd)

    assert result.returncode == 0
    assert result.stdout.startswith("ok ")
    assert result.stdout.endswith(" end")
    assert "�" in result.stdout
