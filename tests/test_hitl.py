import io
import itertools
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console as RealConsole

from ursa.cli.hitl import HITL, UrsaRepl
from ursa.config import UrsaConfig


@pytest.fixture(scope="function")
def ursa_config(tmpdir, chat_model, embedding_model):
    config = UrsaConfig(
        workspace=Path(tmpdir),
        llm_model=chat_model._testing_only_kwargs,
        emb_model=embedding_model._testing_only_kwargs,
    )
    print("ursa config:", config)  # Displayed on test failure
    return config


async def test_default_config_smoke(ursa_config):
    hitl = HITL(ursa_config)
    assert hitl is not None
    assert set(hitl.agents.keys()) >= {"chatter"}
    out = await hitl.run_agent("chatter", "Hello! What is your name?")
    print("chat out:", out)
    assert len(out) > 0
    assert False


def test_has_all_agent_do_methods(ursa_config):
    hitl = HITL(ursa_config)
    repl = UrsaRepl(hitl)
    for name in hitl.agents.keys():
        assert hasattr(repl, f"do_{name}")


def check_script(
    ursa_config: UrsaConfig,
    input_expected: list[tuple[str, str | int | re.Pattern | None]],
):
    stdout = io.StringIO()

    def console_factory(*args, **kwargs):
        kwargs["record"] = True
        kwargs["force_terminal"] = False
        kwargs["force_interactive"] = False
        return RealConsole(*args, **kwargs)

    # Patch the Console constructor so we can snoop
    with patch("ursa.cli.hitl.Console", new=console_factory):
        shell = UrsaRepl(HITL(ursa_config), stdout=stdout)

    trace = []

    # Feed the REPL with the script and check the output matches
    # expectations
    for input, ref in input_expected:
        shell.onecmd(input)
        output = shell.console.export_text()
        match ref:
            case str():
                assert output == ref
            case int():
                assert len(output.strip()) >= ref
            case re.Pattern():
                assert ref.match(output)
            case None:
                pass
            case _:
                assert False, f"Unknown reference type: {ref}"

        trace.append({"input": input, "output": output})

    return trace


def test_repl_smoke(ursa_config):
    trace = check_script(
        ursa_config,
        [
            ("What is your name?", 1),
            ("What is your purpose?", 1),
            ("help", re.compile(r".*Documented commands")),
        ],
    )
    print(trace)


async def test_chat(ursa_config):
    hitl = HITL(ursa_config)
    out = await hitl.run_agent(
        "chatter",
        "What is your name?",
    )
    print(out)
    assert out is not None


@pytest.mark.parametrize(
    "agent",
    ["chat", "execute", "hypothesize", "plan", "web", "recall"],
)
def test_agent_repl_smoke(ursa_config: UrsaConfig, agent: str):
    trace = check_script(
        ursa_config,
        [(f"{agent} What is your purpose?", None)],
    )
    print(trace)
