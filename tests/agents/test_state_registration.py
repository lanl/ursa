"""Guard and ratchet coverage for agent state registration.

The graph compiles from ``state_type``; a TypedDict declared only in the
generic parameter or the legacy ``agent_state`` attribute never applies.
"""

import importlib
import sys
import warnings
from typing import TypedDict

import pytest

from ursa.agents.base import BaseAgent, UnregisteredAgentStateWarning


class SyntheticState(TypedDict, total=False):
    value: int


def test_guard_warns_for_unregistered_generic_parameter():
    with pytest.warns(UnregisteredAgentStateWarning, match="SyntheticState"):

        class Unregistered(BaseAgent[SyntheticState]):
            def _build_graph(self):
                pass


def test_guard_silent_for_registered_state():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnregisteredAgentStateWarning)

        class Registered(BaseAgent[SyntheticState]):
            state_type = SyntheticState

            def _build_graph(self):
                pass


def test_guard_warns_for_legacy_agent_state_attribute():
    sys.modules.pop("ursa.agents.rag_agent", None)
    with pytest.warns(UnregisteredAgentStateWarning, match="RAGState"):
        importlib.import_module("ursa.agents.rag_agent")
