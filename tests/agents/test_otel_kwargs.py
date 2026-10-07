"""D5a acceptance test: the dead otel_metrics kwarg must warn (map A2)."""

import pytest

from ursa.agents.chat_agent import BasicChatAgent


def test_d5a_otel_metrics_kwarg_emits_deprecation_warning(chat_model, tmp_path):
    with pytest.warns(DeprecationWarning, match="otel_metrics"):
        BasicChatAgent(llm=chat_model, workspace=tmp_path, otel_metrics=True)
