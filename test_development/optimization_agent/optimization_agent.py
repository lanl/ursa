from langchain_core.tools import tool

import ursa.agents.optimization_agent as opt_mod


async def test_optimization_agent_produces_verified_summary(
    chat_model, tmpdir, monkeypatch
):
    tool_calls: dict[str, object] = {}

    @tool("feasibility_check_auto")
    def fake_feasibility_check_auto(
        constraints: list[str] | None = None, **kwargs
    ) -> dict[str, object]:
        """Return a simple feasibility verdict while recording invocation args."""
        tool_calls["constraints"] = constraints or []
        tool_calls["kwargs"] = kwargs
        return {"feasible": True, "checks": "stubbed"}

    # Keep the agent's internal prompts brief so the test remains responsive.
    monkeypatch.setattr(opt_mod, "fca", fake_feasibility_check_auto)
    monkeypatch.setattr(
        opt_mod,
        "feasibility_prompt",
        "Use feasibility_check_auto with a small placeholder payload to confirm feasibility.",
    )
    monkeypatch.setattr(
        opt_mod,
        "code_generator_prompt",
        "Return minimal python code: print('ok').",
    )
    monkeypatch.setattr(
        opt_mod,
        "verifier_prompt",
        "Return the previous ProblemSpec unchanged but ensure status is VERIFIED.",
    )
    monkeypatch.setattr(
        opt_mod,
        "explainer_prompt",
        "Summarize the verified optimization problem in 25 words or fewer.",
    )

    agent = opt_mod.OptimizationAgent(
        llm=chat_model.model_copy(update={"max_tokens": 4000}),
        workspace=tmpdir / ".ursa_workspace",
    )

    request = (
        "Formulate the linear program minimize x + y with constraints x >= 0, "
        "y >= 0, and x + y <= 1. Keep every response concise."
    )

    result = await agent.ainvoke({"user_input": request})

    assert result["problem_spec"]["status"].upper() == "VERIFIED"
    assert result["code"].strip().startswith("print")
    assert result["summary"].strip()
    assert "solver" in result["solver"]
    assert "constraints" in tool_calls and tool_calls["constraints"] is not None
