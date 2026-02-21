from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ursa.tools import solve_cmm_supply_chain_optimization
from ursa.workflows.base_workflow import BaseWorkflow


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("final_summary", "summary", "result", "output"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text
        messages = value.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            text = getattr(last, "text", None)
            if isinstance(text, str) and text.strip():
                return text
            content = getattr(last, "content", None)
            if isinstance(content, str) and content.strip():
                return content
    return str(value)


class CriticalMineralsWorkflow(BaseWorkflow):
    """Compose planning, retrieval, domain tools, and execution for minerals tasks."""

    def __init__(
        self,
        planner: Any,
        executor: Any,
        *,
        acquisition_agents: Mapping[str, Any] | None = None,
        rag_agent: Any | None = None,
        materials_agent: Any | None = None,
        simulation_agent: Any | None = None,
        workspace: str | Path = "critical_minerals_workspace",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.planner = planner
        self.executor = executor
        self.acquisition_agents = dict(acquisition_agents or {})
        self.rag_agent = rag_agent
        self.materials_agent = materials_agent
        self.simulation_agent = simulation_agent
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _invoke(self, inputs: Mapping[str, Any], **kw: Any) -> dict[str, Any]:
        del kw
        task = str(inputs["task"])
        domain_context = str(
            inputs.get(
                "domain_context",
                "critical minerals and materials supply chains",
            )
        )
        acquisition_context = str(inputs.get("acquisition_context", task))
        rag_context = str(inputs.get("rag_context", task))
        materials_context = str(inputs.get("materials_context", task))
        local_corpus_path = inputs.get("local_corpus_path")
        source_queries = inputs.get("source_queries", {})
        optimization_input = inputs.get("optimization_input")

        if not isinstance(source_queries, Mapping):
            raise TypeError("source_queries must be a mapping of source_name -> query")

        planning_prompt = (
            "Generate an implementation-ready plan for this technical task.\n"
            f"Domain context: {domain_context}\n"
            f"Task: {task}\n"
            "Prioritize data provenance, uncertainty handling, and reproducibility."
        )
        planning_output = self.planner.invoke(planning_prompt)

        acquisition_outputs: dict[str, Any] = {}
        for source_name, agent in self.acquisition_agents.items():
            query = str(source_queries.get(source_name, task))
            payload = {"query": query, "context": acquisition_context}
            try:
                acquisition_outputs[source_name] = agent.invoke(payload)
            except TypeError:
                acquisition_outputs[source_name] = agent.invoke(**payload)

        rag_output: Any | None = None
        if self.rag_agent is not None:
            if local_corpus_path and hasattr(self.rag_agent, "database_path"):
                self.rag_agent.database_path = Path(str(local_corpus_path))
            rag_output = self.rag_agent.invoke({"context": rag_context})

        materials_output: Any | None = None
        if self.materials_agent is not None and "materials_query" in inputs:
            materials_output = self.materials_agent.invoke(
                {
                    "query": inputs["materials_query"],
                    "context": materials_context,
                }
            )

        simulation_output: Any | None = None
        if self.simulation_agent is not None and "simulation_input" in inputs:
            simulation_output = self.simulation_agent.invoke(
                inputs["simulation_input"]
            )

        sections: list[str] = [
            f"Task:\n{task}",
            f"Domain context:\n{domain_context}",
            f"Planning output:\n{_coerce_text(planning_output)}",
        ]

        if acquisition_outputs:
            for source_name, output in acquisition_outputs.items():
                sections.append(
                    f"Acquisition ({source_name}) summary:\n{_coerce_text(output)}"
                )

        if rag_output is not None:
            sections.append(f"RAG summary:\n{_coerce_text(rag_output)}")

        if materials_output is not None:
            sections.append(
                f"Materials intelligence summary:\n{_coerce_text(materials_output)}"
            )

        if simulation_output is not None:
            sections.append(
                f"Simulation summary:\n{_coerce_text(simulation_output)}"
            )

        optimization_output: dict[str, Any] | None = None
        if optimization_input is not None:
            if not isinstance(optimization_input, Mapping):
                raise TypeError("optimization_input must be a mapping")
            optimization_output = solve_cmm_supply_chain_optimization(
                dict(optimization_input)
            )
            sections.append(
                "Optimization output (deterministic JSON):\n"
                + json.dumps(optimization_output, indent=2, sort_keys=True)
            )

        execution_instruction = str(
            inputs.get(
                "execution_instruction",
                "Produce a technically rigorous synthesis with explicit source"
                " grounding, assumptions, uncertainty notes, and actionable next"
                " steps for critical minerals decisions.",
            )
        )
        sections.append(f"Execution instruction:\n{execution_instruction}")

        executor_output = self.executor.invoke("\n\n".join(sections))

        return {
            "task": task,
            "plan": planning_output,
            "acquisition": acquisition_outputs,
            "rag": rag_output,
            "materials": materials_output,
            "simulation": simulation_output,
            "optimization": optimization_output,
            "executor_output": executor_output,
            "final_summary": _coerce_text(executor_output),
        }
