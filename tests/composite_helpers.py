from __future__ import annotations

from collections.abc import Iterator

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field

from ursa.agents.execution_agent import ReviewAssessment
from ursa.agents.planning_agent import Plan, PlanStep


def two_step_plan() -> Plan:
    return Plan(
        steps=[
            PlanStep(
                name="Collect facts",
                description="Gather task constraints.",
                requires_code=False,
                expected_outputs=["constraints list"],
                success_criteria=["constraints identified"],
            ),
            PlanStep(
                name="Produce answer",
                description="Deliver the final result.",
                requires_code=False,
                expected_outputs=["final answer"],
                success_criteria=["answer is complete"],
            ),
        ]
    )


class CompositeFakeModel(GenericFakeChatModel):
    """Deterministic model supporting both normal and structured calls."""

    plan: Plan = Field(default_factory=two_step_plan)
    plain_requests: list[list[BaseMessage]] = Field(default_factory=list)
    structured_requests: list[tuple[str, list[BaseMessage]]] = Field(
        default_factory=list
    )
    messages: Iterator = Field(
        default_factory=lambda: iter([
            AIMessage(content="step-1-work"),
            AIMessage(content="step-1-summary"),
            AIMessage(content="step-2-work"),
            AIMessage(content="step-2-summary"),
        ])
    )

    def bind_tools(self, _tools, **_kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.plain_requests.append(list(messages))
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    def with_structured_output(self, schema, **_kwargs):
        model = self

        class StructuredResult:
            def invoke(self, messages, config=None):
                del config
                model.structured_requests.append((
                    schema.__name__,
                    list(messages),
                ))
                if schema is Plan:
                    return model.plan
                if schema is ReviewAssessment:
                    return ReviewAssessment(
                        is_complete=True,
                        reason="The requested step is complete.",
                    )
                raise AssertionError(f"Unexpected structured schema: {schema}")

        return StructuredResult()


def request_text(messages: list[BaseMessage]) -> str:
    return "\n".join(message.text for message in messages)
