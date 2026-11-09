from typing import Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from ursa.util import Checkpointer
from ursa.util.optimization_schema import (
    ConstraintType,
    DecisionVariableType,
    NotesType,
    ObjectiveType,
    ParameterType,
    ProblemSpec,
    SolverSpec,
)


class CustomStateModel(BaseModel):
    value: str


class StructuredState(TypedDict):
    models: list[Any]


@pytest.mark.parametrize(
    "factory",
    [
        "from_workspace",
        "from_path",
        "async_from_workspace",
        "async_from_path",
    ],
)
async def test_checkpointer_restores_structured_optimization_state(
    tmp_path, factory
):
    problem = ProblemSpec(
        title="Example",
        description_nl="Minimize x",
        decision_variables=[
            DecisionVariableType(
                name="x",
                type="continuous",
                domain="nonnegative",
                description="value",
            )
        ],
        parameters=[
            ParameterType(
                name="bound",
                value=0,
                description="lower bound",
                is_user_supplied=True,
            )
        ],
        objective=ObjectiveType(
            sense="minimize", expression_nl="x", tags=["linear"]
        ),
        constraints=[
            ConstraintType(
                name="bound", expression_nl="x >= 0", tags=["linear"]
            )
        ],
        status="VERIFIED",
        notes=NotesType(
            verifier="checked", feasibility="feasible", user="", assumptions=""
        ),
    )
    models = [
        problem,
        *problem.decision_variables,
        *problem.parameters,
        problem.objective,
        *problem.constraints,
        problem.notes,
        SolverSpec(solver="highs", library="scipy"),
        CustomStateModel(value="custom"),
    ]
    saver = getattr(Checkpointer, factory)(
        tmp_path, allowed_msgpack_modules=[CustomStateModel]
    )
    is_async = factory.startswith("async_")
    if is_async:
        saver = await saver
    builder = StateGraph(StructuredState)
    builder.add_node("keep", lambda state: state)
    builder.add_edge(START, "keep")
    builder.add_edge("keep", END)
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "structured"}}
    try:
        if is_async:
            await graph.ainvoke({"models": models}, config)
            restored = (await graph.aget_state(config)).values["models"]
        else:
            graph.invoke({"models": models}, config)
            restored = graph.get_state(config).values["models"]
        assert [type(model) for model in restored] == [
            type(model) for model in models
        ]
        assert restored == models
    finally:
        if is_async:
            await saver.conn.close()
        else:
            saver.conn.close()
