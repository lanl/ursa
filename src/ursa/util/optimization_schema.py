from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DecisionVariableType(BaseModel):
    name: str = Field(description="Decision variable name.")
    type: Literal[
        "continuous",
        "integer",
        "logical",
        "infinite-dimensional",
        "finite-dimensional",
    ] = Field(description="Decision variable type.")
    domain: str = Field(description="Allowable values of the variable.")
    description: str = Field(description="Natural language description.")


class ParameterType(BaseModel):
    name: str = Field(description="Parameter name.")
    value: Any | None = Field(
        default=None, description="Parameter value, or None if unspecified."
    )
    description: str = Field(description="Natural language description.")
    is_user_supplied: bool = Field(
        description="Whether the user supplied this parameter."
    )


class ObjectiveType(BaseModel):
    sense: Literal["minimize", "maximize"] = Field(
        description="Objective sense."
    )
    expression_nl: str = Field(
        description="Sympy-representable mathematical expression."
    )
    tags: list[
        Literal["linear", "quadratic", "nonlinear", "convex", "nonconvex"]
    ] = Field(description="Objective type tags.")


class ConstraintType(BaseModel):
    name: str = Field(description="Constraint name.")
    expression_nl: str = Field(
        description="Sympy-representable mathematical expression."
    )
    tags: list[
        Literal[
            "linear",
            "integer",
            "nonlinear",
            "equality",
            "inequality",
            "infinite-dimensional",
            "finite-dimensional",
        ]
    ] = Field(description="Constraint type tags.")


class NotesType(BaseModel):
    verifier: str = Field(
        description="Problem verification status and explanation."
    )
    feasibility: str = Field(description="Problem feasibility status.")
    user: str = Field(description="Notes to user.")
    assumptions: str = Field(description="Assumptions made during formulation.")


class ProblemSpec(BaseModel):
    title: str = Field(description="Name of the problem.")
    description_nl: str = Field(description="Natural language description.")
    decision_variables: list[DecisionVariableType] = Field(
        description="List of all decision variables."
    )
    parameters: list[ParameterType] = Field(
        description="List of all parameters."
    )
    objective: ObjectiveType = Field(
        description="Structured objective function details."
    )
    constraints: list[ConstraintType] = Field(
        description="Structured constraint details."
    )
    problem_class: str | None = Field(
        default=None, description="Optimization problem class."
    )
    latex: str | None = Field(
        default=None, description="LaTeX formulation of the problem."
    )
    status: Literal["DRAFT", "VERIFIED", "ERROR"] = Field(
        description="Problem status."
    )
    notes: NotesType = Field(description="Structured notes data.")


class SolverSpec(BaseModel):
    solver: str = Field(description="Name of the solver.")
    library: str = Field(
        description="Library or relevant packages for the solver."
    )
    algorithm: str | None = Field(
        default=None, description="Algorithm used to solve the problem."
    )
    license: str | None = Field(
        default=None,
        description="License status of the solver, e.g. open-source/commercial.",
    )
    parameters: list[dict[str, Any]] | None = Field(
        default=None, description="Other parameters relevant to the problem."
    )
    notes: str | None = Field(
        default=None, description="Justification for the solver choice."
    )
