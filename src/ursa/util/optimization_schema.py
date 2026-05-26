from typing import Any, Literal, TypedDict


class DecisionVariableType(TypedDict):
    name: str  # decision variable name
    type: Literal[
        "continuous",
        "integer",
        "logical",
        "infinite-dimensional",
        "finite-dimensional",
    ]  # decision variable type
    domain: str  # allowable values of variable
    description: str  # natural language description


class ParameterType(TypedDict):
    name: str  # parameter name
    value: Any | None  # parameter value; None
    description: str  # natural language description
    is_user_supplied: bool  # 1 if user supplied parameter


class ObjectiveType(TypedDict):
    sense: Literal["minimize", "maximize"]  # objective sense
    expression_nl: str  # sympy-representable mathematical expression
    tags: list[
        Literal["linear", "quadratic", "nonlinear", "convex", "nonconvex"]
    ]  # objective type


class ConstraintType(TypedDict):
    name: str  # constraint name
    expression_nl: str  # sympy-representable mathematical expression
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
    ]  # constraint type


class NotesType(TypedDict):
    verifier: str  # problem verification status and explanation
    feasibility: str  # problem feasibility status
    user: str  # notes to user
    assumptions: str  # assumptions made during formulation


class ProblemSpec(TypedDict):
    title: str  # name of the problem
    description_nl: str  # natural language description
    decision_variables: list[
        DecisionVariableType
    ]  # list of all decision variables
    parameters: list[ParameterType]  # list of all parameters
    objective: ObjectiveType  # structred objective function details
    constraints: list[ConstraintType]  # structured constraint details
    problem_class: str | None  # optimization problem class
    latex: str | None  # latex formulation of the problem
    status: Literal["DRAFT", "VERIFIED", "ERROR"]  # problem status
    notes: NotesType  # structured notes data


class SolverSpec(TypedDict):
    solver: str  # name of the solver, replace with Literal["Gurobi","Ipopt",...] to restrict solvers
    library: str  # library or relevant packages for the solver
    algorithm: str | None  # algorithm used to solve the problem
    license: (
        str | None
    )  # License status of the solver (open-source, commercial,etc.)
    parameters: list[dict] | None  # other parameters relevant to the problem
    notes: str | None  # justifying the choice of solver
