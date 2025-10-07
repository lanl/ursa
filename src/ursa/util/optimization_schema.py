from typing import Any, List, Literal, Optional, TypedDict


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
    value: Optional[Any]  # parameter value; None
    description: str  # natural language description
    is_user_supplied: bool  # 1 if user supplied parameter


class ObjectiveType(TypedDict):
    sense: Literal["minimize", "maximize"]  # objective sense
    expression_nl: str  # sympy-representable mathematical expression
    tags: List[
        Literal["linear", "quadratic", "nonlinear", "convex", "nonconvex"]
    ]  # objective type


class ConstraintType(TypedDict):
    name: str  # constraint name
    expression_nl: str  # sympy-representable mathematical expression
    tags: List[
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
    decision_variables: List[
        DecisionVariableType
    ]  # list of all decision variables
    parameters: List[ParameterType]  # list of all parameters
    objective: ObjectiveType  # structred objective function details
    constraints: List[ConstraintType]  # structured constraint details
    problem_class: Optional[str]  # optimization problem class
    latex: Optional[str]  # latex formulation of the problem
    status: Literal["DRAFT", "VERIFIED", "ERROR"]  # problem status
    notes: NotesType  # structured notes data


class SolverSpec(TypedDict):
    solver: str  # name of the solver, replace with Literal["Gurobi","Ipopt",...] to restrict solvers
    library: str  # library or relevant packages for the solver
    algorithm: Optional[str]  # algorithm used to solve the problem
    license: Optional[
        str
    ]  # License status of the solver (open-source, commercial,etc.)
    parameters: Optional[List[dict]]  # other parameters relevant to the problem
    notes: Optional[str]  # justifying the choice of solver
