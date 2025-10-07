import random
from typing import Annotated, List, Tuple

import numpy as np
import sympy as sp
from langchain_core.tools import tool
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


@tool(parse_docstring=True)
def heuristic_feasibility_check(
    constraints: Annotated[List[str], "List of strings like 'x0+x1<=5'"],
    variable_name: Annotated[
        List[str], "List of strings like 'x0', 'x1', etc."
    ],
    variable_type: Annotated[
        List[str], "List of strings like 'real', 'integer', 'boolean', etc."
    ],
    variable_bounds: Annotated[
        List[List[float]],
        "List of (lower bound, upper bound) tuples for x0, x1, ...'",
    ],
    samples: Annotated[int, "Number of random sample. Default 10000"] = 10000,
) -> Tuple[str]:
    """
    A tool for checking feasibility of the constraints.

    Args:
        constraints: list of strings like 'x0 + x1 <= 5', etc.
        variable_name: list of strings containing variable names used in constraint expressions.
        variable_type: list of strings like 'real', 'integer', 'boolean', etc.
        variable_bounds: list of (lower, upper) tuples for x0, x1, etc.
        samples: number of random samples, default value 10000

    Returns:
        A string indicating whether a feasible solution was found.
    """

    symbols = sp.symbols(variable_name)

    # Build a dict mapping each name to its Symbol, for parsing
    locals_map = {name: sym for name, sym in zip(variable_name, symbols)}

    # Parse constraints into Sympy Boolean expressions
    parsed_constraints = []
    try:
        for expr in constraints:
            parsed = parse_expr(
                expr,
                local_dict=locals_map,
                transformations=standard_transformations,
                evaluate=False,
            )
            parsed_constraints.append(parsed)
    except Exception as e:
        return f"Error parsing constraints: {e}"

    # Sampling loop
    n = len(parsed_constraints)
    funcs = [
        sp.lambdify(symbols, c, modules=["math", "numpy"])
        for c in parsed_constraints
    ]
    constraint_satisfied = np.zeros(n, dtype=int)
    for _ in range(samples):
        point = {}
        for i, sym in enumerate(symbols):
            typ = variable_type[i].lower()
            low, high = variable_bounds[i]
            if typ == "integer":
                value = random.randint(int(low), int(high))
            elif typ in ("real", "continuous"):
                value = random.uniform(low, high)
            elif typ in ("boolean", "logical"):
                value = random.choice([False, True])
            else:
                raise ValueError(
                    f"Unknown type {variable_type[i]} for variable {variable_name[i]}"
                )
            point[sym] = value

        # Evaluate all constraints at this point
        try:
            vals = [point[s] for s in symbols]
            cons_satisfaction = [
                bool(np.asarray(f(*vals)).all()) for f in funcs
            ]
            if all(cons_satisfaction):
                # Found a feasible point
                readable = {str(k): round(v, 3) for k, v in point.items()}
                return f"Feasible solution found: {readable}"
            else:
                constraint_satisfied += np.array(cons_satisfaction)
        except Exception as e:
            return f"Error evaluating constraint at point {point}: {e}"

    rates = constraint_satisfied / samples  # fraction satisfied per constraint
    order = np.argsort(rates)  # lowest (most violated) first

    lines = []
    for rank, idx in enumerate(order, start=1):
        expr_text = constraints[
            idx
        ]  # use the original string; easier to read than str(sympy_expr)
        sat = constraint_satisfied[idx]
        lines.append(
            f"[C{idx + 1}] {expr_text} — satisfied {sat:,}/{samples:,} ({sat / samples:.1%}), "
            f"violated {1 - sat / samples:.1%}"
        )

    return (
        f"No feasible solution found after {samples:,} samples. Most violated constraints (low→high satisfaction):\n "
        + "\n  ".join(lines)
    )
