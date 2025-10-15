"""
Unified feasibility checker with heuristic pre-check and exact auto-routing.

Backends (imported lazily and used only if available):
- PySMT (cvc5/msat/yices/z3) for SMT-style logic, disjunctions, and nonlinear constructs.
- OR-Tools CP-SAT for strictly linear integer/boolean instances with integer coefficients.
- OR-Tools CBC (pywraplp) for linear MILP/LP (mixed real + integer, or pure LP).
- SciPy HiGHS (linprog) for pure continuous LP feasibility.

Install any subset you need:
    pip install pysmt && pysmt-install --cvc5        # or --z3/--msat/--yices
    pip install ortools
    pip install scipy
    pip install numpy

This file exposes a single LangChain tool: `feasibility_check_auto`.
"""

import math
import random
from typing import Annotated, Any, Dict, List, Optional, Tuple

import sympy as sp
from langchain_core.tools import tool
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

# Optional deps — handled gracefully if not installed
try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    from pysmt.shortcuts import GE as PS_GE
    from pysmt.shortcuts import GT as PS_GT
    from pysmt.shortcuts import LE as PS_LE
    from pysmt.shortcuts import LT as PS_LT
    from pysmt.shortcuts import And as PS_And
    from pysmt.shortcuts import Bool as PS_Bool
    from pysmt.shortcuts import Equals as PS_Eq
    from pysmt.shortcuts import Int as PS_Int
    from pysmt.shortcuts import Not as PS_Not
    from pysmt.shortcuts import Or as PS_Or
    from pysmt.shortcuts import Plus as PS_Plus
    from pysmt.shortcuts import Real as PS_Real
    from pysmt.shortcuts import Solver as PS_Solver
    from pysmt.shortcuts import Symbol as PS_Symbol
    from pysmt.shortcuts import Times as PS_Times
    from pysmt.typing import BOOL as PS_BOOL
    from pysmt.typing import INT as PS_INT
    from pysmt.typing import REAL as PS_REAL

    _HAS_PYSMT = True
except Exception:
    _HAS_PYSMT = False

try:
    from ortools.sat.python import cp_model as _cpsat

    _HAS_CPSAT = True
except Exception:
    _HAS_CPSAT = False

try:
    from ortools.linear_solver import pywraplp as _lp

    _HAS_LP = True
except Exception:
    _HAS_LP = False

try:
    from scipy.optimize import linprog as _linprog

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =========================
# Parsing & classification
# =========================


def _parse_constraints(
    constraints: List[str], variable_name: List[str]
) -> Tuple[List[sp.Symbol], List[sp.Expr]]:
    """Parse user constraint strings into SymPy expressions.

    Args:
        constraints: Constraint strings (e.g., "x + 2*y <= 5").
        variable_name: Names of variables referenced in constraints.

    Returns:
        A tuple (symbols, sympy_constraints) where `symbols` are the SymPy symbols for
        `variable_name` and `sympy_constraints` are parsed SymPy expressions.

    Raises:
        Exception: If SymPy fails to parse any constraint string.
    """
    syms = sp.symbols(variable_name)
    local_dict = {n: s for n, s in zip(variable_name, syms)}
    sympy_cons = [
        parse_expr(
            c,
            local_dict=local_dict,
            transformations=standard_transformations,
            evaluate=False,
        )
        for c in constraints
    ]
    return syms, sympy_cons


def _flatten_conjunction(expr: sp.Expr) -> Tuple[List[sp.Expr], bool]:
    """Flatten a conjunction into a list of conjuncts.

    If the expression is a chain of ANDs, returns all atomic conjuncts and `False`
    for the non-conjunctive flag. Otherwise, returns [expr] and `True` if a non-AND
    logical structure (e.g., OR/NOT) is detected.

    Args:
        expr: A SymPy boolean/relational expression.

    Returns:
        A tuple (conjuncts, nonconj) where `conjuncts` is a list of SymPy expressions
        and `nonconj` is True if expr contains non-conjunctive logic (e.g., Or/Not).
    """
    from sympy.logic.boolalg import And, Not, Or

    if isinstance(expr, And):
        out, stack = [], list(expr.args)
        while stack:
            a = stack.pop()
            if isinstance(a, And):
                stack.extend(a.args)
            else:
                out.append(a)
        return out, False

    is_rel = isinstance(expr, Relational)
    if is_rel or expr in (sp.true, sp.false):
        return [expr], False
    if isinstance(expr, (Or, Not)):
        return [expr], True
    return [expr], True


def _linear_relational(
    expr: sp.Expr, symbols: List[sp.Symbol]
) -> Optional[bool]:
    """Check whether a relational constraint is linear in the given symbols.

    Args:
        expr: SymPy expression (ideally a relational like <=, >=, ==, <, >).
        symbols: The variables with respect to which linearity is checked.

    Returns:
        True if linear, False if nonlinear, or None if `expr` is not a relational.

    Notes:
        Any presence of non-polynomial functions (e.g., sin, abs) returns False.
    """
    if not isinstance(expr, Relational):
        return None
    diff = sp.simplify(expr.lhs - expr.rhs)
    try:
        poly = sp.Poly(diff, *symbols, domain="QQ")
        return poly.total_degree() <= 1
    except Exception:
        return False


def _has_boolean_logic(expr: sp.Expr) -> bool:
    """Return True if the expression is a boolean combinator (And/Or/Not).

    Args:
        expr: SymPy expression.

    Returns:
        True if expr is an instance of And/Or/Not; False otherwise.
    """
    from sympy.logic.boolalg import And, Not, Or

    return isinstance(expr, (And, Or, Not))


def _classify(
    sympy_cons: List[sp.Expr], symbols: List[sp.Symbol], vtypes: List[str]
) -> Dict[str, Any]:
    """Classify the problem structure for routing.

    Args:
        sympy_cons: List of SymPy constraints.
        symbols: Variable symbols.
        vtypes: Variable types aligned with `symbols` (e.g., "real", "integer", "boolean").

    Returns:
        A dictionary with keys:
            - requires_smt: True if non-conjunctive or boolean logic present.
            - all_linear: True if all relational atoms are linear.
            - has_int, has_bool, has_real: Presence flags for variable domains.
            - only_conjunction: True if the top-level structure is pure conjunction.
    """
    requires_smt, only_conj, linear_ok = False, True, True
    for c in sympy_cons:
        conjuncts, nonconj = _flatten_conjunction(c)
        if nonconj:
            requires_smt, only_conj = True, False
        for a in conjuncts:
            if _has_boolean_logic(a):
                requires_smt = True
            is_lin = _linear_relational(a, symbols)
            if is_lin is False:
                linear_ok = False
            if is_lin is None and a not in (sp.true, sp.false):
                requires_smt = True

    vtypes_l = [t.lower() for t in vtypes]
    return {
        "requires_smt": requires_smt,
        "all_linear": linear_ok,
        "has_int": any(t in ("int", "integer") for t in vtypes_l),
        "has_bool": any(t in ("bool", "boolean", "logical") for t in vtypes_l),
        "has_real": any(
            t in ("real", "float", "double", "continuous") for t in vtypes_l
        ),
        "only_conjunction": only_conj,
    }


def _is_int_like(x: Optional[float], tol: float = 1e-9) -> bool:
    """Check if a value is (approximately) an integer.

    Args:
        x: Value to test, possibly None.
        tol: Absolute tolerance.

    Returns:
        True if x is within tol of an integer; False otherwise.
    """
    return x is not None and abs(x - round(x)) <= tol


def _coeffs_linear(
    expr: sp.Expr, symbols: List[sp.Symbol]
) -> Tuple[Dict[str, float], float]:
    """Extract linear coefficients and constant term of an expression.

    The expression is interpreted as:
        expr == sum_i coeff_i * x_i + const

    Args:
        expr: SymPy expression assumed linear in `symbols`.
        symbols: Variable symbols.

    Returns:
        A tuple (coeffs, const), where `coeffs[name]` is the coefficient of the
        variable `name`, and `const` is the constant term.

    Raises:
        ValueError: If non-linearity is detected via second derivatives.
    """
    coeffs: Dict[str, float] = {}
    for s in symbols:
        if sp.simplify(sp.diff(expr, s, 2)) != 0:
            raise ValueError("Non-linear term detected.")
        c = float(sp.N(sp.diff(expr, s)))
        if abs(c) > 0.0:
            coeffs[str(s)] = c
    const = float(sp.N(expr.subs({s: 0 for s in symbols})))
    return coeffs, const


def _all_int_coeffs(
    coeffs: Dict[str, float], const: float, tol: float = 1e-9
) -> bool:
    """Return True if all coefficients and the constant are integer-like.

    Args:
        coeffs: Mapping from variable name to coefficient.
        const: Constant term.
        tol: Integer-likeness tolerance.

    Returns:
        True if every coefficient and `const` is within `tol` of an integer.
    """
    return all(_is_int_like(v, tol) for v in coeffs.values()) and _is_int_like(
        const, tol
    )


# =========================
# Heuristic feasibility
# =========================


def _rand_unif(lo: Optional[float], hi: Optional[float], R: float) -> float:
    """Sample a uniform random real value within [lo, hi], with fallback radius.

    Args:
        lo: Lower bound or None for unbounded.
        hi: Upper bound or None for unbounded.
        R: Fallback radius if a side is unbounded.

    Returns:
        A float sampled uniformly within the determined interval.
    """
    lo = -R if lo is None or math.isinf(lo) else float(lo)
    hi = R if hi is None or math.isinf(hi) else float(hi)
    if lo > hi:
        lo, hi = hi, lo
    return random.uniform(lo, hi)


def _rand_int(lo: Optional[float], hi: Optional[float], R: int) -> int:
    """Sample a uniform random integer within [lo, hi], with fallback radius.

    Args:
        lo: Lower bound or None for unbounded.
        hi: Upper bound or None for unbounded.
        R: Fallback radius if a side is unbounded.

    Returns:
        An integer sampled uniformly within the determined interval.
    """
    lo = -R if lo is None or math.isinf(lo) else int(math.floor(lo))
    hi = R if hi is None or math.isinf(hi) else int(math.ceil(hi))
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _eval_relational(
    lhs_num: float, rhs_num: float, rel_op: str, tol: float
) -> bool:
    """Evaluate a relational comparison with tolerance.

    Args:
        lhs_num: Left-hand numeric value.
        rhs_num: Right-hand numeric value.
        rel_op: The relational operator string (one of '==','<=','<','>=','>').
        tol: Numeric tolerance for equality/inequality strictness.

    Returns:
        True if relation holds under tolerance; False otherwise.
    """
    d = lhs_num - rhs_num
    if rel_op == "==":
        return abs(d) <= tol
    if rel_op == "<=":
        return d <= tol
    if rel_op == "<":
        return d < -tol
    if rel_op == ">=":
        return d >= -tol
    if rel_op == ">":
        return d > tol
    return False


def _eval_bool_expr(e: sp.Expr, env: Dict[sp.Symbol, Any], tol: float) -> bool:
    """Evaluate a boolean/relational SymPy expression under an assignment.

    Args:
        e: SymPy boolean/relational expression.
        env: Mapping from symbol to Python numeric/bool value.
        tol: Numeric tolerance for evaluating relational operators.

    Returns:
        True if the expression is satisfied; False otherwise.
    """
    # Relational
    if isinstance(e, Relational):
        lhs = float(sp.N(e.lhs.subs(env)))
        rhs = float(sp.N(e.rhs.subs(env)))
        return _eval_relational(lhs, rhs, e.rel_op, tol)

    # Boolean logic
    from sympy.logic.boolalg import And, Not, Or

    if isinstance(e, And):
        return all(_eval_bool_expr(a, env, tol) for a in e.args)
    if isinstance(e, Or):
        return any(_eval_bool_expr(a, env, tol) for a in e.args)
    if isinstance(e, Not):
        return not _eval_bool_expr(e.args[0], env, tol)

    # Literals
    if e is sp.true:
        return True
    if e is sp.false:
        return False

    # Fallback: cast numeric to bool (non-zero -> True). Not generally recommended.
    try:
        return bool(sp.N(e.subs(env)))
    except Exception:
        return False


def _heuristic_feasible(
    sympy_cons: List[sp.Expr],
    symbols: List[sp.Symbol],
    variable_name: List[str],
    variable_type: List[str],
    variable_bounds: List[List[Optional[float]]],
    samples: int = 2000,
    seed: Optional[int] = None,
    tol: float = 1e-8,
    unbounded_radius_real: float = 1e3,
    unbounded_radius_int: int = 10**6,
) -> Optional[Dict[str, Any]]:
    """Try to find a satisfying assignment via randomized sampling.

    Args:
        sympy_cons: Parsed SymPy constraints.
        symbols: SymPy symbols aligned with `variable_name`.
        variable_name: Variable names.
        variable_type: Variable types aligned with `variable_name`.
        variable_bounds: Per-variable [low, high] bounds (None means unbounded side).
        samples: Number of random samples to try.
        seed: Random seed for reproducibility.
        tol: Tolerance for evaluating relational constraints.
        unbounded_radius_real: Sampling radius for unbounded real variables.
        unbounded_radius_int: Sampling radius for unbounded integer variables.

    Returns:
        A dict mapping variable names to sampled values if a witness is found; otherwise None.

    Notes:
        This does not prove infeasibility; it only returns a witness if one is found.
    """
    if seed is not None:
        random.seed(seed)
        if _HAS_NUMPY:
            np.random.seed(seed)

    sym_by_name = {str(s): s for s in symbols}

    for _ in range(samples):
        env: Dict[sp.Symbol, Any] = {}

        # Sample a point
        for n, t, (lo, hi) in zip(
            variable_name, variable_type, variable_bounds
        ):
            t_l = t.lower()
            if t_l in ("boolean", "bool", "logical"):
                val = bool(random.getrandbits(1))
            elif t_l in ("integer", "int"):
                val = _rand_int(lo, hi, unbounded_radius_int)
            else:
                val = _rand_unif(lo, hi, unbounded_radius_real)
            env[sym_by_name[n]] = val

        # Check all constraints
        ok = True
        for c in sympy_cons:
            if not _eval_bool_expr(c, env, tol):
                ok = False
                break
        if ok:
            return {n: env[sym_by_name[n]] for n in variable_name}
    return None


# =========================
# Exact backends
# =========================


def _solve_with_pysmt(
    sympy_cons: List[sp.Expr],
    symbols: List[sp.Symbol],
    variable_name: List[str],
    variable_type: List[str],
    variable_bounds: List[List[Optional[float]]],
    solver_name: str = "cvc5",
) -> str:
    """Solve via PySMT (SMT backends like cvc5/msat/yices/z3).

    Args:
        sympy_cons: Parsed SymPy constraints (may include boolean logic).
        symbols: SymPy symbols aligned with `variable_name`.
        variable_name: Variable names.
        variable_type: Variable types ("real", "integer", "boolean").
        variable_bounds: Per-variable [low, high] bounds (None for unbounded).
        solver_name: PySMT backend name ("cvc5", "msat", "yices", "z3").

    Returns:
        A formatted string with status and (if SAT) an example model.

    Raises:
        ValueError: If an unknown variable type is encountered.
    """
    if not _HAS_PYSMT:
        return "PySMT not installed. `pip install pysmt` and run `pysmt-install --cvc5` (or other backend)."

    # Build PySMT symbols
    ps_vars: Dict[str, Any] = {}
    for n, t in zip(variable_name, variable_type):
        t_l = t.lower()
        if t_l in ("integer", "int"):
            ps_vars[n] = PS_Symbol(n, PS_INT)
        elif t_l in ("real", "float", "double", "continuous"):
            ps_vars[n] = PS_Symbol(n, PS_REAL)
        elif t_l in ("boolean", "bool", "logical"):
            ps_vars[n] = PS_Symbol(n, PS_BOOL)
        else:
            raise ValueError(f"Unknown type: {t}")

    sym2ps = {s: ps_vars[n] for n, s in zip(variable_name, symbols)}

    def conv(e: sp.Expr):
        """Convert a SymPy expression to a PySMT node."""
        if isinstance(e, sp.Symbol):
            return sym2ps[e]
        if isinstance(e, sp.Integer):
            return PS_Int(int(e))
        if isinstance(e, (sp.Rational, sp.Float)):
            return PS_Real(float(e))
        if isinstance(e, sp.Eq):
            return PS_Eq(conv(e.lhs), conv(e.rhs))
        if isinstance(e, sp.Le):
            return PS_LE(conv(e.lhs), conv(e.rhs))
        if isinstance(e, sp.Lt):
            return PS_LT(conv(e.lhs), conv(e.rhs))
        if isinstance(e, sp.Ge):
            return PS_GE(conv(e.lhs), conv(e.rhs))
        if isinstance(e, sp.Gt):
            return PS_GT(conv(e.lhs), conv(e.rhs))
        from sympy.logic.boolalg import And, Not, Or

        if isinstance(e, And):
            return PS_And(*[conv(a) for a in e.args])
        if isinstance(e, Or):
            return PS_Or(*[conv(a) for a in e.args])
        if isinstance(e, Not):
            return PS_Not(conv(e.args[0]))
        if e is sp.true:
            return PS_Bool(True)
        if e is sp.false:
            return PS_Bool(False)
        if isinstance(e, sp.Add):
            terms = [conv(a) for a in e.args]
            out = terms[0]
            for t in terms[1:]:
                out = PS_Plus(out, t)
            return out
        if isinstance(e, sp.Mul):
            terms = [conv(a) for a in e.args]
            out = terms[0]
            for t in terms[1:]:
                out = PS_Times(out, t)
            return out
        raise ValueError(f"Unsupported function for PySMT conversion: {e}")

    # Append bounds as assertions
    ps_all = []
    for (n, t), (lo, hi) in zip(
        zip(variable_name, variable_type), variable_bounds
    ):
        v = ps_vars[n]
        t_l = t.lower()
        if t_l in ("boolean", "bool", "logical"):
            continue
        if lo is not None:
            ps_all.append(
                PS_LE(
                    PS_Real(float(lo)) if t_l != "integer" else PS_Int(int(lo)),
                    v,
                )
            )
        if hi is not None:
            ps_all.append(
                PS_LE(
                    v,
                    PS_Real(float(hi)) if t_l != "integer" else PS_Int(int(hi)),
                )
            )
    for c in sympy_cons:
        ps_all.append(conv(c))

    with PS_Solver(name=solver_name) as s:
        if ps_all:
            s.add_assertion(PS_And(ps_all))
        res = s.solve()
        if res:
            model = {n: str(s.get_value(ps_vars[n])) for n in variable_name}
            return f"[backend=pysmt:{solver_name}] Feasible. Example model: {model}"
        return f"[backend=pysmt:{solver_name}] Infeasible."


def _solve_with_cpsat_integer_boolean(
    conjuncts: List[sp.Expr],
    symbols: List[sp.Symbol],
    variable_name: List[str],
    variable_type: List[str],
    variable_bounds: List[List[Optional[float]]],
) -> str:
    """Solve linear integer/boolean feasibility via OR-Tools CP-SAT.

    Args:
        conjuncts: A list of atomic conjuncts (linear relational constraints).
        symbols: Variable symbols.
        variable_name: Variable names.
        variable_type: Variable types; only integer/boolean are supported here.
        variable_bounds: Per-variable [low, high] bounds; None means unbounded side.

    Returns:
        A formatted string with status and example integer/boolean model if feasible.

    Notes:
        If non-integer coefficients/constants are detected, the function returns a
        message requesting routing to MILP/LP instead (CBC branch).
    """
    if not _HAS_CPSAT:
        return "OR-Tools CP-SAT not installed. `pip install ortools`."

    m = _cpsat.CpModel()
    name_to_var: Dict[str, Any] = {}

    # Create int/bool vars
    for n, t, (lo, hi) in zip(variable_name, variable_type, variable_bounds):
        t_l = t.lower()
        if t_l in ("boolean", "bool", "logical"):
            v = m.NewBoolVar(n)
        else:
            lo_i = int(lo) if lo is not None else -(10**9)
            hi_i = int(hi) if hi is not None else 10**9
            v = m.NewIntVar(lo_i, hi_i, n)
        name_to_var[n] = v

    # Add constraints
    for c in conjuncts:
        if isinstance(c, (sp.Eq, sp.Le, sp.Lt, sp.Ge, sp.Gt)):
            diff = sp.simplify(c.lhs - c.rhs)
            coeffs, const = _coeffs_linear(diff, symbols)
            if not _all_int_coeffs(coeffs, const):
                return "Detected non-integer coefficients/constant; routing to MILP/LP."
            expr = sum(
                int(round(coeffs.get(n, 0))) * name_to_var[n]
                for n in variable_name
            ) + int(round(const))
            if isinstance(c, sp.Eq):
                m.Add(expr == 0)
            elif isinstance(c, sp.Le):
                m.Add(expr <= 0)
            elif isinstance(c, sp.Ge):
                m.Add(expr >= 0)
            elif isinstance(c, sp.Lt):
                m.Add(expr <= -1)  # strict for integers
            elif isinstance(c, sp.Gt):
                m.Add(expr >= 1)
        elif c is sp.true:
            pass
        elif c is sp.false:
            m.Add(0 == 1)
        else:
            return "Non-relational/non-linear constraint; CP-SAT handles linear conjunctions only."

    solver = _cpsat.CpSolver()
    status = solver.Solve(m)
    if status in (_cpsat.OPTIMAL, _cpsat.FEASIBLE):
        model = {n: int(solver.Value(name_to_var[n])) for n in variable_name}
        return f"[backend=cp-sat] Feasible. Example solution: {model}"
    return "[backend=cp-sat] Infeasible."


def _solve_with_cbc_milp(
    conjuncts: List[sp.Expr],
    symbols: List[sp.Symbol],
    variable_name: List[str],
    variable_type: List[str],
    variable_bounds: List[List[Optional[float]]],
) -> str:
    """Solve linear MILP/LP feasibility via OR-Tools CBC (pywraplp).

    Args:
        conjuncts: A list of atomic conjuncts (linear relational constraints).
        symbols: Variable symbols.
        variable_name: Variable names.
        variable_type: Variable types; booleans will be modeled as {0,1} integers.
        variable_bounds: Per-variable [low, high] bounds; None means unbounded side.

    Returns:
        A formatted string with status and example model if feasible, or UNSAT.

    Raises:
        RuntimeError: If the CBC solver cannot be created (bad OR-Tools install).
    """
    if not _HAS_LP:
        return (
            "OR-Tools linear solver (CBC) not installed. `pip install ortools`."
        )

    solver = _lp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        return "Failed to create CBC solver. Ensure OR-Tools is properly installed."

    var_objs: Dict[str, Any] = {}
    for n, t, (lo, hi) in zip(variable_name, variable_type, variable_bounds):
        t_l = t.lower()
        lo_v = -_lp.Solver.infinity() if lo is None else float(lo)
        hi_v = _lp.Solver.infinity() if hi is None else float(hi)
        if t_l in ("boolean", "bool", "logical"):
            var = solver.IntVar(0, 1, n)
        elif t_l in ("integer", "int"):
            var = solver.IntVar(math.floor(lo_v), math.ceil(hi_v), n)
        else:
            var = solver.NumVar(lo_v, hi_v, n)
        var_objs[n] = var

    eps = 1e-9
    for c in conjuncts:
        if isinstance(c, (sp.Eq, sp.Le, sp.Lt, sp.Ge, sp.Gt)):
            diff = sp.simplify(c.lhs - c.rhs)
            coeffs, const = _coeffs_linear(diff, symbols)

            # Build bounds L <= sum(coeff*var) <= U, shifting by -const.
            if isinstance(c, sp.Eq):
                L = U = -const
            elif isinstance(c, sp.Le):
                L, U = -_lp.Solver.infinity(), -const
            elif isinstance(c, sp.Ge):
                L, U = -const, _lp.Solver.infinity()
            elif isinstance(c, sp.Lt):
                L, U = -_lp.Solver.infinity(), -const - eps
            elif isinstance(c, sp.Gt):
                L, U = -const + eps, _lp.Solver.infinity()

            ct = solver.RowConstraint(L, U, "")
            for n, v in var_objs.items():
                ct.SetCoefficient(v, coeffs.get(n, 0.0))

        elif c is sp.true:
            pass
        elif c is sp.false:
            ct = solver.RowConstraint(1, _lp.Solver.infinity(), "")
        else:
            return "Non-relational or non-linear constraint encountered; CBC supports linear conjunctions only."

    solver.Minimize(0)
    status = solver.Solve()
    if status in (_lp.Solver.OPTIMAL, _lp.Solver.FEASIBLE):
        model: Dict[str, Any] = {}
        int_like = {
            n
            for n, t in zip(variable_name, variable_type)
            if t.lower() in ("integer", "int", "boolean", "bool", "logical")
        }
        for n, var in var_objs.items():
            val = var.solution_value()
            model[n] = int(round(val)) if n in int_like else float(val)
        return f"[backend=cbc] Feasible. Example solution: {model}"
    if status == _lp.Solver.INFEASIBLE:
        return "[backend=cbc] Infeasible."
    return f"[backend=cbc] Solver status: {status}"


def _solve_with_highs_lp(
    conjuncts: List[sp.Expr],
    symbols: List[sp.Symbol],
    variable_name: List[str],
    variable_bounds: List[List[Optional[float]]],
) -> str:
    """Solve pure continuous LP feasibility via SciPy HiGHS.

    Args:
        conjuncts: A list of atomic conjuncts (linear relational constraints).
        symbols: Variable symbols.
        variable_name: Variable names (continuous).
        variable_bounds: Per-variable [low, high] bounds; None means unbounded side.

    Returns:
        A formatted string with status and example model if feasible, or a failure message.

    Notes:
        This route supports only continuous variables and linear relational constraints.
    """
    if not _HAS_SCIPY:
        return "SciPy not installed. `pip install scipy`."

    var_index = {n: i for i, n in enumerate(variable_name)}
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    eps = 1e-9

    for c in conjuncts:
        if not isinstance(c, (sp.Eq, sp.Le, sp.Lt, sp.Ge, sp.Gt)):
            if c is sp.true:
                continue
            if c is sp.false:
                return "[backend=highs] Infeasible."
            return "Only linear relational constraints supported by LP route."
        diff = sp.simplify(c.lhs - c.rhs)
        coeffs, const = _coeffs_linear(diff, symbols)
        row = [0.0] * len(variable_name)
        for n, v in coeffs.items():
            row[var_index[n]] = v
        if isinstance(c, sp.Eq):
            A_eq.append(row)
            b_eq.append(-const)
        elif isinstance(c, sp.Le):
            A_ub.append(row)
            b_ub.append(-const)
        elif isinstance(c, sp.Ge):
            A_ub.append([-v for v in row])
            b_ub.append(const)
        elif isinstance(c, sp.Lt):
            A_ub.append(row)
            b_ub.append(-const - eps)
        elif isinstance(c, sp.Gt):
            A_ub.append([-v for v in row])
            b_ub.append(const - eps)

    bounds = []
    for lo, hi in variable_bounds:
        lo_v = -math.inf if lo is None else float(lo)
        hi_v = math.inf if hi is None else float(hi)
        bounds.append((lo_v, hi_v))

    import numpy as np

    c = np.zeros(len(variable_name))
    res = _linprog(
        c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    if res.success:
        model = {n: float(res.x[i]) for i, n in enumerate(variable_name)}
        return f"[backend=highs] Feasible. Example solution: {model}"
    return f"[backend=highs] Infeasible or solver failed: {res.message}"


# =========================
# Router tool (with heuristic)
# =========================


@tool(parse_docstring=True)
def feasibility_check_auto(
    constraints: Annotated[
        List[str],
        "Constraint strings like 'x0 + 2*x1 <= 5' or '(x0<=3) | (x1>=2)'",
    ],
    variable_name: Annotated[List[str], "['x0','x1',...]"],
    variable_type: Annotated[List[str], "['real'|'integer'|'boolean', ...]"],
    variable_bounds: Annotated[
        List[List[Optional[float]]],
        "[(low, high), ...] (use None for unbounded)",
    ],
    prefer_smt_solver: Annotated[
        str, "SMT backend if needed: 'cvc5'|'msat'|'yices'|'z3'"
    ] = "cvc5",
    heuristic_enabled: Annotated[
        bool, "Run a fast randomized search first?"
    ] = True,
    heuristic_first: Annotated[
        bool, "Try heuristic before exact routing"
    ] = True,
    heuristic_samples: Annotated[int, "Samples for heuristic search"] = 2000,
    heuristic_seed: Annotated[Optional[int], "Seed for reproducibility"] = None,
    heuristic_unbounded_radius_real: Annotated[
        float, "Sampling range for unbounded real vars"
    ] = 1e3,
    heuristic_unbounded_radius_int: Annotated[
        int, "Sampling range for unbounded integer vars"
    ] = 10**6,
    numeric_tolerance: Annotated[
        float, "Tolerance for relational checks (Eq/Lt/Le/etc.)"
    ] = 1e-8,
) -> str:
    """Unified feasibility checker with heuristic pre-check and exact auto-routing.

    Performs an optional randomized feasibility search. If no witness is found (or the
    heuristic is disabled), the function auto-routes to an exact backend based on the
    detected problem structure (PySMT for SMT/logic/nonlinear, OR-Tools CP-SAT for
    linear integer/boolean, OR-Tools CBC for MILP/LP, or SciPy HiGHS for pure LP).

    Args:
        constraints: Constraint strings such as "x0 + 2*x1 <= 5" or "(x0<=3) | (x1>=2)".
        variable_name: Variable names, e.g., ["x0", "x1"].
        variable_type: Variable types aligned with `variable_name`. Each must be one of
            "real", "integer", or "boolean".
        variable_bounds: Per-variable [low, high] bounds aligned with `variable_name`.
            Use None to denote an unbounded side.
        prefer_smt_solver: SMT backend name used by PySMT ("cvc5", "msat", "yices", or "z3").
        heuristic_enabled: Whether to run the heuristic sampler.
        heuristic_first: If True, run the heuristic before exact routing; if False, run it after.
        heuristic_samples: Number of heuristic samples.
        heuristic_seed: Random seed for reproducibility.
        heuristic_unbounded_radius_real: Sampling radius for unbounded real variables.
        heuristic_unbounded_radius_int: Sampling radius for unbounded integer variables.
        numeric_tolerance: Tolerance used in relational checks (e.g., Eq, Lt, Le).

    Returns:
        A message indicating the chosen backend and the feasibility result. On success,
        includes an example model (assignment). On infeasibility, includes a short
        diagnostic or solver status.

    Raises:
        ValueError: If constraints cannot be parsed or an unsupported variable type is provided.
    """
    # 1) Parse
    try:
        symbols, sympy_cons = _parse_constraints(constraints, variable_name)
    except Exception as e:
        return f"Parse error: {e}"

    # 2) Heuristic (optional)
    if heuristic_enabled and heuristic_first:
        try:
            h_model = _heuristic_feasible(
                sympy_cons,
                symbols,
                variable_name,
                variable_type,
                variable_bounds,
                samples=heuristic_samples,
                seed=heuristic_seed,
                tol=numeric_tolerance,
                unbounded_radius_real=heuristic_unbounded_radius_real,
                unbounded_radius_int=heuristic_unbounded_radius_int,
            )
            if h_model is not None:
                return f"[backend=heuristic] Feasible (sampled witness). Example solution: {h_model}"
        except Exception:
            # Ignore heuristic issues and continue to exact route
            pass

    # 3) Classify & route
    info = _classify(sympy_cons, symbols, variable_type)

    # SMT needed or nonlinear / non-conj
    if info["requires_smt"] or not info["all_linear"]:
        res = _solve_with_pysmt(
            sympy_cons,
            symbols,
            variable_name,
            variable_type,
            variable_bounds,
            solver_name=prefer_smt_solver,
        )
        # Optional heuristic after exact if requested
        if (
            heuristic_enabled
            and not heuristic_first
            and any(
                kw in res.lower()
                for kw in ("unknown", "not installed", "unsupported", "failed")
            )
        ):
            h_model = _heuristic_feasible(
                sympy_cons,
                symbols,
                variable_name,
                variable_type,
                variable_bounds,
                samples=heuristic_samples,
                seed=heuristic_seed,
                tol=numeric_tolerance,
                unbounded_radius_real=heuristic_unbounded_radius_real,
                unbounded_radius_int=heuristic_unbounded_radius_int,
            )
            if h_model is not None:
                return f"[backend=heuristic] Feasible (sampled witness). Example solution: {h_model}"
        return res

    # Linear-only path: collect atomic conjuncts
    conjuncts: List[sp.Expr] = []
    for c in sympy_cons:
        atoms, _ = _flatten_conjunction(c)
        conjuncts.extend(atoms)

    has_int, has_bool, has_real = (
        info["has_int"],
        info["has_bool"],
        info["has_real"],
    )

    # Pure LP (continuous only)
    if not has_int and not has_bool and has_real:
        res = _solve_with_highs_lp(
            conjuncts, symbols, variable_name, variable_bounds
        )
        if "not installed" in res.lower():
            res = _solve_with_cbc_milp(
                conjuncts,
                symbols,
                variable_name,
                variable_type,
                variable_bounds,
            )
        if (
            heuristic_enabled
            and not heuristic_first
            and any(kw in res.lower() for kw in ("failed", "unknown"))
        ):
            h_model = _heuristic_feasible(
                sympy_cons,
                symbols,
                variable_name,
                variable_type,
                variable_bounds,
                samples=heuristic_samples,
                seed=heuristic_seed,
                tol=numeric_tolerance,
                unbounded_radius_real=heuristic_unbounded_radius_real,
                unbounded_radius_int=heuristic_unbounded_radius_int,
            )
            if h_model is not None:
                return f"[backend=heuristic] Feasible (sampled witness). Example solution: {h_model}"
        return res

    # All integer/boolean → CP-SAT first (if integer coefficients), else CBC MILP
    if (has_int or has_bool) and not has_real:
        res = _solve_with_cpsat_integer_boolean(
            conjuncts, symbols, variable_name, variable_type, variable_bounds
        )
        if (
            any(
                kw in res
                for kw in (
                    "routing to MILP/LP",
                    "handles linear conjunctions only",
                )
            )
            or "not installed" in res.lower()
        ):
            res = _solve_with_cbc_milp(
                conjuncts,
                symbols,
                variable_name,
                variable_type,
                variable_bounds,
            )
        return res

    # Mixed reals + integers → CBC MILP
    res = _solve_with_cbc_milp(
        conjuncts, symbols, variable_name, variable_type, variable_bounds
    )

    # Optional heuristic after exact (if backend missing/failing)
    if (
        heuristic_enabled
        and not heuristic_first
        and any(
            kw in res.lower() for kw in ("not installed", "failed", "status:")
        )
    ):
        h_model = _heuristic_feasible(
            sympy_cons,
            symbols,
            variable_name,
            variable_type,
            variable_bounds,
            samples=heuristic_samples,
            seed=heuristic_seed,
            tol=numeric_tolerance,
            unbounded_radius_real=heuristic_unbounded_radius_real,
            unbounded_radius_int=heuristic_unbounded_radius_int,
        )
        if h_model is not None:
            return f"[backend=heuristic] Feasible (sampled witness). Example solution: {h_model}"

    return res
