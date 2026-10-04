"""Parse and compile sympy expressions shared by every viva-expressions process.

Identifiers are bound to plain ``sp.Symbol`` objects explicitly, so model names
such as ``N``, ``gamma``, ``beta``, ``E``, ``I``, ``S`` and ``Q`` stay
symbols instead of resolving to sympy built-ins (bare ``sp.sympify("N*x")``
raises ``TypeError``). Names used as calls (``sin(x)``, ``exp(x)``) stay sympy
functions, and ``pi`` stays the constant unless it is declared as a name.
"""
import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

TRANSFORMATIONS = standard_transformations + (convert_xor,)
CONSTANTS = frozenset({"pi"})


def identifiers(expr: str) -> set[str]:
    """Return the variable names in ``expr``, excluding called function names."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"cannot parse expression {expr!r}: {e.msg}") from e
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - called


def parse(expr: str, allowed: Iterable[str] | None = None) -> sp.Basic:
    """Parse ``expr`` with every identifier bound to a plain symbol.

    If ``allowed`` is given, any identifier outside it (other than a constant
    such as ``pi``) raises ``ValueError``.
    """
    allowed = None if allowed is None else set(allowed)
    names = identifiers(expr)
    if allowed is None:
        names -= CONSTANTS
    else:
        names -= CONSTANTS - allowed
        unknown = names - allowed
        if unknown:
            raise ValueError(
                f"undeclared symbol(s) {sorted(unknown)} in {expr!r}")
    local = {n: sp.Symbol(n) for n in names}
    return parse_expr(expr, local_dict=local, transformations=TRANSFORMATIONS)


def substitute(expr: sp.Basic, values: dict[str, float]) -> sp.Basic:
    """Substitute named values, keyed by symbol rather than by string."""
    return expr.subs({sp.Symbol(k): v for k, v in values.items()})


@dataclass(frozen=True)
class CompiledRHS:
    """A compiled ODE right-hand side ``dy/dt = f(y, u, p)``.

    ``f`` and ``jac`` take positional vectors ordered as ``state``, ``inputs``
    and ``params``, and return arrays of shape ``(n,)`` and ``(n, n)``.
    """
    state: tuple[str, ...]
    inputs: tuple[str, ...]
    params: tuple[str, ...]
    exprs: dict[str, sp.Basic]
    f: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    jac: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def compile_rhs(
    rhs: dict[str, str],
    state: Iterable[str],
    inputs: Iterable[str] = (),
    params: Iterable[str] = (),
) -> CompiledRHS:
    """Compile ``{var: "expression"}`` into a vectorized RHS and its Jacobian."""
    state, inputs, params = tuple(state), tuple(inputs), tuple(params)
    missing = set(state) - set(rhs)
    extra = set(rhs) - set(state)
    if missing or extra:
        raise ValueError(
            f"rhs keys must equal the state variables: "
            f"missing {sorted(missing)}, unexpected {sorted(extra)}")
    clash = (set(state) & set(inputs)) | (set(state) & set(params)) \
        | (set(inputs) & set(params))
    if clash:
        raise ValueError(f"names declared more than once: {sorted(clash)}")

    allowed = state + inputs + params
    exprs = {v: parse(rhs[v], allowed) for v in state}
    y, u, p = ([sp.Symbol(n) for n in names] for names in (state, inputs, params))
    vector = sp.Matrix([exprs[v] for v in state])
    f_raw = sp.lambdify([y, u, p], vector, modules="numpy")
    jac_raw = sp.lambdify([y, u, p], vector.jacobian(y), modules="numpy")
    n = len(state)

    def f(yv, uv, pv):
        return np.asarray(f_raw(yv, uv, pv), dtype=float).reshape(n)

    def jac(yv, uv, pv):
        return np.asarray(jac_raw(yv, uv, pv), dtype=float).reshape(n, n)

    return CompiledRHS(state, inputs, params, exprs, f, jac)
