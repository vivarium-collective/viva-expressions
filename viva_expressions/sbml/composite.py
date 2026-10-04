"""Render an imported SBML model as a workspace ``*.composite.yaml`` spec.

The spec follows this workspace's composite conventions (registry addresses,
stores under ``stores``, a ``JSONEmitter``), and exposes every model
parameter and initial value as a typed composite ``parameter``, so studies
can vary them through the workbench like any hand-written composite.
"""
from viva_expressions.expressions import CONSTANTS, identifiers
from viva_expressions.sbml.read import OdeModel

SOLVER = {
    "method": ("string", "LSODA", "scipy solve_ivp method (LSODA, RK45, DOP853, Radau, BDF)."),
    "rtol": ("float", 1e-8, "Solver relative tolerance."),
    "atol": ("float", 1e-10, "Solver absolute tolerance."),
}


def _unique(name: str, taken: set[str]) -> str:
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def exact_binary(interval: float, max_exponent: int = 30) -> bool:
    """Whether ``interval`` is a dyadic fraction (k / 2**n), so repeated additions
    of it to the clock are exact in floating point."""
    _, denominator = float(interval).as_integer_ratio()
    return denominator & (denominator - 1) == 0 and denominator <= 2 ** max_exponent


def composite_spec(model: OdeModel, name: str, interval: float,
                   description: str = "", source: str = "") -> dict:
    """A composite spec dict (dump it with ``yaml.safe_dump``).

    A time-dependent model (``model.time_var``) needs an interval that is exact
    in binary (0.5, 0.25, 0.125, ...). A composite spec cannot carry
    ``global_time_precision`` (process-bigraph's CompositeSpec drops it), so the
    clock is a running float sum of intervals. With e.g. 0.1 that sum drifts
    (10 steps give 0.9999999999999999), and a switch in time that falls on a
    sample point lands on the wrong side.
    """
    if model.time_var and not exact_binary(interval):
        raise ValueError(
            f"interval {interval!r} is not exact in binary; a time-dependent model "
            "needs one (e.g. 0.5, 0.25, 0.125) so the model clock stays exact")
    # model names are kept; solver and initial-value parameters take free names
    taken = {*model.params, *model.initial, *model.assignments,
             *([model.time_var] if model.time_var else [])}
    solver = {k: _unique(k, taken) for k in SOLVER}
    initial_param = {x: _unique(f"{x}_0", taken) for x in model.initial}
    parameters = {
        p: {"type": "float", "default": float(v),
            "description": f"SBML {model.kinds.get(p, 'parameter')} "
                           f"{model.sbml_ids.get(p, p)!r} (held constant)."}
        for p, v in model.params.items()}
    parameters.update({
        initial_param[x]: {"type": "float", "default": float(v),
                           "description": f"Initial value of {x} ({model.kinds[x]})."}
        for x, v in model.initial.items()})
    parameters.update({solver[k]: {"type": t, "default": d, "description": text}
                       for k, (t, d, text) in SOLVER.items()})

    stored = [*model.initial, *model.assignments]
    state = {
        "ode": {
            "_type": "process",
            "address": "local:OdeProcess",
            "config": {
                "rhs": dict(model.rhs),
                "params": {p: f"${{{p}}}" for p in model.params},
                "state_vars": list(model.initial),
                "input_vars": [],
                **({"time_var": model.time_var} if model.time_var else {}),
                **{k: f"${{{solver[k]}}}" for k in SOLVER},
            },
            "interval": float(interval),
            "inputs": {**{x: ["stores", x] for x in model.initial},
                       **({model.time_var: ["global_time"]} if model.time_var else {})},
            "outputs": {x: ["stores", x] for x in model.initial},
        },
        "stores": {**{x: f"${{{initial_param[x]}}}" for x in model.initial},
                   **{a: 0.0 for a in model.assignments}},
    }
    requires = ["OdeProcess"]
    if model.assignments:
        needs = sorted(set().union(*map(identifiers, model.assignments.values()))
                       - set(model.assignments) - set(model.params) - CONSTANTS)
        state["assignments"] = {
            "_type": "step",
            "address": "local:MathExpressionStep",
            "config": {
                "expressions": [{"out": a, "expr": e} for a, e in model.assignments.items()],
                "params": {p: f"${{{p}}}" for p in model.params},
            },
            "inputs": {x: (["global_time"] if x == model.time_var else ["stores", x])
                       for x in needs},
            "outputs": {a: ["stores", a] for a in model.assignments},
        }
        requires.append("MathExpressionStep")
    clock = _unique("time", set(stored))   # the emitted global time, never a model variable
    state["emitter"] = {
        "_type": "step",
        "address": "local:JSONEmitter",
        "config": {"emit": {**{x: "float" for x in stored}, clock: "float"}},
        "inputs": {**{x: ["stores", x] for x in stored}, clock: ["global_time"]},
    }
    text = description or f"SBML model {model.model_id!r} imported into OdeProcess."
    if source:
        text += f" Source: {source}."
    if model.notes:
        text += " " + " ".join(model.notes)
    return {"name": name, "description": text, "tags": ["ode", "expressions", "sbml-import"],
            "requires": {"processes": requires}, "parameters": parameters, "state": state}
