"""Build and run process-bigraph documents for viva-expressions processes."""
import numpy as np
from process_bigraph import Composite, allocate_core
from process_bigraph.composite import interval_time_precision
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

ODE_ADDRESS = "local:!viva_expressions.processes.ode.OdeProcess"


def ode_document(
    rhs: dict[str, str],
    params: dict[str, float],
    initial: dict[str, float],
    inputs: dict[str, float] | None = None,
    interval: float = 0.1,
    **solver,
) -> dict:
    """A self-contained, JSON-serializable document running one OdeProcess.

    ``initial`` holds the state variables in integration order; ``inputs``
    holds exogenous values. Both are wired to same-named top-level stores and
    emitted together with ``global_time``. ``global_time_precision`` rounds
    accumulated time to the interval's decimals, so a run of ``t_end`` lands on
    exactly ``t_end / interval`` steps instead of losing the last one to float drift.
    """
    inputs = inputs or {}
    state_vars, input_vars = list(initial), list(inputs)
    ports = state_vars + input_vars
    return {"global_time_precision": interval_time_precision(float(interval)), "state": {
        **{k: float(v) for k, v in initial.items()},
        **{k: float(v) for k, v in inputs.items()},
        "ode": {
            "_type": "process",
            "address": ODE_ADDRESS,
            "config": {
                "rhs": dict(rhs),
                "params": {k: float(v) for k, v in params.items()},
                "state_vars": state_vars,
                "input_vars": input_vars,
                **solver,
            },
            "interval": float(interval),
            "inputs": {v: [v] for v in ports},
            "outputs": {v: [v] for v in state_vars},
        },
        "emitter": emitter_from_wires(
            {v: [v] for v in [*ports, "global_time"]}),
    }}


def run_document(doc: dict, t_end: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run ``doc`` to ``t_end``; return emitted times and per-variable series."""
    composite = Composite(doc, core=allocate_core())
    composite.run(t_end)
    rows = gather_emitter_results(composite)[("emitter",)]
    t = np.array([row["global_time"] for row in rows], dtype=float)
    keys = [k for k in rows[0] if k != "global_time"]
    return t, {k: np.array([row[k] for row in rows], dtype=float) for k in keys}
