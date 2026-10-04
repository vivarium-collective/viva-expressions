import logging

import numpy as np
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

ADDRESS = "local:!viva_expressions.processes.math_expression.MathExpressionStep"


def run_step(config, state, t_end=1.0):
    step = config.pop("_step_wiring")
    doc = {"state": {
        **state,
        "expr": {"_type": "step", "address": ADDRESS, "config": config, **step},
        "emitter": emitter_from_wires({k: [k] for k in [*state, "global_time"]}),
    }}
    composite = Composite(doc, core=allocate_core())
    composite.run(t_end)
    return composite, gather_emitter_results(composite)[("emitter",)]


def test_chained_expressions_with_clashing_names_in_composite(caplog):
    config = {
        "expressions": [
            {"out": "z", "expr": "a*N + sin(c)"},
            {"out": "w", "expr": "gamma*z"},
        ],
        "params": {"gamma": 3.0},
        "debug": True,
        "_step_wiring": {
            "inputs": {"a": ["a"], "N": ["N"], "c": ["c"]},
            "outputs": {"z": ["z"], "w": ["w"]},
        },
    }
    state = {"a": 2.0, "N": 1.5, "c": 0.7, "z": 0.0, "w": 0.0}
    with caplog.at_level(logging.INFO, logger="viva_expressions.processes.math_expression"):
        composite, _ = run_step(config, state)
    z = 2.0 * 1.5 + np.sin(0.7)
    np.testing.assert_allclose(composite.state["z"], z)
    np.testing.assert_allclose(composite.state["w"], 3.0 * z)
    assert "w = gamma*z" in caplog.text
