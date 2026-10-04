import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from viva_expressions.composites import ode_document, run_document


def test_logistic_in_composite_matches_analytic_solution():
    r, K, n0 = 0.8, 5.0, 0.1
    doc = ode_document({"N": "r*N*(1 - N/K)"}, {"r": r, "K": K}, {"N": n0}, interval=0.1)
    t, series = run_document(doc, 20.0)
    assert len(t) == 201 and t[-1] == pytest.approx(20.0)
    analytic = K / (1 + (K - n0) / n0 * np.exp(-r * t))
    np.testing.assert_allclose(series["N"], analytic, atol=1e-6)


def test_lotka_volterra_with_input_matches_independent_solve_ivp():
    doc = ode_document(
        {"x": "a*x - b*x*y + u", "y": "-c*y + d*x*y"},
        {"a": 1.0, "b": 0.5, "c": 1.0, "d": 0.25},
        {"x": 2.0, "y": 1.0}, inputs={"u": 0.1}, interval=0.25)
    t, series = run_document(doc, 15.0)

    def lv(_, z):
        x, y = z
        return [x - 0.5 * x * y + 0.1, -y + 0.25 * x * y]

    ref = solve_ivp(lv, (0, 15.0), [2.0, 1.0], t_eval=t, method="DOP853",
                    rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(series["x"], ref.y[0], atol=1e-5)
    np.testing.assert_allclose(series["y"], ref.y[1], atol=1e-5)
    np.testing.assert_allclose(series["u"], 0.1)


def test_document_round_trips_through_json():
    doc = ode_document({"x": "-k*x"}, {"k": 0.5}, {"x": 1.0}, interval=0.5)
    t, series = run_document(json.loads(json.dumps(doc)), 4.0)
    np.testing.assert_allclose(series["x"], np.exp(-0.5 * t), atol=1e-7)


def test_integration_failure_raises_instead_of_returning_nan():
    # dx/dt = x^2 from x0 = 1 blows up at t = 1; LSODA alone stalls here forever
    doc = ode_document({"x": "x**2"}, {}, {"x": 1.0}, interval=2.0)
    with pytest.raises(RuntimeError, match="exceeded max_abs"):
        run_document(doc, 2.0)
