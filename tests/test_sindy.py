from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from viva_expressions.synthesize import synthesize
from viva_expressions.synthesize.sindy import SindyUnavailable, discover
from viva_expressions.synthesize.spec import load_spec, parse_spec

pytest.importorskip("pysindy")
EXAMPLES = Path(__file__).resolve().parents[1] / "systems" / "examples"
LV = {"c_x_x": 1.0, "c_x_x_y": -0.5, "c_y_y": -1.0, "c_y_x_y": 0.25}


def data_spec(tmp_path, t, columns, states, **extra):
    header = ",".join(["t", *columns])
    np.savetxt(tmp_path / "d.csv", np.column_stack([t, *columns.values()]),
               delimiter=",", header=header, comments="", fmt="%.12g")
    return parse_spec({
        "name": "d", "time": {"t_end": float(t[-1]), "n_points": len(t)},
        "variables": {"y": [{"name": v, "value": float(columns[v][0])} for v in states]},
        "data": {"path": "d.csv", "time_column": "t", "columns": {v: v for v in states}},
        **extra}, tmp_path)


def lv_series(noise=0.0):
    t = np.linspace(0, 20, 401)
    y = solve_ivp(lambda _, z: [z[0] - 0.5 * z[0] * z[1], -z[1] + 0.25 * z[0] * z[1]],
                  (0, 20), [2.0, 1.0], t_eval=t, method="DOP853", rtol=1e-11, atol=1e-12).y
    y = y * (1 + noise * np.random.default_rng(0).standard_normal(y.shape))
    return t, {"x": y[0], "y": y[1]}


def pattern(models, params):
    return next((m for m in models if set(m.params) == set(params)), None)


@pytest.mark.parametrize("noise, rel", [(0.0, 0.02), (0.01, 0.10)])
def test_lotka_volterra_structure_recovered_without_spurious_terms(tmp_path, noise, rel):
    t, cols = lv_series(noise)
    model = pattern(discover(data_spec(tmp_path, t, cols, ["x", "y"])), LV)
    assert model is not None
    for p, true in LV.items():
        assert model.initial[p] == pytest.approx(true, rel=rel)


def test_logistic_coefficients(tmp_path):
    r, K, t = 0.8, 5.0, np.linspace(0, 20, 401)
    n = K / (1 + (K - 0.1) / 0.1 * np.exp(-r * t))
    model = pattern(discover(data_spec(tmp_path, t, {"N": n}, ["N"])), ["c_N_N", "c_N_N2"])
    assert model.rhs == {"N": "c_N_N*N + c_N_N2*N**2"}
    assert model.initial["c_N_N"] == pytest.approx(r, rel=0.02)
    assert model.initial["c_N_N2"] == pytest.approx(-r / K, rel=0.02)


def test_unavailable_without_full_state_data(tmp_path):
    t, cols = lv_series()
    spec = data_spec(tmp_path, t, {"x": cols["x"]}, ["x"])
    two_states = parse_spec({
        "name": "partial", "time": {"t_end": 20, "n_points": 401},
        "variables": {"y": [{"name": "x", "value": 2.0}, {"name": "y", "value": 1.0}]},
        "data": {"path": "d.csv", "time_column": "t", "columns": {"x": "x"}},
        "candidates": [{"motif": "lotka_volterra", "bind": {"prey": "x", "predator": "y"}}]},
        tmp_path)
    with pytest.raises(SindyUnavailable, match="every state observed"):
        discover(two_states)
    assert discover(spec)  # the observed single state alone is fine


def test_lotka_volterra_example_end_to_end():
    spec = load_spec(EXAMPLES / "lotka_volterra.yml")
    result = synthesize(spec)
    assert result.ok  # exported, re-run in a real Composite, re-scored
    winner = result.ranked[0]
    # the predator-prey structure, whether it arrived as the motif or from SINDy
    if winner.model.origin == "sindy":
        assert set(winner.theta) == set(LV)
        np.testing.assert_allclose([winner.theta[p] for p in LV], list(LV.values()), rtol=0.02)
    else:
        assert winner.model.origin == "motif:lotka_volterra"
        np.testing.assert_allclose([winner.theta[p] for p in "abcd"], [1, 0.5, 1, 0.25], rtol=0.02)
    for v, obs in spec.data.columns.items():
        assert result.verification.data_rmse[v] < 0.01 * np.ptp(obs)
