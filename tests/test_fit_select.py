from pathlib import Path

import numpy as np
import pytest

from viva_expressions.composites import ode_document, run_document
from viva_expressions.synthesize.behavior import settling_time
from viva_expressions.synthesize.fit import fit_model
from viva_expressions.synthesize.motifs import resolve
from viva_expressions.synthesize.select import rank
from viva_expressions.synthesize.spec import load_spec, parse_spec

EXAMPLES = Path(__file__).resolve().parents[1] / "systems" / "examples"


def fit_all(spec):
    return rank([fit_model(resolve(c, spec), spec) for c in spec.candidates])


@pytest.fixture(scope="module")
def logistic():
    spec = load_spec(EXAMPLES / "logistic.yml")
    return spec, fit_all(spec)


def test_logistic_winner_behaves_in_a_real_composite(logistic):
    spec, ranked = logistic
    winner = ranked[0]
    assert winner.feasible
    doc = ode_document(winner.model.rhs, winner.theta,
                       {v.name: v.value for v in spec.states}, interval=0.1)
    t, series = run_document(doc, spec.time.t_end)
    n = series["N"]
    assert n[-1] == pytest.approx(5.0, rel=0.02)
    assert np.all(np.diff(n) >= 0)
    assert abs(settling_time(t, n, 0.02) - 12.0) <= 3.0


def test_exponential_growth_is_infeasible_and_ranked_last(logistic):
    _, ranked = logistic
    assert ranked[-1].model.origin == "motif:exponential_growth"
    assert not ranked[-1].feasible


def test_steady_state_alone_leaves_rate_unidentifiable():
    spec = parse_spec({
        "name": "ss_only", "time": {"t_end": 30, "n_points": 301},
        "variables": {"y": [{"name": "N", "value": 0.1}]},
        "behavior": {"features": [{"kind": "steady_state", "var": "N", "value": 5.0, "tol": 0.1}]},
        "candidates": [{"motif": "logistic", "bind": {"x": "N"},
                        "params": {"r": [0.5, 5.0], "K": [1.0, 10.0]}}]})
    result = fit_model(resolve(spec.candidates[0], spec), spec)
    assert result.feasible
    assert result.theta["K"] == pytest.approx(5.0, abs=0.1)
    assert result.non_identifiable == ["r"]


def test_oscillator_frequency_and_parsimony():
    spec = load_spec(EXAMPLES / "oscillator.yml")
    ranked = fit_all(spec)
    assert all(r.feasible for r in ranked)
    assert ranked[0].model.origin == "motif:harmonic_oscillator"  # fewer params wins
    for r in ranked:
        assert r.theta["omega"] == pytest.approx(1.0, rel=0.05)


def test_single_unconstrained_parameter_is_flagged():
    # Regression: a purely relative SVD threshold can never flag the largest
    # direction, so a lone parameter the spec doesn't constrain looked determined.
    spec = parse_spec({
        "name": "decay", "time": {"t_end": 30, "n_points": 301},
        "variables": {"y": [{"name": "x", "value": 1.0}]},
        "behavior": {"features": [{"kind": "steady_state", "var": "x", "value": 0, "tol": 0.1}]},
        "candidates": [{"motif": "exponential_decay", "bind": {"x": "x"},
                        "params": {"k": [0.5, 10]}}]})
    result = fit_model(resolve(spec.candidates[0], spec), spec)
    assert result.feasible and result.non_identifiable == ["k"]
