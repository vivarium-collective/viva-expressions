from pathlib import Path

import numpy as np
import pytest

from viva_expressions.synthesize.spec import (
    CustomCandidate,
    MotifCandidate,
    Oscillation,
    SpecError,
    SteadyState,
    load_spec,
    parse_spec,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "systems" / "examples"


def minimal(**overrides):
    spec = {
        "name": "decay",
        "time": {"t_end": 10, "n_points": 101},
        "variables": {"y": [{"name": "x", "value": 1.0}]},
        "behavior": {"features": [
            {"kind": "steady_state", "var": "x", "value": 0.0, "tol": 0.01}]},
        "candidates": [{"name": "decay", "rhs": {"x": "-k*x"}, "params": {"k": [0, 5]}}],
    }
    spec.update(overrides)
    return spec


def errors_of(raw, base_dir="."):
    with pytest.raises(SpecError) as exc:
        parse_spec(raw, base_dir)
    return exc.value.errors


@pytest.mark.parametrize("name", ["logistic", "oscillator", "lotka_volterra"])
def test_examples_load(name):
    spec = load_spec(EXAMPLES / f"{name}.yml")
    assert spec.name == name and spec.candidates


def test_lotka_volterra_data_matches_csv():
    spec = load_spec(EXAMPLES / "lotka_volterra.yml")
    raw = np.genfromtxt(EXAMPLES / "lv.csv", delimiter=",", names=True)
    np.testing.assert_array_equal(spec.data.t, raw["t"])
    np.testing.assert_array_equal(spec.data.columns["x"], raw["prey"])
    np.testing.assert_array_equal(spec.data.columns["y"], raw["predator"])
    assert spec.data.columns["x"][0] == spec.states[0].value


def test_minimal_spec_parses_into_typed_objects():
    spec = parse_spec(minimal())
    assert spec.state_names == ("x",) and spec.input_names == ()
    assert spec.behavior.features == (SteadyState("x", 0.0, 0.01),)
    assert spec.candidates == (CustomCandidate("decay", {"x": "-k*x"}, {"k": (0.0, 5.0)}),)
    assert spec.fit.seed == 0 and spec.data is None
    np.testing.assert_allclose(spec.time.grid[[0, -1]], [0, 10])


def test_free_text_behavior_needs_data_or_features():
    errs = errors_of(minimal(behavior="it decays"))
    assert any(e.startswith("spec:") for e in errs)


def test_every_error_is_reported_at_once():
    raw = minimal(
        time={"t_end": -1, "n_points": 1},
        variables={"y": [{"name": "x", "value": 1.0}, {"name": "x", "value": 2.0}]},
        behavior={"features": [
            {"kind": "oscillation", "var": "z", "period": 2, "tol": 1.5},
            {"kind": "wobble", "var": "x"}]},
        candidates=[{"name": "bad", "rhs": {"x": "-k*x*q"}, "params": {"k": [5, 0]}}],
        fit={"seed": -1},
        extra=1,
    )
    errs = errors_of(raw)
    expected = ["extra:", "time.t_end", "time.n_points", "variables.y[1].name",
                "behavior.features[0].var", "behavior.features[0].tol",
                "behavior.features[1].kind", "candidates[0].params.k",
                "candidates[0].rhs.x: undeclared symbol(s) ['q']", "fit.seed"]
    for prefix in expected:
        assert any(e.startswith(prefix) for e in errs), (prefix, errs)
    # invalid sections are reported once, not again as missing
    assert not any(e.startswith(("spec:", "candidates:")) for e in errs), errs


def test_candidate_must_be_motif_xor_custom_and_bind_declared_vars():
    errs = errors_of(minimal(candidates=[
        {"motif": "logistic", "bind": {"x": "nope"}},
        {"motif": "logistic", "rhs": {"x": "x"}},
        {"name": "c", "rhs": {"x": "k"}, "params": {"x": [0, 1]}}]))
    assert any(e.startswith("candidates[0].bind.x") for e in errs)
    assert any(e.startswith("candidates[1]:") for e in errs)
    assert any(e.startswith("candidates[2].params.x") for e in errs)


def test_no_data_requires_candidates():
    errs = errors_of(minimal(candidates=None))
    assert any(e.startswith("candidates:") for e in errs)


def test_motif_candidate_and_oscillation_defaults():
    spec = parse_spec(minimal(
        time={"t_end": 20, "n_points": 201},
        variables={"y": [{"name": "x", "value": 1.0}, {"name": "v", "value": 0.0}]},
        behavior={"features": [{"kind": "oscillation", "var": "x", "period": 6.28, "tol": 0.05}]},
        candidates=[{"motif": "harmonic_oscillator", "bind": {"x": "x", "v": "v"}}]))
    assert spec.behavior.features == (Oscillation("x", 6.28, 0.05, None, True),)
    assert spec.candidates == (MotifCandidate("harmonic_oscillator", {"x": "x", "v": "v"}),)


def test_motif_param_overrides():
    spec = parse_spec(minimal(candidates=[
        {"motif": "logistic", "bind": {"x": "x"}, "params": {"K": [1, 10]}}]))
    assert spec.candidates[0].params == {"K": (1.0, 10.0)}
    errs = errors_of(minimal(candidates=[
        {"motif": "logistic", "bind": {"x": "x"}, "params": {"K": [10, 1]}}]))
    assert any(e.startswith("candidates[0].params.K") for e in errs)


def test_initial_value_outside_bounds_is_rejected():
    errs = errors_of(minimal(variables={"y": [{"name": "x", "value": -1.0, "bounds": [0, None]}]}))
    assert any(e.startswith("variables.y[0].value") for e in errs)


def test_data_errors(tmp_path):
    (tmp_path / "d.csv").write_text("t,a\n0,1\n2,2\n1,3\n")
    data = {"path": "d.csv", "time_column": "t", "columns": {"x": "a"}}
    errs = errors_of(minimal(data=data), tmp_path)
    assert any(e.startswith("data.time_column") for e in errs)

    errs = errors_of(minimal(data={**data, "columns": {"x": "missing"}}), tmp_path)
    assert any("not in d.csv header" in e for e in errs)

    (tmp_path / "late.csv").write_text("t,a\n0,1\n5,2\n11,3\n")
    errs = errors_of(minimal(data={**data, "path": "late.csv"}), tmp_path)
    assert any("must lie in [0, time.t_end=10" in e for e in errs)

    errs = errors_of(minimal(data={**data, "path": "nope.csv"}), tmp_path)
    assert any(e.startswith("data.path: file not found") for e in errs)


def test_data_without_candidates_is_valid(tmp_path):
    (tmp_path / "d.csv").write_text("t,a\n0,1\n1,0.5\n2,0.25\n")
    spec = parse_spec(minimal(candidates=None, behavior=None,
                              data={"path": "d.csv", "time_column": "t", "columns": {"x": "a"}}),
                      tmp_path)
    np.testing.assert_array_equal(spec.data.columns["x"], [1, 0.5, 0.25])


def test_invalid_yaml_is_a_spec_error(tmp_path):
    p = tmp_path / "bad.yml"
    p.write_text("name: [unclosed\n")
    with pytest.raises(SpecError, match="not valid YAML"):
        load_spec(p)


def test_oscillation_needs_two_periods_of_simulated_time():
    errs = errors_of(minimal(behavior={"features": [
        {"kind": "oscillation", "var": "x", "period": 6, "tol": 0.05}]}))
    assert any(e.startswith("behavior.features[0].period: time.t_end=10") for e in errs)
