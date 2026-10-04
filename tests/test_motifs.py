import numpy as np
import pytest

from viva_expressions.expressions import compile_rhs
from viva_expressions.synthesize.motifs import MOTIFS, describe, instantiate, resolve
from viva_expressions.synthesize.spec import CustomCandidate, MotifCandidate, parse_spec


@pytest.mark.parametrize("name", sorted(MOTIFS))
def test_every_motif_compiles_and_evaluates_finite(name):
    m = MOTIFS[name]
    model = instantiate(m, {r: r for r in m.roles})
    rhs = compile_rhs(model.rhs, m.state_roles, m.input_roles, list(model.params))
    p = np.array([(lo + hi) / 2 for lo, hi in model.params.values()])
    out = rhs.f(np.ones(len(m.state_roles)), np.ones(len(m.input_roles)), p)
    assert out.shape == (len(m.state_roles),) and np.all(np.isfinite(out))


def test_instantiate_renames_roles_simultaneously_and_prefixes_params():
    m = MOTIFS["harmonic_oscillator"]
    model = instantiate(m, {"x": "v", "v": "x"}, prefix="o_")  # swapped names
    assert model.rhs == {"v": "x", "x": "-o_omega**2*v"}
    assert model.params == {"o_omega": (0.0, 10.0)}


def test_instantiate_rejects_bad_bindings_and_overrides():
    m = MOTIFS["logistic"]
    with pytest.raises(ValueError, match="needs bind for roles"):
        instantiate(m, {"y": "N"})
    with pytest.raises(ValueError, match="has no params"):
        instantiate(m, {"x": "N"}, overrides={"q": (0, 1)})
    with pytest.raises(ValueError, match="clash"):
        instantiate(m, {"x": "K"})


def test_hill_input_role_binds_to_an_input():
    spec = parse_spec({
        "name": "induced", "time": {"t_end": 10, "n_points": 11},
        "variables": {"x": [{"name": "ligand", "value": 1.0}],
                      "y": [{"name": "gfp", "value": 0.0}]},
        "behavior": {"features": [{"kind": "steady_state", "var": "gfp", "value": 1, "tol": 0.1}]},
        "candidates": [{"motif": "hill_activation", "bind": {"x": "gfp", "s": "ligand"},
                        "params": {"n": [1, 2]}}]})
    model = resolve(spec.candidates[0], spec)
    assert model.rhs == {"gfp": "beta*ligand**n/(Kd**n + ligand**n) - gamma*gfp"}
    assert model.params["n"] == (1.0, 2.0) and model.origin == "motif:hill_activation"


def test_resolve_requires_equations_for_exactly_the_spec_states():
    spec = parse_spec({
        "name": "two", "time": {"t_end": 10, "n_points": 11},
        "variables": {"y": [{"name": "a", "value": 1.0}, {"name": "b", "value": 1.0}]},
        "behavior": {"features": [{"kind": "steady_state", "var": "a", "value": 0, "tol": 0.1}]},
        "candidates": [{"motif": "exponential_decay", "bind": {"x": "a"}}]})
    with pytest.raises(ValueError, match="spec's states are"):
        resolve(spec.candidates[0], spec)
    with pytest.raises(ValueError, match="unknown motif"):
        resolve(MotifCandidate("nope", {}), spec)
    model = resolve(CustomCandidate("c", {"a": "-k*a", "b": "k*a"}, {"k": (0.0, 1.0)}), spec)
    assert model.origin == "custom"


def test_describe_lists_every_motif():
    text = describe()
    assert all(name in text for name in MOTIFS)


def test_prefix_resolves_parameter_clash_with_a_state():
    spec = parse_spec({
        "name": "k_state", "time": {"t_end": 10, "n_points": 11},
        "variables": {"y": [{"name": "K", "value": 0.1}]},
        "behavior": {"features": [{"kind": "steady_state", "var": "K", "value": 5, "tol": 0.1}]},
        "candidates": [{"motif": "logistic", "bind": {"x": "K"}},
                       {"motif": "logistic", "bind": {"x": "K"}, "prefix": "g_"}]})
    with pytest.raises(ValueError, match="give the candidate a `prefix`"):
        resolve(spec.candidates[0], spec)
    model = resolve(spec.candidates[1], spec)
    assert model.rhs == {"K": "g_r*K*(1 - K/g_K)"}
