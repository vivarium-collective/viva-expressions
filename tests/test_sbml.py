"""SBML <-> expressions conversion, checked against libroadrunner.

The oracle is libroadrunner (via tellurium's stack) running the *original*
SBML. It implements SBML semantics (species amounts vs concentrations,
compartments, rules, function definitions) independently of the importer, so
agreement is evidence the translation is faithful, not that it agrees with itself.

Agreement metric: per variable, ``max_t |ours - ref| / max_t |ref|`` (scale
floored at 1e-10); any non-finite value fails. Threshold ``AGREE`` = 1e-6,
fixed in advance from the solver tolerances (both sides at rtol 1e-12,
atol 1e-14): semantic errors are O(1) of a variable's scale, integrator error
is orders smaller. ``test_error_converges_with_tolerance`` shows the remaining
difference is integrator error, and ``test_negative_control`` that the check
detects a one-term error.
"""
import hashlib
import math
from pathlib import Path

import libsbml
import numpy as np
import pytest
import roadrunner
import yaml
from process_bigraph import allocate_core
from process_bigraph.composite_spec import substitute_parameters
from scipy.special import lambertw

from viva_expressions.composites import ode_document, run_document
from viva_expressions.processes.math_expression import MathExpressionStep
from viva_expressions.sbml import (
    UnsupportedSBML,
    document,
    ode_from_state,
    read_sbml,
    write_sbml,
)

DATA = Path(__file__).parent / "data" / "biomodels"
COMPOSITES = Path(__file__).parent.parent / "viva_expressions" / "composites"
AGREE = 1e-6
TOL = {"rtol": 1e-12, "atol": 1e-14}

# Pinned BioModels (scripts/fetch_biomodels.py), each chosen for a feature:
IMPORTABLE = {
    "BIOMD0000000005": "boundary species, assignment rules",
    "BIOMD0000000010": "local parameters, MAPK oscillator",
    "BIOMD0000000012": "hasOnlySubstanceUnits, assignment rules (repressilator)",
    "BIOMD0000000041": "two compartments of different sizes",
    "BIOMD0000000254": "rate rules only, no reactions",
    "BIOMD0000000400": "rate rules, piecewise, time csymbol",
    "BIOMD0000000700": "function definitions, piecewise",
    "BIOMD0000001000": "initial assignments, function definitions",
}


def reference(sbml: str, model, t_end: float, n: int, rtol=TOL["rtol"], atol=TOL["atol"]):
    """libroadrunner on the original SBML, selecting each name in its own units."""
    rr = roadrunner.RoadRunner(sbml)
    rr.integrator.relative_tolerance, rr.integrator.absolute_tolerance = rtol, atol
    names = [*model.initial, *model.assignments]
    rr.timeCourseSelections = ["time"] + [
        "time" if model.kinds[x] == "time"
        else f"[{model.sbml_ids[x]}]" if model.kinds[x] == "concentration"
        else model.sbml_ids[x] for x in names]
    out = rr.simulate(0, t_end, n)
    return out[:, 0], {x: out[:, j + 1] for j, x in enumerate(names)}


def worst_error(ours: dict, ref: dict) -> tuple[str, float]:
    errors = {}
    for k, r in ref.items():
        a = np.asarray(ours[k], dtype=float)
        errors[k] = (math.inf if not np.all(np.isfinite(a)) else
                     float(np.max(np.abs(a - r)) / max(float(np.max(np.abs(r))), 1e-10)))
    k = max(errors, key=errors.get)
    return k, errors[k]


def run_against_reference(path: Path, t_end=50.0, dt=0.5, solver=TOL):
    model = read_sbml(path)
    t, ours = run_document(document(model, dt, **solver), t_end)
    t_ref, ref = reference(str(path), model, t_end, len(t), solver["rtol"], solver["atol"])
    np.testing.assert_allclose(t, t_ref, rtol=0, atol=1e-12)
    return model, ours, ref


def test_pinned_files_match_checksums():
    for line in (DATA / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("model_id", sorted(IMPORTABLE))
def test_import_matches_roadrunner(model_id):
    _, ours, ref = run_against_reference(DATA / f"{model_id}.xml")
    name, err = worst_error(ours, ref)
    assert err <= AGREE, f"{model_id} ({IMPORTABLE[model_id]}): {name} off by {err:.2e}"


def test_error_converges_with_tolerance():
    """Shrinking tolerances shrink the gap: integrator error, not semantics."""
    errors = []
    for rtol in (1e-8, 1e-10, 1e-12):
        solver = {"rtol": rtol, "atol": rtol * 1e-2}
        _, ours, ref = run_against_reference(DATA / "BIOMD0000000005.xml", solver=solver)
        errors.append(worst_error(ours, ref)[1])
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[2] <= AGREE


def test_negative_control():
    """A one-term change (sign of one rate) must fail the agreement check."""
    path = DATA / "BIOMD0000000012.xml"
    model = read_sbml(path)
    x = next(iter(model.rhs))
    broken = {**model.rhs, x: f"-({model.rhs[x]})"}
    t, ours = run_document(ode_document(broken, model.params, model.initial, interval=0.5,
                                        assignments=model.assignments, **TOL), 50.0)
    _, ref = reference(str(path), model, 50.0, len(t))
    assert worst_error(ours, ref)[1] > 1e3 * AGREE


def test_assignments_are_emitted_and_match():
    model, ours, ref = run_against_reference(DATA / "BIOMD0000000012.xml")
    assert model.assignments, "the repressilator declares assignment rules"
    for name in model.assignments:
        assert name in ours
        r = ref[name]
        assert np.max(np.abs(ours[name] - r)) <= AGREE * max(np.max(np.abs(r)), 1e-10)


def test_time_dependent_model_is_autonomized():
    model = read_sbml(DATA / "BIOMD0000000400.xml")
    assert model.rhs["time"] == "1" and model.initial["time"] == 0.0
    assert model.kinds["time"] == "time" and model.notes


def test_event_model_is_rejected():
    with pytest.raises(UnsupportedSBML, match="event"):
        read_sbml(DATA / "BIOMD0000000001.xml")


# Hand-built models (real libsbml documents) for constructs the pinned set lacks.

def _l3_model(build, level=3, version=2) -> str:
    doc = libsbml.SBMLDocument(level, version)
    m = doc.createModel()
    m.setId("handbuilt")
    build(m)
    return libsbml.writeSBMLToString(doc)


def _compartment(m, cid, size):
    c = m.createCompartment()
    c.setId(cid), c.setSize(size), c.setConstant(True), c.setSpatialDimensions(3)


def _species(m, sid, comp, conc=None, amount=None, only_substance=False, cf=None):
    s = m.createSpecies()
    s.setId(sid), s.setCompartment(comp), s.setHasOnlySubstanceUnits(only_substance)
    s.setBoundaryCondition(False), s.setConstant(False)
    if conc is not None:
        s.setInitialConcentration(conc)
    if amount is not None:
        s.setInitialAmount(amount)
    if cf:
        s.setConversionFactor(cf)


def _parameter(m, pid, value):
    p = m.createParameter()
    p.setId(pid), p.setValue(value), p.setConstant(True)


def _reaction(m, rid, reactants, products, formula):
    r = m.createReaction()
    r.setId(rid), r.setReversible(False)
    for sid, stoich in reactants:
        ref = r.createReactant()
        ref.setSpecies(sid), ref.setStoichiometry(stoich), ref.setConstant(True)
    for sid, stoich in products:
        ref = r.createProduct()
        ref.setSpecies(sid), ref.setStoichiometry(stoich), ref.setConstant(True)
    r.createKineticLaw().setMath(_math(m, formula))
    return r


def _math(m, formula):
    """Parse an L3 formula with the model's ids in scope (so ``pi``, ``time``
    resolve to the model's own elements where they exist)."""
    settings = libsbml.L3ParserSettings()
    settings.setModel(m)
    return libsbml.parseL3FormulaWithSettings(formula, settings)


def _rule(m, kind, variable, formula):
    rule = m.createRateRule() if kind == "rate" else m.createAssignmentRule()
    rule.setVariable(variable)
    rule.setMath(_math(m, formula))


def _variable(m, pid, value):
    p = m.createParameter()
    p.setId(pid), p.setConstant(False)
    if value is not None:
        p.setValue(value)


def assert_matches_roadrunner(sbml, t_end=10.0, dt=0.25):
    model = read_sbml(sbml)
    t, ours = run_document(document(model, dt, **TOL), t_end)
    _, ref = reference(sbml, model, t_end, len(t))
    name, err = worst_error(ours, ref)
    assert err <= AGREE, f"{name} off by {err:.2e}"
    return model


def _transport(m):
    """Species moving between compartments of different sizes, both units."""
    _compartment(m, "outside", 2.0)
    _compartment(m, "inside", 0.5)
    _species(m, "A", "outside", conc=3.0)
    _species(m, "in", "inside", amount=0.0, only_substance=True, cf="half")  # keyword id
    _parameter(m, "k", 0.7)
    _parameter(m, "half", 0.5)
    _reaction(m, "uptake", [("A", 1)], [("in", 2)], "k * A * outside * (1 + 0.1 * sin(time))")


def test_handbuilt_compartments_units_conversion_factor_keyword_id():
    sbml = _l3_model(_transport)
    model = read_sbml(sbml)
    assert model.sbml_ids["in_"] == "in" and model.kinds["in_"] == "amount"
    assert model.kinds["A"] == "concentration"
    t, ours = run_document(document(model, 0.25, **TOL), 10.0)
    _, ref = reference(sbml, model, 10.0, len(t))
    name, err = worst_error(ours, ref)
    assert err <= AGREE, f"{name} off by {err:.2e}"


def test_piecewise_without_otherwise_is_rejected():
    def build(m):
        _compartment(m, "c", 1.0)
        _species(m, "X", "c", conc=1.0)
        _reaction(m, "r", [("X", 1)], [], "piecewise(X, X > 0.5)")
    with pytest.raises(UnsupportedSBML, match="otherwise"):
        read_sbml(_l3_model(build))


def test_exact_float_constants_survive_import():
    """libsbml reads full precision; the importer must keep every bit of it."""
    def build(m):
        _compartment(m, "c", 1.0)
        _species(m, "X", "c", conc=1.0)
        _reaction(m, "r", [("X", 1)], [], "0.25 * X")
    sbml = _l3_model(build).replace("<cn> 0.25 </cn>", "<cn> 0.30000000000000004 </cn>")
    model = read_sbml(sbml)
    assert "0.30000000000000004" in model.rhs["X"]


# Export: expressions -> SBML

def _workspace_composite(name):
    spec = yaml.safe_load((COMPOSITES / f"{name}.composite.yaml").read_text(encoding="utf-8"))
    return ode_from_state(substitute_parameters(spec["state"], spec.get("parameters", {})))


@pytest.mark.parametrize("name,t_end", [
    ("logistic_growth", 10.0), ("lotka_volterra", 10.0),
    ("michaelis_menten", 10.0), ("quadratic_blowup", 0.5)])
def test_export_runs_identically_in_roadrunner(name, t_end):
    args = _workspace_composite(name)
    sbml = write_sbml(**args, model_id=name)
    t, ours = run_document(ode_document(**args, interval=0.05, **TOL), t_end)
    rr = roadrunner.RoadRunner(sbml)
    rr.integrator.relative_tolerance, rr.integrator.absolute_tolerance = TOL["rtol"], TOL["atol"]
    names = list(args["initial"])
    rr.timeCourseSelections = ["time", *names]
    out = rr.simulate(0, t_end, len(t))
    var, err = worst_error(ours, {x: out[:, j + 1] for j, x in enumerate(names)})
    assert err <= AGREE, f"{name}: {var} off by {err:.2e}"


@pytest.mark.parametrize("name,t_end", [("logistic_growth", 10.0), ("lotka_volterra", 10.0), ("michaelis_menten", 10.0), ("quadratic_blowup", 0.5)])
def test_round_trip_reproduces_trajectories(name, t_end):
    args = _workspace_composite(name)
    model = read_sbml(write_sbml(**args, model_id=name))
    assert model.params == args["params"] and model.initial == args["initial"]
    _, before = run_document(ode_document(**args, interval=0.1, **TOL), t_end)
    _, after = run_document(document(model, 0.1, **TOL), t_end)
    for x in args["initial"]:
        np.testing.assert_array_equal(after[x], before[x])


def test_export_of_imported_model_with_assignments_is_valid_and_faithful():
    """BioModels -> expressions -> SBML still matches roadrunner on the original."""
    path = DATA / "BIOMD0000000012.xml"
    model = read_sbml(path)
    again = read_sbml(write_sbml(model.rhs, model.params, model.initial,
                                 assignments=model.assignments, model_id="repressilator"))
    t, ours = run_document(document(again, 0.5, **TOL), 50.0)
    _, ref = reference(str(path), model, 50.0, len(t))
    name, err = worst_error(ours, ref)
    assert err <= AGREE, f"{name} off by {err:.2e}"


def test_export_keeps_libsbml_precision():
    """libsbml writes doubles to 15 significant digits (it has no setting for
    more), so an exported value is exact to that, a relative change <= 5e-15."""
    x = 0.1 + 0.2
    model = read_sbml(write_sbml({"y": "-k*y"}, {"k": x}, {"y": 1.0}))
    assert model.params["k"] == float(f"{x:.15g}")
    assert abs(model.params["k"] - x) <= 5e-15 * x


def test_export_rejects_invalid_identifier():
    with pytest.raises(UnsupportedSBML, match="identifier"):
        write_sbml({"x": "-x"}, {}, {"x": 1.0}, model_id="not valid")


# Regression: MathExpressionStep substituted parameters before compiling, so
# sympy folded exp(-80*(t - ts)) at ts = 30 into 1e1042*exp(-80*t) = inf*0 = nan.

def test_math_expression_parameters_do_not_overflow():
    step = MathExpressionStep(
        {"expressions": [{"out": "y", "expr": "1/(1 + exp(-80*(t - ts)))"}],
         "params": {"ts": 30.0}}, core=allocate_core())
    y = step.update({"t": 30.05})["y"]
    assert math.isfinite(y)
    assert y == pytest.approx(1 / (1 + math.exp(-4.0)), rel=1e-12)


# Workspace composite spec + CLI (through process-bigraph's own spec loader and
# the workspace core, as the workbench runs composites)

def test_import_cli_writes_a_runnable_workspace_composite(tmp_path, monkeypatch):
    from process_bigraph import Composite
    from process_bigraph.composite_spec import CompositeSpec
    from process_bigraph.emitter import gather_emitter_results

    from viva_expressions.core import build_core
    from viva_expressions.sbml.__main__ import main

    path = DATA / "BIOMD0000000012.xml"
    assert main(["import", str(path), "--name", "repressilator", "--out", str(tmp_path),
                 "--interval", "0.5", "--source", "BIOMD0000000012"]) == 0
    monkeypatch.chdir(tmp_path)    # JSONEmitter writes into the working directory
    spec = CompositeSpec.from_file(tmp_path / "repressilator.composite.yaml")
    assert "BIOMD0000000012" in spec.description
    composite = Composite(spec.to_document(overrides={"rtol": 1e-12, "atol": 1e-14}),
                          core=build_core())
    composite.run(50.0)
    rows = gather_emitter_results(composite)[("emitter",)]
    model = read_sbml(path)
    _, ref = reference(str(path), model, 50.0, len(rows))
    ours = {k: np.array([r[k] for r in rows]) for k in ref}
    name, err = worst_error(ours, ref)
    assert err <= AGREE, f"{name} off by {err:.2e}"


def test_composite_parameters_are_variable(tmp_path):
    """A study variant overriding a parameter must change the dynamics."""
    from process_bigraph.composite_spec import CompositeSpec

    from viva_expressions.sbml.composite import composite_spec

    model = read_sbml(DATA / "BIOMD0000000012.xml")
    spec_path = tmp_path / "r.composite.yaml"
    spec_path.write_text(yaml.safe_dump(composite_spec(model, "r", 0.5)))
    spec = CompositeSpec.from_file(spec_path)
    p = next(iter(model.params))
    doc = spec.to_document(overrides={p: 2 * model.params[p]})
    assert doc["state"]["ode"]["config"]["params"][p] == 2 * model.params[p]
    x = next(iter(model.initial))
    assert doc["state"]["stores"][x] == model.initial[x]


def test_cli_exit_codes(tmp_path):
    from viva_expressions.sbml.__main__ import main
    assert main(["import", str(DATA / "BIOMD0000000001.xml"), "--name", "x",
                 "--out", str(tmp_path)]) == 2
    assert main(["import", str(tmp_path / "missing.xml"), "--name", "x",
                 "--out", str(tmp_path)]) == 1


def test_export_cli_from_workspace_composite(tmp_path):
    from viva_expressions.sbml.__main__ import main
    out = tmp_path / "mm.xml"
    assert main(["export", str(COMPOSITES / "michaelis_menten.composite.yaml"),
                 "--out", str(out)]) == 0
    model = read_sbml(out)
    assert model.model_id == "michaelis_menten"
    assert model.params == {"Vmax": 1.0, "Km": 2.0}
    assert model.initial == {"S": 10.0, "P": 0.0}


# Edge cases from the adversarial review: each one a real libsbml model,
# checked against roadrunner (or, for rejections, required to raise).

def _volume_and_boundary(m):
    """Rate rule on a concentration species with V != 1; boundary species with a
    rate rule; species- vs model-level conversion factors; a reaction id in math;
    log and root with their default base and degree."""
    _compartment(m, "c", 2.0)
    _species(m, "A", "c", conc=1.0)
    _species(m, "B", "c", conc=0.5)
    _species(m, "C", "c", conc=0.0, cf="cf_species")
    _species(m, "D", "c", conc=0.0)
    _species(m, "E", "c", conc=1.0)
    m.getSpecies("B").setBoundaryCondition(True)
    _parameter(m, "k", 0.3)
    _parameter(m, "cf_species", 3.0)
    _parameter(m, "cf_model", 0.5)
    m.setConversionFactor("cf_model")
    _reaction(m, "v", [("A", 1)], [("C", 1), ("D", 2)], "k * A * c")
    _rule(m, "rate", "E", "-0.1 * log(10, E + 1) - 0.05 * root(2, E) - 0.01 * A")
    _rule(m, "rate", "B", "0.2 * v")
    _variable(m, "flux", None)
    _rule(m, "assignment", "flux", "v / c")


def _function_names_and_constants(m):
    """A parameter named ``pi`` beside the constant pi, a parameter named ``sin``
    beside a sin() call, keyword ids, xor, abs and floor."""
    _compartment(m, "c", 1.0)
    _species(m, "X", "c", conc=1.0)
    _parameter(m, "sin", 0.4)
    _parameter(m, "lambda", 0.2)
    pi_param = m.createParameter()
    pi_param.setId("pi"), pi_param.setValue(5.0), pi_param.setConstant(True)
    r = _reaction(m, "r", [("X", 1)], [], "sin * X")
    math = libsbml.ASTNode(libsbml.AST_TIMES)    # pi (the parameter) * <pi/> (the constant)
    name = libsbml.ASTNode(libsbml.AST_NAME)
    name.setName("pi")
    math.addChild(name), math.addChild(libsbml.ASTNode(libsbml.AST_CONSTANT_PI))
    _variable(m, "z", 0.0)
    m.createRateRule().setVariable("z")
    m.getRateRule("z").setMath(math)
    _variable(m, "w", 1.0)
    _rule(m, "rate", "w", "piecewise(lambda * abs(sin(time) - 0.5) + 0.01 * floor(w), "
                          "xor(X > 0.5, time > 2), -0.1 * w)")
    assert r is not None


@pytest.mark.parametrize("build", [_volume_and_boundary, _function_names_and_constants])
def test_handbuilt_semantics_match_roadrunner(build):
    assert_matches_roadrunner(_l3_model(build))


def test_reserved_and_keyword_ids_are_renamed_and_recorded():
    model = read_sbml(_l3_model(_function_names_and_constants))
    assert model.sbml_ids["pi_"] == "pi" and model.sbml_ids["sin_"] == "sin"
    assert model.sbml_ids["lambda_"] == "lambda"
    assert model.rhs["z"] in ("pi*pi_", "pi_*pi")   # the parameter times the constant
    assert model.kinds["pi_"] == "parameter"


def _bad(construct):
    def build(m):
        _compartment(m, "c", 1.0)
        _species(m, "X", "c", conc=1.0)
        _parameter(m, "k", 1.0)
        r = _reaction(m, "r", [("X", 1)], [], "k * X")
        if construct == "delay":
            r.getKineticLaw().setMath(_math(m, "k * delay(X, 1)"))
        elif construct == "algebraic":
            _variable(m, "y", 1.0)
            rule = m.createAlgebraicRule()
            rule.setMath(_math(m, "y - X"))
        elif construct == "compartment_rule":
            m.getCompartment("c").setConstant(False)
            _rule(m, "rate", "c", "0.1")
        elif construct == "variable_stoichiometry":
            r.getReactant(0).setId("sr"), r.getReactant(0).setConstant(False)
            _rule(m, "assignment", "sr", "1 + X")
        elif construct == "infinity":
            r.getKineticLaw().setMath(_math(m, "k * X * INF"))
        elif construct == "event":
            e = m.createEvent()
            e.setUseValuesFromTriggerTime(True)
            t = e.createTrigger()
            t.setMath(_math(m, "time > 1")), t.setPersistent(True), t.setInitialValue(False)
            a = e.createEventAssignment()
            a.setVariable("k"), a.setMath(_math(m, "2"))
            m.getParameter("k").setConstant(False)
    return build


@pytest.mark.parametrize("construct,match", [
    ("delay", "delay"), ("algebraic", "algebraic"), ("compartment_rule", "compartment"),
    ("variable_stoichiometry", "stoichiometry"), ("infinity", "non-finite"),
    ("event", "event")])
def test_unsupported_constructs_raise(construct, match):
    with pytest.raises(UnsupportedSBML, match=match):
        read_sbml(_l3_model(_bad(construct)))


def test_fast_reaction_raises():
    def build(m):
        _compartment(m, "c", 1.0)
        s = m.createSpecies()
        s.setId("X"), s.setCompartment("c"), s.setInitialConcentration(1.0)
        _parameter(m, "k", 1.0)
        r = m.createReaction()
        r.setId("r"), r.setFast(True), r.setReversible(False)
        r.createReactant().setSpecies("X")
        r.createKineticLaw().setMath(libsbml.parseL3Formula("k * X"))
    with pytest.raises(UnsupportedSBML, match="fast"):
        read_sbml(_l3_model(build, level=2, version=4))


def test_composite_keeps_model_names_that_match_solver_or_clock(tmp_path, monkeypatch):
    """Model parameters named atol/method and a species named time survive."""
    from process_bigraph import Composite
    from process_bigraph.composite_spec import CompositeSpec
    from process_bigraph.emitter import gather_emitter_results

    from viva_expressions.core import build_core
    from viva_expressions.sbml.composite import composite_spec

    def build(m):
        _compartment(m, "c", 1.0)
        _species(m, "time", "c", conc=2.0)
        _parameter(m, "atol", 0.3)
        _reaction(m, "r", [("time", 1)], [], "atol * time")
    sbml = _l3_model(build)
    model = read_sbml(sbml)
    spec = composite_spec(model, "clash", 0.25)
    assert spec["parameters"]["atol"]["default"] == 0.3
    monkeypatch.chdir(tmp_path)    # JSONEmitter writes into the working directory
    path = tmp_path / "clash.composite.yaml"
    path.write_text(yaml.safe_dump(spec))
    composite = Composite(CompositeSpec.from_file(path).to_document(), core=build_core())
    composite.run(5.0)
    rows = gather_emitter_results(composite)[("emitter",)]
    ours = np.array([r["time"] for r in rows])
    assert np.allclose(ours, 2.0 * np.exp(-0.3 * np.array([r["time_"] for r in rows])),
                       rtol=1e-6)


def test_export_cli_keeps_assignments(tmp_path):
    """BioModels -> composite.yaml -> SBML keeps the assignment rules and still
    matches roadrunner on the original."""
    from viva_expressions.sbml.__main__ import main
    path = DATA / "BIOMD0000000012.xml"
    assert main(["import", str(path), "--name", "rep", "--out", str(tmp_path)]) == 0
    out = tmp_path / "rep.xml"
    assert main(["export", str(tmp_path / "rep.composite.yaml"), "--out", str(out)]) == 0
    doc = libsbml.readSBMLFromFile(str(out))
    original = read_sbml(path)
    assert doc.getModel().getNumRules() == len(original.rhs) + len(original.assignments)
    model = read_sbml(out)
    t, ours = run_document(document(model, 0.5, **TOL), 50.0)
    _, ref = reference(str(path), original, 50.0, len(t))
    name, err = worst_error(ours, ref)
    assert err <= AGREE, f"{name} off by {err:.2e}"


def test_export_rejects_processes_it_cannot_carry():
    from viva_expressions.sbml.write import ode_from_state
    state = ode_document({"x": "-x"}, {}, {"x": 1.0})["state"]
    state["other"] = {"_type": "process", "address": "local:SomethingElse",
                      "config": {}, "inputs": {}, "outputs": {}}
    with pytest.raises(UnsupportedSBML, match="SomethingElse"):
        ode_from_state(state)


CLOSED_FORMS = {
    # name: (t_end, {variable: x(t, parameter defaults)})
    "logistic_growth": (10.0, lambda t, p: {
        "N": p["K"] / (1 + (p["K"] / p["N0"] - 1) * np.exp(-p["r"] * t))}),
    "quadratic_blowup": (0.5, lambda t, p: {"x": p["x0"] / (1 - p["x0"] * t)}),
    "michaelis_menten": (10.0, lambda t, p: {"S": p["Km"] * np.real(lambertw(
        p["S0"] / p["Km"] * np.exp((p["S0"] - p["Vmax"] * t) / p["Km"])))}),
}


@pytest.mark.parametrize("name", sorted(CLOSED_FORMS))
def test_exported_sbml_matches_closed_form(name):
    """Roadrunner on the exported SBML against the exact solution."""
    spec = yaml.safe_load((COMPOSITES / f"{name}.composite.yaml").read_text(encoding="utf-8"))
    defaults = {k: v["default"] for k, v in spec["parameters"].items()}
    t_end, exact = CLOSED_FORMS[name]
    rr = roadrunner.RoadRunner(write_sbml(**_workspace_composite(name), model_id=name))
    rr.integrator.relative_tolerance, rr.integrator.absolute_tolerance = TOL["rtol"], TOL["atol"]
    truth = exact(np.linspace(0, t_end, 101), defaults)
    rr.timeCourseSelections = ["time", *truth]
    out = rr.simulate(0, t_end, 101)
    var, err = worst_error({x: out[:, j + 1] for j, x in enumerate(truth)}, truth)
    assert err <= AGREE, f"{name}: {var} off by {err:.2e}"


def test_exported_lotka_volterra_conserves_its_first_integral():
    spec = yaml.safe_load((COMPOSITES / "lotka_volterra.composite.yaml").read_text(encoding="utf-8"))
    p = {k: v["default"] for k, v in spec["parameters"].items()}
    rr = roadrunner.RoadRunner(write_sbml(**_workspace_composite("lotka_volterra")))
    rr.integrator.relative_tolerance, rr.integrator.absolute_tolerance = TOL["rtol"], TOL["atol"]
    rr.timeCourseSelections = ["x", "y"]
    out = rr.simulate(0, 10.0, 101)
    x, y = out[:, 0], out[:, 1]
    v = p["d"] * x - p["g"] * np.log(x) + p["b"] * y - p["a"] * np.log(y)
    assert np.max(np.abs(v - v[0])) <= AGREE * abs(v[0])


# Regression: MathExpressionStep returned numpy 0-d arrays for Piecewise outputs
# (numpy.select), which JSONEmitter cannot serialize. The imported BIOMD400
# composite then died at t = 0 under the workspace's emitter.

def test_math_expression_float_outputs_are_python_floats():
    step = MathExpressionStep(
        {"expressions": [{"out": "y", "expr": "Piecewise((x, x > 0), (0, True))"}],
         "params": {}}, core=allocate_core())
    y = step.update({"x": 2.0})["y"]
    assert type(y) is float and y == 2.0


def test_piecewise_composite_runs_under_the_json_emitter(tmp_path, monkeypatch):
    from process_bigraph import Composite
    from process_bigraph.composite_spec import CompositeSpec
    from process_bigraph.emitter import gather_emitter_results

    from viva_expressions.core import build_core
    from viva_expressions.sbml.composite import composite_spec

    path = DATA / "BIOMD0000000400.xml"
    model = read_sbml(path)
    monkeypatch.chdir(tmp_path)    # JSONEmitter writes into the working directory
    spec_path = tmp_path / "b400.composite.yaml"
    spec_path.write_text(yaml.safe_dump(composite_spec(model, "b400", 0.5)), encoding="utf-8")
    spec = CompositeSpec.from_file(spec_path)
    composite = Composite(spec.to_document(overrides={"rtol": 1e-12, "atol": 1e-14}),
                          core=build_core())
    composite.run(50.0)
    rows = gather_emitter_results(composite)[("emitter",)]
    assert len(rows) == 101
    _, ref = reference(str(path), model, 50.0, len(rows))
    ours = {k: np.array([r[k] for r in rows]) for k in ref}
    name, err = worst_error(ours, ref)
    assert err <= AGREE, f"{name} off by {err:.2e}"
