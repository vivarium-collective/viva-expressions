import json
from dataclasses import replace
from pathlib import Path

import pytest

from viva_expressions.composites import run_document
from viva_expressions.synthesize import export
from viva_expressions.synthesize.__main__ import main
from viva_expressions.synthesize.fit import fit_model
from viva_expressions.synthesize.motifs import resolve
from viva_expressions.synthesize.spec import load_spec

EXAMPLES = Path(__file__).resolve().parents[1] / "systems" / "examples"


@pytest.fixture(scope="module")
def logistic_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("logistic")
    code = main([str(EXAMPLES / "logistic.yml"), "--out", str(out)])
    return code, out


def test_feasible_spec_exits_zero_and_writes_artifacts(logistic_run):
    code, out = logistic_run
    assert code == 0
    assert {p.name for p in out.iterdir()} == {"report.json", "report.md", "composite.json"}
    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rep["status"] == "verified" and rep["winner"]["origin"] == "motif:logistic"
    assert "## Composite verification" in (out / "report.md").read_text(encoding="utf-8")


def test_fresh_composite_from_exported_json_reproduces_steady_state(logistic_run):
    _, out = logistic_run
    doc = json.loads((out / "composite.json").read_text(encoding="utf-8"))
    t, series = run_document(doc, 30.0)
    assert t[-1] == pytest.approx(30.0)
    assert series["N"][-1] == pytest.approx(5.0, abs=0.1)


def test_check_reports_bad_field_and_exits_one(tmp_path, capsys):
    bad = tmp_path / "bad.yml"
    bad.write_text("name: bad\ntime: {t_end: 10, n_points: 11}\n"
                   "variables: {y: [{name: x, value: 1}]}\n"
                   "behavior: {features: [{kind: steady_state, var: x, value: 0, tol: -1}]}\n"
                   "candidates: [{motif: exponential_decay, bind: {x: x}}]\n")
    assert main([str(bad), "--check"]) == 1
    assert "behavior.features[0].tol: must be > 0" in capsys.readouterr().err


def test_unknown_motif_is_a_spec_error(tmp_path, capsys):
    bad = tmp_path / "bad.yml"
    bad.write_text("name: bad\ntime: {t_end: 10, n_points: 11}\n"
                   "variables: {y: [{name: x, value: 1}]}\n"
                   "behavior: {features: [{kind: steady_state, var: x, value: 0, tol: 0.1}]}\n"
                   "candidates: [{motif: nope, bind: {x: x}}]\n")
    assert main([str(bad), "--check"]) == 1
    assert "candidates[0]: unknown motif 'nope'" in capsys.readouterr().err


def test_infeasible_spec_exits_two_without_composite(tmp_path):
    spec = tmp_path / "grow.yml"
    spec.write_text("name: grow\ntime: {t_end: 10, n_points: 101}\n"
                    "variables: {y: [{name: x, value: 1}]}\n"
                    "behavior: {features: [{kind: steady_state, var: x, value: 5, tol: 0.1}]}\n"
                    "candidates: [{motif: exponential_decay, bind: {x: x}}]\n")
    out = tmp_path / "out"
    assert main([str(spec), "--out", str(out)]) == 2
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["status"] == "infeasible"
    assert not (out / "composite.json").exists()


def test_verify_catches_fit_export_drift():
    spec = load_spec(EXAMPLES / "logistic.yml")
    result = fit_model(resolve(spec.candidates[0], spec), spec)
    doc = export.to_document(result, spec)
    assert export.verify(result, spec, doc).verified
    # the document no longer matches the parameters it claims to export
    drifted = replace(result, theta={**result.theta, "r": result.theta["r"] * 1.05})
    v = export.verify(drifted, spec, doc)
    assert not v.verified and v.feasible and v.max_deviation > export.AGREEMENT_RTOL


def test_list_motifs(capsys):
    assert main(["--list-motifs"]) == 0
    assert "goodwin" in capsys.readouterr().out


def test_coarse_interval_is_still_verified(tmp_path):
    # Regression: interpolating a coarse Composite onto the fit grid exceeded
    # the agreement tolerance by itself and reported a correct model unverified.
    out = tmp_path / "out"
    assert main([str(EXAMPLES / "logistic.yml"), "--out", str(out), "--interval", "0.7"]) == 0
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["verification"]["max_deviation"] < 1e-4


def test_input_named_like_a_motif_parameter_fails_check(tmp_path, capsys):
    # Regression: --check passed, then fitting crashed with a bare ValueError.
    spec = tmp_path / "clash.yml"
    spec.write_text("name: clash\ntime: {t_end: 30, n_points: 301}\n"
                    "variables: {x: [{name: r, value: 1}], y: [{name: N, value: 0.1}]}\n"
                    "behavior: {features: [{kind: steady_state, var: N, value: 5, tol: 0.1}]}\n"
                    "candidates: [{motif: logistic, bind: {x: N}}]\n")
    assert main([str(spec), "--check"]) == 1
    assert "candidates[0]: names declared more than once: ['r']" in capsys.readouterr().err
    spec.write_text(spec.read_text(encoding="utf-8").replace("bind: {x: N}}", "bind: {x: N}, prefix: g_}"),
                    encoding="utf-8")
    assert main([str(spec), "--check"]) == 0


@pytest.fixture
def ascii_locale():
    """The C locale: text I/O that relies on the locale default becomes ASCII.

    Native libraries (libsbml, roadrunner, antimony, COPASI) call setlocale, and
    on CI's Linux runner the default encoding became ASCII mid-session, so
    report.md's em dash failed to write. Writes must name their encoding.
    """
    import locale
    previous = locale.setlocale(locale.LC_CTYPE)
    locale.setlocale(locale.LC_CTYPE, "C")
    try:
        yield
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)


def test_reports_write_as_utf8_under_an_ascii_locale(tmp_path, ascii_locale):
    spec = tmp_path / "grow.yml"
    spec.write_text("name: grow — decay\ntime: {t_end: 10, n_points: 101}\n"
                    "variables: {y: [{name: x, value: 1}]}\n"
                    "behavior: {features: [{kind: steady_state, var: x, value: 5, tol: 0.1}]}\n"
                    "candidates: [{motif: exponential_decay, bind: {x: x}}]\n",
                    encoding="utf-8")
    out = tmp_path / "out"
    assert main([str(spec), "--out", str(out)]) == 2
    assert "—" in (out / "report.md").read_text(encoding="utf-8")
