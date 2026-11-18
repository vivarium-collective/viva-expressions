"""POST-HOC resolution analysis of bench-tellurium v2 (not pre-registered).

The adversarial panel on v2 asked what the v2 tests can and cannot resolve.
This script reproduces those figures from committed inputs alone, so the study
can cite them. It uses the same method as the tests: OdeProcess at the
study's 0.5 interval to t = 100, libroadrunner stepped in 0.5 chunks as
TelluriumProcess does, and the v2 metric and compared set
(tests/test_agreement_v2.py). For each model it computes:

* ``floor``: each solver's self-convergence floor, the v2 metric between its
  rtol 1e-12/atol 1e-14 and rtol 1e-13/atol 1e-15 runs. ``ratio_tau`` is
  tau_m / the larger floor.
* ``sensitivity``: for every constant parameter (local ones promoted), the
  roadrunner trajectory sensitivity to a +1e-3 relative change (as in
  select_controls.py, but over every parameter). ``undetected_0p1pct`` counts
  parameters whose 0.1% change moves the trajectories by <= tau_m.
* ``resolution``: tau_m / sensitivity of the control parameter, i.e. the
  smallest relative error in it that H1 can resolve.
* ``planted``: the control parameter scaled by (1 + delta) on the OdeProcess
  side for delta in 1e-10..1e-6. Gives the smallest delta that fails H1 (tight
  vs the recorded tellurium_tight run) or H2 (log10(default/tight) < 2.5,
  against the recorded tellurium runs).
* ``restart``: roadrunner stepped in 0.5 chunks vs one continuous integration,
  both at rtol 1e-12 (the size of the shared restart artefact).
* ``excluded_outputs``: the assignment outputs TelluriumProcess does not emit
  (boundary species), OdeProcess at tight tolerance vs a continuous
  roadrunner run selecting them directly.
* ``transient_in_first_interval``: per species, the share of its total
  movement over t = 0..100 that happens in [0, 0.5], the interval the
  comparison skips because Tellurium's t = 0 row is empty.

    uv run python <this file>  ->  analysis/posthoc_resolution.json
"""
import copy
import importlib.util
import json
import math
from pathlib import Path

import libsbml
import numpy as np
import roadrunner

from viva_expressions.composites import run_document
from viva_expressions.sbml import document, read_sbml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v2", HERE.parent / "tests" / "test_agreement_v2.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

TIGHT, TIGHTER = (1e-12, 1e-14), (1e-13, 1e-15)
DEFAULT = (1e-8, 1e-10)
DELTAS = [1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6]


def sbml(model_id: int) -> str:
    """The pinned SBML with local parameters promoted (as the importer does)."""
    doc = libsbml.readSBMLFromFile(str(v2.MODELS_DIR / f"BIOMD{model_id:010d}.xml"))
    props = libsbml.ConversionProperties()
    props.addOption("promoteLocalParameters", True)
    assert doc.convert(props) == libsbml.LIBSBML_OPERATION_SUCCESS
    return libsbml.writeSBMLToString(doc)


def run_ode(model, tol, overrides=None) -> dict:
    model = copy.deepcopy(model)
    model.params.update(overrides or {})
    _, series = run_document(document(model, 0.5, rtol=tol[0], atol=tol[1]), 100.0)
    return series


def run_rr(text: str, tol, overrides=None) -> dict:
    """roadrunner stepped in 0.5 chunks, as TelluriumProcess runs it."""
    rr = roadrunner.RoadRunner(text)
    for k, v in (overrides or {}).items():
        rr[k] = v
    rr.integrator.relative_tolerance, rr.integrator.absolute_tolerance = tol
    species, params = rr.model.getFloatingSpeciesIds(), rr.model.getGlobalParameterIds()
    rows = []
    for _ in range(200):
        t0 = rr.getCurrentTime()
        rr.simulate(t0, t0 + 0.5, 2)
        rows.append({**{f"species.{s}": float(rr[s]) for s in species},
                     **{f"parameters.{p}": float(rr[p]) for p in params}})
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def errors(model_id: int, ours: dict, ref: dict) -> dict[str, float]:
    """The v2 metric over the v2 compared set; ``ref`` holds Tellurium-style
    columns without the t = 0 row, ``ours`` OdeProcess series with it."""
    imported = json.loads((v2.COMPOSITES / f"biomd{model_id:010d}_ode.import.json").read_text(
        encoding="utf-8"))
    volume = v2._volumes(model_id)
    names = [*imported["initial"], *(a for a in imported["assignments"]
                                     if f"parameters.{imported['sbml_ids'][a]}" in ref)]
    out = {}
    for name in names:
        kind, sid = imported["kinds"][name], imported["sbml_ids"][name]
        if name in imported["initial"] and kind in ("concentration", "amount"):
            r = ref[f"species.{sid}"] / (volume[sid] if kind == "concentration" else 1.0)
        else:
            r = ref[f"parameters.{sid}"]
        a = np.asarray(ours[name])[1:]
        out[name] = (math.inf if not (np.all(np.isfinite(a)) and np.all(np.isfinite(r))) else
                     float(np.max(np.abs(a - r)) / max(float(np.max(np.abs(r))), 1e-10)))
    return out


def worst(errs: dict) -> float:
    return max(errs.values())


def as_ode(model_id: int, rr_series: dict) -> dict:
    """roadrunner columns re-keyed to OdeProcess names (with a dummy t = 0 row),
    so a solver's self-floor uses the same metric."""
    imported = json.loads((v2.COMPOSITES / f"biomd{model_id:010d}_ode.import.json").read_text(
        encoding="utf-8"))
    volume = v2._volumes(model_id)
    out = {}
    for name in [*imported["initial"], *imported["assignments"]]:
        kind, sid = imported["kinds"][name], imported["sbml_ids"][name]
        key = (f"species.{sid}" if name in imported["initial"] and kind in ("concentration", "amount")
               else f"parameters.{sid}")
        if key in rr_series:
            col = rr_series[key] / (volume[sid] if key.startswith("species.") and kind == "concentration" else 1.0)
            out[name] = np.concatenate([[np.nan], col])
    return out


def tel_recorded(label: str) -> dict:
    run = v2.latest(label)
    return {c: np.asarray(run.observable(c)) for c in run.trajectory.columns
            if c.startswith(("species.", "parameters."))}


def analyse(model_id: int) -> dict:
    tau = v2.tau(model_id)
    model, text = read_sbml(v2.MODELS_DIR / f"BIOMD{model_id:010d}.xml"), sbml(model_id)
    control = v2.CONTROLS[f"BIOMD{model_id:010d}"]
    name, value = control["import_name"], control["value"]

    lsoda = worst(errors(model_id, run_ode(model, TIGHT), _as_tel(model_id, run_ode(model, TIGHTER))))
    rr_tight, rr_tighter = run_rr(text, TIGHT), run_rr(text, TIGHTER)
    cvode = worst(errors(model_id, as_ode(model_id, rr_tight), rr_tighter))
    floor = max(lsoda, cvode)

    base = run_rr(text, TIGHT)
    sens = {}
    m = libsbml.readSBMLFromString(text).getModel()
    ruled = {r.getVariable() for r in m.getListOfRules()} | {a.getSymbol() for a in m.getListOfInitialAssignments()}
    for p in m.getListOfParameters():
        if p.getConstant() and p.getId() not in ruled and p.isSetValue() and p.getValue() != 0.0:
            moved = run_rr(text, TIGHT, {p.getId(): p.getValue() * (1 + 1e-3)})
            sens[p.getId()] = max(
                float(np.max(np.abs(moved[k] - base[k])) / max(float(np.max(np.abs(base[k]))), 1e-10))
                for k in base if k.startswith("species.")) / 1e-3
    s_control = sens[control["sbml_id"]]

    tel_tight, tel_default = tel_recorded(f"biomd{model_id}_tellurium_tight"), tel_recorded(f"biomd{model_id}_tellurium")
    planted = []
    for d in DELTAS:
        over = {name: value * (1 + d)}
        e_t = worst(errors(model_id, run_ode(model, TIGHT, over), tel_tight))
        e_d = worst(errors(model_id, run_ode(model, DEFAULT, over), tel_default))
        planted.append({"delta": d, "tight": e_t, "orders": math.log10(e_d / e_t),
                        "h1_fails": e_t > tau, "h2_fails": math.log10(e_d / e_t) < 2.5})
    caught = [p for p in planted if p["h1_fails"] or p["h2_fails"]]
    return {
        "tau": tau,
        "floor": {"lsoda": lsoda, "cvode": cvode, "ratio_tau": tau / floor},
        "sensitivity": {"n": len(sens), "undetected_0p1pct": sum(1e-3 * s <= tau for s in sens.values()),
                        "control": control["sbml_id"], "control_s": s_control},
        "resolution": tau / s_control,
        "planted": planted,
        "smallest_caught": caught[0]["delta"] if caught else None,
        "largest_admitted_deviation": max((p["tight"] for p in planted
                                           if not (p["h1_fails"] or p["h2_fails"])), default=None),
    }


def _as_tel(model_id: int, ode_series: dict) -> dict:
    """OdeProcess series re-keyed to Tellurium-style columns (dropping t = 0)."""
    imported = json.loads((v2.COMPOSITES / f"biomd{model_id:010d}_ode.import.json").read_text(
        encoding="utf-8"))
    volume = v2._volumes(model_id)
    out = {}
    for name in [*imported["initial"], *imported["assignments"]]:
        kind, sid = imported["kinds"][name], imported["sbml_ids"][name]
        col = np.asarray(ode_series[name])[1:]
        if name in imported["initial"] and kind in ("concentration", "amount"):
            out[f"species.{sid}"] = col * (volume[sid] if kind == "concentration" else 1.0)
        else:
            out[f"parameters.{sid}"] = col
    return out


def run_rr_continuous(text: str, tol, ids) -> dict:
    rr = roadrunner.RoadRunner(text)
    rr.integrator.relative_tolerance, rr.integrator.absolute_tolerance = tol
    rr.timeCourseSelections = list(ids)
    out = np.asarray(rr.simulate(0, 100.0, 201))
    return {k: out[1:, j] for j, k in enumerate(ids)}


def extras(model_id: int) -> dict:
    model, text = read_sbml(v2.MODELS_DIR / f"BIOMD{model_id:010d}.xml"), sbml(model_id)
    rr = roadrunner.RoadRunner(text)
    species = list(rr.model.getFloatingSpeciesIds())
    chunked = run_rr(text, TIGHT)
    continuous = run_rr_continuous(text, TIGHT, species)
    restart = max(float(np.max(np.abs(chunked[f"species.{s}"] - continuous[s]))
                        / max(float(np.max(np.abs(continuous[s]))), 1e-10)) for s in species)
    imported = json.loads((v2.COMPOSITES / f"biomd{model_id:010d}_ode.import.json").read_text(
        encoding="utf-8"))
    emitted = {k.split(".", 1)[1] for k in chunked}
    excluded = [a for a in imported["assignments"] if imported["sbml_ids"][a] not in emitted]
    out = {"restart": restart, "excluded_outputs": {}}
    if excluded:
        ours = run_ode(model, TIGHT)
        sel = [(f"[{imported['sbml_ids'][a]}]" if imported["kinds"][a] == "concentration"
                else imported["sbml_ids"][a]) for a in excluded]
        ref = run_rr_continuous(text, TIGHT, sel)
        out["excluded_outputs"] = {
            a: float(np.max(np.abs(np.asarray(ours[a])[1:] - ref[s]))
                     / max(float(np.max(np.abs(ref[s]))), 1e-10)) for a, s in zip(excluded, sel)}
    fine = roadrunner.RoadRunner(text)
    fine.integrator.relative_tolerance, fine.integrator.absolute_tolerance = TIGHT
    fine.timeCourseSelections = ["time"] + species
    traj = np.asarray(fine.simulate(0, 100.0, 20001))
    early = traj[:, 0] <= 0.5
    shares = {}
    for j, s in enumerate(species, start=1):
        total = float(np.sum(np.abs(np.diff(traj[:, j]))))
        shares[s] = (float(np.sum(np.abs(np.diff(traj[early, j])))) / total) if total > 0 else 0.0
    out["transient_in_first_interval"] = {"min": min(shares.values()), "max": max(shares.values())}
    return out


def main() -> None:
    out = {f"BIOMD{m:010d}": {**analyse(m), **extras(m)} for m in v2.MODELS}
    (HERE / "posthoc_resolution.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for k, r in out.items():
        print(k, f"floor={max(r['floor']['lsoda'], r['floor']['cvode']):.2e} tau/floor={r['floor']['ratio_tau']:.1f}",
              f"resolution={r['resolution']:.1e} undetected={r['sensitivity']['undetected_0p1pct']}/{r['sensitivity']['n']}",
              f"smallest_caught={r['smallest_caught']} admitted={r['largest_admitted_deviation']}",
              f"restart={r['restart']:.1e} excluded={r['excluded_outputs']}",
              f"transient[0,0.5]={r['transient_in_first_interval']['min']:.3f}-{r['transient_in_first_interval']['max']:.3f}")


if __name__ == "__main__":
    main()
