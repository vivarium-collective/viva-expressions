"""bench-tellurium v2: the hypotheses pre-registered in study.yaml
``preregistration.v2`` (commit 542bf96, before any v2 run). v1 stays in
test_agreement.py.

Differences from v1, each with its rationale in the pre-registration:

* **Compared set.** Every OdeProcess state, plus every assignment-rule output
  that TelluriumProcess emits in its ``parameters`` map. Boundary-species
  outputs, which Tellurium doesn't emit, stay excluded.
* **H1.** Tight-tolerance agreement within tau_m: 1e-8, except BIOMD5 at 1e-7.
* **H2.** Convergence: log10(worst default / worst tight) >= 2.5.
* **H3.** The median-sensitivity parameter of each model (selected on the
  oracle side by analysis/select_controls.py), perturbed by +1% at tight
  tolerances, must exceed tau_m.
* **t = 0.** Still excluded: viva-tellurium#7 (empty first row) is unfixed.

Metric (unchanged): per variable, max_t |ode - tellurium| / max_t |tellurium|,
with the scale floored at 1e-10. Any non-finite value fails.
"""
import json
import math
import sqlite3
from pathlib import Path

import libsbml
import numpy as np
import pytest
from vivarium_workbench.testing import Run

STUDY = Path(__file__).resolve().parents[1]
WORKSPACE = next(p for p in STUDY.parents if (p / "workspace.yaml").is_file())
MODELS_DIR = STUDY.parents[1] / "inputs" / "models"
COMPOSITES = WORKSPACE / "viva_expressions" / "composites"
CONTROLS = json.loads((STUDY / "analysis" / "negative_controls.json").read_text(encoding="utf-8"))
DB = STUDY / "runs.db"

MODELS = (5, 10, 12, 41, 254, 400, 700, 1000)
TAU = {5: 1e-7}               # pre-registered; every other model is 1e-8
TAU_DEFAULT = 1e-8
CONVERGENCE_ORDERS = 2.5


def tau(model_id: int) -> float:
    return TAU.get(model_id, TAU_DEFAULT)


def latest(label: str) -> Run:
    """The most recent completed run launched as ``label``."""
    row = sqlite3.connect(DB).execute(
        "SELECT run_id FROM runs_meta WHERE label = ? AND status = 'completed' "
        "ORDER BY started_at DESC LIMIT 1", (label,)).fetchone()
    if row is None:
        pytest.fail(f"no completed run for {label!r}: the study's run set is incomplete")
    return Run(DB, row[0])


def _volumes(model_id: int) -> dict[str, float]:
    m = libsbml.readSBMLFromFile(str(MODELS_DIR / f"BIOMD{model_id:010d}.xml")).getModel()
    size = {c.getId(): c.getSize() for c in m.getListOfCompartments()}
    return {s.getId(): size[s.getCompartment()] for s in m.getListOfSpecies()}


def errors(model_id: int, ode: Run, tel: Run) -> dict[str, float]:
    """Normalized max error per compared variable, on the shared t = 0.5.. grid."""
    imported = json.loads((COMPOSITES / f"biomd{model_id:010d}_ode.import.json").read_text(
        encoding="utf-8"))
    t_ode, t_tel = ode.observable("time")[1:], tel.observable("model_time")[1:]
    assert len(t_ode) == len(t_tel) > 0 and np.array_equal(t_ode, t_tel), "model clocks differ"
    volume = _volumes(model_id)
    tel_parameters = set(tel.trajectory.columns)
    out = {}
    compared = [*imported["initial"],
                *(a for a in imported["assignments"]
                  if f"parameters.{imported['sbml_ids'][a]}" in tel_parameters)]
    for name in compared:
        kind, sid = imported["kinds"][name], imported["sbml_ids"][name]
        if name in imported["initial"] and kind in ("concentration", "amount"):
            ref = tel.observable(f"species.{sid}")
            ref = ref / volume[sid] if kind == "concentration" else ref
        else:
            ref = tel.observable(f"parameters.{sid}")
        ours = ode.observable(name)[1:]
        assert len(ours) == len(ref) == len(t_tel), f"{name}: missing samples"
        out[name] = (math.inf if not (np.all(np.isfinite(ours)) and np.all(np.isfinite(ref))) else
                     float(np.max(np.abs(ours - ref)) / max(float(np.max(np.abs(ref))), 1e-10)))
    return out


def worst(errs: dict[str, float]) -> tuple[str, float]:
    name = max(errs, key=errs.get)
    return name, errs[name]


@pytest.mark.parametrize("model_id", MODELS)
def test_v2_h1_tight_agreement_within_tau(model_id):
    name, err = worst(errors(model_id, latest(f"biomd{model_id}_ode_tight"),
                             latest(f"biomd{model_id}_tellurium_tight")))
    assert err <= tau(model_id), f"BIOMD{model_id}: {name} off by {err:.2e} > {tau(model_id):g}"


@pytest.mark.parametrize("model_id", MODELS)
def test_v2_h2_convergence(model_id):
    _, default = worst(errors(model_id, latest(f"biomd{model_id}_ode"),
                              latest(f"biomd{model_id}_tellurium")))
    _, tight = worst(errors(model_id, latest(f"biomd{model_id}_ode_tight"),
                            latest(f"biomd{model_id}_tellurium_tight")))
    orders = math.log10(default / max(tight, 1e-300))
    assert orders >= CONVERGENCE_ORDERS, (
        f"BIOMD{model_id}: default {default:.2e} / tight {tight:.2e} = {orders:.2f} orders")


@pytest.mark.parametrize("model_id", MODELS)
def test_v2_h3_median_sensitivity_control_is_detected(model_id):
    control = CONTROLS[f"BIOMD{model_id:010d}"]
    name, err = worst(errors(model_id, latest(f"biomd{model_id}_ode_ctrl_v2"),
                             latest(f"biomd{model_id}_tellurium_tight")))
    assert err > tau(model_id), (
        f"BIOMD{model_id}: +1% {control['sbml_id']} not detected ({name} off by only {err:.2e})")
