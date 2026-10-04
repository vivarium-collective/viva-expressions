"""bench-tellurium: the pre-registered hypotheses H1-H3 (see study.yaml).

Each model is run twice through the workbench: the /sbml-system import in
OdeProcess (``biomd<N>_ode``), and the original SBML in TelluriumProcess
(``biomd<N>_tellurium``). Both runs use the same 0.5 interval, at default
tolerances and at tight ones (``_tight``).

Agreement metric (fixed before any run): per compared variable,
``max_t |ode - tellurium| / max_t |tellurium|`` (scale floored at 1e-10). Any
non-finite value counts as failure. Units: TelluriumProcess reports species
AMOUNTS (roadrunner's bare-id selection), while OdeProcess integrates each
species in its symbol's units. A concentration species is therefore compared
with ``amount / compartment size``, taking the size from the pinned SBML.
Parameter, compartment and time states are compared with Tellurium's
``parameters`` and ``model_time``.

Time grid: Tellurium's first emitted row (t = 0) has empty maps, because its
stores are not initialised from ``TelluriumProcess.initial_state``. So the
comparison covers t = 0.5..100, and the two clocks are asserted identical
there. The t = 0 values are the SBML initial values, which the importer's
own tests check.
"""
import json
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
DB = STUDY / "runs.db"

MODELS = (5, 10, 12, 41, 254, 400, 700, 1000)
AGREE = 1e-6          # H1 threshold, pre-registered


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
    """Normalized max error per OdeProcess state, on the shared t = 0.5.. grid."""
    imported = json.loads((COMPOSITES / f"biomd{model_id:010d}_ode.import.json").read_text(
        encoding="utf-8"))
    clock = "time_" if "time" in imported["initial"] else "time"
    t_ode, t_tel = ode.observable(clock)[1:], tel.observable("time")[1:]
    assert len(t_ode) == len(t_tel) > 0 and np.array_equal(t_ode, t_tel), "time grids differ"
    volume = _volumes(model_id)
    out = {}
    for name in imported["initial"]:
        kind, sid = imported["kinds"][name], imported["sbml_ids"][name]
        if kind == "time":
            ref = tel.observable("model_time")[1:]
        elif kind in ("concentration", "amount"):
            ref = tel.observable(f"species.{sid}")
            ref = ref / volume[sid] if kind == "concentration" else ref
        else:
            ref = tel.observable(f"parameters.{sid}")
        ours = ode.observable(name)[1:]
        assert len(ours) == len(ref) == len(t_tel), f"{name}: missing samples"
        out[name] = (np.inf if not np.all(np.isfinite(ours)) else
                     float(np.max(np.abs(ours - ref)) / max(float(np.max(np.abs(ref))), 1e-10)))
    return out


def worst(errs: dict[str, float]) -> tuple[str, float]:
    name = max(errs, key=errs.get)
    return name, errs[name]


@pytest.mark.parametrize("model_id", MODELS)
def test_h1_tight_tolerance_agreement(model_id):
    """H1: at rtol 1e-12 / atol 1e-14 every state agrees within 1e-6 of scale."""
    name, err = worst(errors(model_id, latest(f"biomd{model_id}_ode_tight"),
                             latest(f"biomd{model_id}_tellurium_tight")))
    assert err <= AGREE, f"BIOMD{model_id}: {name} off by {err:.2e}"


@pytest.mark.parametrize("model_id", MODELS)
def test_h2_default_disagreement_is_integrator_error(model_id):
    """H2: at default tolerances the disagreement is never smaller than at tight ones."""
    _, default = worst(errors(model_id, latest(f"biomd{model_id}_ode"),
                              latest(f"biomd{model_id}_tellurium")))
    _, tight = worst(errors(model_id, latest(f"biomd{model_id}_ode_tight"),
                            latest(f"biomd{model_id}_tellurium_tight")))
    assert default >= tight, f"BIOMD{model_id}: default {default:.2e} < tight {tight:.2e}"


def test_h3_negative_control_is_detected():
    """H3: a 0.1% change of one parameter (BIOMD12 eff) exceeds the 1e-6 threshold."""
    name, err = worst(errors(12, latest("biomd12_ode_eff_plus_0p1pct"),
                             latest("biomd12_tellurium_tight")))
    assert err > AGREE, f"negative control not detected: {name} off by only {err:.2e}"
