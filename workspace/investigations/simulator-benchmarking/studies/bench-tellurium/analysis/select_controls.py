"""Select the v2 negative controls (pre-registration H3), using the oracle side only.

For each model, every constant parameter is perturbed by a +1e-3 relative
change in libroadrunner, on the ORIGINAL SBML with local parameters promoted to
global ones (libsbml's promoteLocalParameters, the same semantics-preserving
renaming the importer applies, so the ids match the import's parameter names). Its
sensitivity is the v1 agreement metric of the perturbed trajectory against
the unperturbed one (max over floating species of max_t |delta| / max_t |ref|),
divided by 1e-3. The control for the model is the parameter of median
sensitivity among the nonzero ones, ties broken by SBML id. Nothing here
touches the OdeProcess side, so the choice cannot be fitted to the comparison
it tests.

    uv run python <this file>   ->  analysis/negative_controls.json
"""
import json
from pathlib import Path

import libsbml
import numpy as np
import roadrunner

HERE = Path(__file__).resolve().parent
MODELS = HERE.parents[2] / "inputs" / "models"
COMPOSITES = next(p for p in HERE.parents if (p / "workspace.yaml").is_file()) \
    / "viva_expressions" / "composites"
IDS = (5, 10, 12, 41, 254, 400, 700, 1000)
REL = 1e-3
T_END, N = 100.0, 201
TOL = {"rtol": 1e-12, "atol": 1e-14}


def simulate(rr) -> np.ndarray:
    rr.integrator.relative_tolerance, rr.integrator.absolute_tolerance = TOL["rtol"], TOL["atol"]
    rr.timeCourseSelections = list(rr.model.getFloatingSpeciesIds())
    return np.asarray(rr.simulate(0, T_END, N))


def metric(ref: np.ndarray, other: np.ndarray) -> float:
    scale = np.maximum(np.max(np.abs(ref), axis=0), 1e-10)
    return float(np.max(np.max(np.abs(other - ref), axis=0) / scale))


def promoted(path: Path) -> str:
    """The SBML text with local parameters promoted to global ones."""
    doc = libsbml.readSBMLFromFile(str(path))
    props = libsbml.ConversionProperties()
    props.addOption("promoteLocalParameters", True)
    if doc.convert(props) != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(f"promoteLocalParameters failed for {path.name}")
    return libsbml.writeSBMLToString(doc)


def candidates(sbml: str) -> list[str]:
    """Constant global parameters that no rule or initial assignment sets."""
    m = libsbml.readSBMLFromString(sbml).getModel()
    ruled = {r.getVariable() for r in m.getListOfRules()} | {
        a.getSymbol() for a in m.getListOfInitialAssignments()}
    return sorted(p.getId() for p in m.getListOfParameters()
                  if p.getConstant() and p.getId() not in ruled and p.isSetValue()
                  and p.getValue() != 0.0)


def select(model_id: int) -> dict:
    sbml = promoted(MODELS / f"BIOMD{model_id:010d}.xml")
    base = simulate(roadrunner.RoadRunner(sbml))
    sens = {}
    for pid in candidates(sbml):
        rr = roadrunner.RoadRunner(sbml)
        rr[pid] = rr[pid] * (1 + REL)
        sens[pid] = metric(base, simulate(rr)) / REL
    nonzero = sorted((s, pid) for pid, s in sens.items() if s > 0)
    s_med, p_med = nonzero[(len(nonzero) - 1) // 2]
    imported = json.loads((COMPOSITES / f"biomd{model_id:010d}_ode.import.json").read_text(
        encoding="utf-8"))
    name = {v: k for k, v in imported["sbml_ids"].items()}.get(p_med, p_med)
    return {"sbml_id": p_med, "import_name": name, "sensitivity": s_med,
            "value": imported["params"][name], "n_candidates": len(sens),
            "n_nonzero": len(nonzero)}


def main() -> None:
    out = {f"BIOMD{m:010d}": select(m) for m in IDS}
    (HERE / "negative_controls.json").write_text(json.dumps(out, indent=2) + "\n",
                                                encoding="utf-8")
    for k, v in out.items():
        print(k, v["sbml_id"], f"s={v['sensitivity']:.3g}", f"{v['n_nonzero']}/{v['n_candidates']}")


if __name__ == "__main__":
    main()
