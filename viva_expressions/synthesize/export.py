"""Export a fitted model as a process-bigraph document, verify it, and report.

Verification re-runs the exported document in a real ``Composite`` (through
``OdeProcess``, at its own tolerances) and re-scores it: the export is verified
only if the Composite's trajectory satisfies every behavior feature and agrees
with the fitter's trajectory, so fit and export can't drift apart unnoticed.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from viva_expressions.composites import ode_document, run_document
from viva_expressions.synthesize.behavior import score
from viva_expressions.synthesize.fit import FitResult, Problem
from viva_expressions.synthesize.spec import Spec

# Composite and fitter trajectories must agree to this fraction of each
# variable's scale (they integrate the same compiled RHS at different tolerances).
AGREEMENT_RTOL = 1e-3


@dataclass
class Verification:
    verified: bool
    feasible: bool
    max_deviation: float           # max |composite - fit| / scale over variables
    behavior: list[dict]           # behavior.score() on the Composite trajectory
    data_rmse: dict[str, float]    # per-variable RMSE of the Composite against data


def to_document(result: FitResult, spec: Spec, interval: float | None = None) -> dict:
    """The fitted model as a JSON-serializable document (one ``OdeProcess``)."""
    interval = interval or spec.time.t_end / (spec.time.n_points - 1)
    return ode_document(result.model.rhs, result.theta,
                        {v.name: v.value for v in spec.states},
                        {v.name: v.value for v in spec.inputs}, interval=interval)


def verify(result: FitResult, spec: Spec, document: dict) -> Verification:
    t, series = run_document(document, spec.time.t_end)
    problem = Problem(result.model, spec)
    behavior = score(problem.features, t, series)
    feasible = all(b["satisfied"] for b in behavior)
    # the fitted model, re-integrated at exactly the Composite's emit times, so
    # the comparison has no interpolation error whatever the interval
    Y = problem.simulate(np.array([result.theta[p] for p in problem.names]), t_eval=t)
    deviation = np.inf
    if Y is not None:
        deviation = max(
            float(np.max(np.abs(series[v] - Y[i])) / max(float(np.max(np.abs(Y[i]))), 1e-12))
            for i, v in enumerate(spec.state_names))
    rmse = {}
    if spec.data is not None:
        for v, obs in spec.data.columns.items():
            rmse[v] = float(np.sqrt(np.mean((np.interp(spec.data.t, t, series[v]) - obs) ** 2)))
    return Verification(feasible and deviation <= AGREEMENT_RTOL, feasible,
                        deviation, behavior, rmse)


def _feature(b: dict) -> dict:
    f = b["feature"]
    return {"kind": type(f).__name__, **asdict(f), "worst": round(b["worst"], 6),
            "satisfied": b["satisfied"]}


def _candidate(r: FitResult) -> dict:
    return {"name": r.model.name, "origin": r.model.origin, "rhs": r.model.rhs,
            "theta": r.theta, "k": r.k, "loss": r.loss, "aicc": r.aicc,
            "data_rss": r.data_rss, "feasible": r.feasible,
            "non_identifiable": r.non_identifiable,
            "behavior": [_feature(b) for b in r.behavior]}


def report(spec: Spec, ranked: list[FitResult], verification: Verification | None,
           notes: list[str]) -> dict:
    winner = ranked[0] if ranked and ranked[0].feasible else None
    return {
        "spec": spec.name,
        "status": ("verified" if verification and verification.verified else
                   "unverified" if winner else "infeasible"),
        "winner": _candidate(winner) if winner else None,
        "verification": None if verification is None else {
            "verified": verification.verified, "feasible": verification.feasible,
            "max_deviation": verification.max_deviation,
            "agreement_rtol": AGREEMENT_RTOL, "data_rmse": verification.data_rmse,
            "behavior": [_feature(b) for b in verification.behavior]},
        "candidates": [_candidate(r) for r in ranked],
        "constraints": list(spec.constraints),
        "notes": notes,
    }


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.4g}"


def markdown(rep: dict, top: int | None = None) -> str:
    lines = [f"# Synthesis report: {rep['spec']}", "", f"**Status:** {rep['status']}", ""]
    w = rep["winner"]
    if w:
        lines += ["## Winner", "", f"`{w['name']}` ({w['origin']}), k = {w['k']}", "", "```"]
        lines += [f"d{v}/dt = {e}" for v, e in w["rhs"].items()]
        lines += ["```", "", "| param | value |", "|---|---|"]
        lines += [f"| {p} | {x:.6g} |" for p, x in w["theta"].items()]
        if w["non_identifiable"]:
            lines += ["", (f"**Not identifiable from this spec:** {', '.join(w['non_identifiable'])}"
                           " — the behavior/data don't constrain them; any value along a flat"
                           " direction fits equally well.")]
    v = rep["verification"]
    if v:
        lines += ["", "## Composite verification", "",
                  (f"Re-run through a real Composite: verified = **{v['verified']}**, "
                   f"max deviation from fit = {v['max_deviation']:.2e} "
                   f"(limit {v['agreement_rtol']:g})."), "",
                  "| feature | var | worst |r| | satisfied |", "|---|---|---|---|"]
        lines += [f"| {b['kind']} | {b['var']} | {b['worst']:.3g} | {b['satisfied']} |"
                  for b in v["behavior"]]
        if v["data_rmse"]:
            lines += ["", "Data RMSE: " + ", ".join(f"{k} = {x:.3g}"
                                                    for k, x in v["data_rmse"].items())]
    cands = rep["candidates"][:top] if top else rep["candidates"]
    lines += ["", "## Candidates (ranked)", "",
              "| # | name | origin | k | feasible | AICc | loss | worst feature |",
              "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(cands, 1):
        worst = max(c["behavior"], key=lambda b: b["worst"], default=None)
        wtxt = f"{worst['kind']}({worst['var']}) {worst['worst']:.3g}" if worst else "—"
        lines.append(f"| {i} | `{c['name']}` | {c['origin']} | {c['k']} | {c['feasible']} | "
                     f"{_fmt(c['aicc'])} | {_fmt(c['loss'])} | {wtxt} |")
    if rep["constraints"]:
        lines += ["", "## Constraints (free text, not scored)", ""]
        lines += [f"- {c}" for c in rep["constraints"]]
    if rep["notes"]:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in rep["notes"]]
    return "\n".join(lines) + "\n"


def write(out_dir: str | Path, rep: dict, document: dict | None, top: int | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float),
                                     encoding="utf-8")
    (out / "report.md").write_text(markdown(rep, top), encoding="utf-8")
    if document is not None:
        (out / "composite.json").write_text(json.dumps(document, indent=2),
                                            encoding="utf-8")
