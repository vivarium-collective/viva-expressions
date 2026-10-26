"""expression-kinetics-analytic: the hypothesis in study.yaml ``hypothesis``.

* **H1.** At the default tolerances (rtol 1e-8, atol 1e-10) every analytic
  baseline lands within a relative error of 1e-6 of its oracle, for every
  method.
* **H2.** Error grows with rtol but stays inside a band set by that rtol:
  ``err <= 100 * rtol`` (the multiplier H1 implies: 1e-6 at rtol 1e-8), and
  ``err(tight) <= err(default) <= err(loose)`` per model.
* **H3.** dx/dt = x**2 raises the OdeProcess max_abs RuntimeError at
  t* = 1/x0 instead of stalling.

The metric and the band were fixed on 2026-10-06, after the runs and before
any error was computed (see the study's ``amendments``). Metric: per compared
quantity, the pointwise relative error max_t |q - q*| / |q*| over every
emitted time on t = 0..5 (every 0.5 for logistic and
Michaelis-Menten, every 0.1 for Lotka-Volterra, the processes' intervals). Every compared oracle stays away from zero
on that window; P (P(0) = 0) is checked through the conservation law S + P = S0
instead. Compared quantities:

* logistic: N against N*(t) = K / (1 + ((K - N0)/N0) exp(-r t));
* Michaelis-Menten: S against S*(t) = Km W((S0/Km) exp((S0 - Vmax t)/Km)),
  evaluated as Km * wrightomega(ln(S0/Km) + (S0 - Vmax t)/Km) (W(e^u) without
  overflow at Km = 1e-4), and S + P against S0;
* Lotka-Volterra: x and y against a reference solve (no closed form exists:
  scipy DOP853 at rtol 1e-13, atol 1e-15, one continuous solve evaluated at the
  emitted times), and the first integral V = d x - g ln x + b y - a ln y
  against V(0). V alone cannot catch a stalled or mistimed trajectory, which
  stays on its orbit.

S + P = S0 is a structural check (the two right-hand sides are exact
negatives, so every solver conserves it to ~1e-16; it catches P never
updating), not evidence of accuracy. Scope: on t = 0..5 the ``mm_stiff_*``
variants are not stiff yet (|dS'/dS| <= 4e-6; stiffness appears near t ~ 10).

Runs are read from runs.db: the latest completed run per label (H1, H2) and
the latest run per label (H3, which must be failed, with its recorded history
ending in the interval that contains t* and matching x0/(1 - x0 t)). H3 also
re-runs the composite in process, so the failure itself is the max_abs
RuntimeError, not just any failure.
"""
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from process_bigraph import Composite
from process_bigraph.composite_spec import CompositeSpec
from scipy.integrate import solve_ivp
from scipy.special import wrightomega
from vivarium_workbench.testing import Run

from viva_expressions.core import build_core

STUDY = Path(__file__).resolve().parents[1]
WORKSPACE = next(p for p in STUDY.parents if (p / "workspace.yaml").is_file())
COMPOSITES = WORKSPACE / "viva_expressions" / "composites"
DB = STUDY / "runs.db"

BAND = 100.0          # err <= BAND * rtol; 1e-6 at the default rtol 1e-8 (H1)
T_END = 5.0

# label -> (composite file stem, model)
ANALYTIC = {
    "logistic": ("logistic_growth", "logistic"),
    "logistic_rk45": ("logistic_growth", "logistic"),
    "logistic_radau": ("logistic_growth", "logistic"),
    "logistic_rtol_loose": ("logistic_growth", "logistic"),
    "logistic_rtol_tight": ("logistic_growth", "logistic"),
    "michaelis_menten": ("michaelis_menten", "mm"),
    "mm_bdf": ("michaelis_menten", "mm"),
    "mm_stiff_lsoda": ("michaelis_menten", "mm"),
    "mm_stiff_radau": ("michaelis_menten", "mm"),
    "mm_stiff_rk45": ("michaelis_menten", "mm"),
    "lotka_volterra": ("lotka_volterra", "lv"),
    "lv_dop853": ("lotka_volterra", "lv"),
    "lv_rk45_loose": ("lotka_volterra", "lv"),
    "lv_rtol_tight": ("lotka_volterra", "lv"),
}
# (tight, default, loose) per model, for H2's ordering
RTOL_LADDERS = {
    "logistic": ("logistic_rtol_tight", "logistic", "logistic_rtol_loose"),
    "lv": ("lv_rtol_tight", "lotka_volterra", "lv_rk45_loose"),
}
BLOWUP = {"quadratic_blowup": {}, "blowup_low_guard": {"max_abs": 1000}, "blowup_x0_2": {"x0": 2}}


def spec(stem: str) -> CompositeSpec:
    return CompositeSpec.from_file(COMPOSITES / f"{stem}.composite.yaml")


def latest(label: str, status: str | None = "completed") -> tuple[str, str]:
    """``(run_id, status)`` of the most recent run launched as ``label``
    (restricted to ``status`` unless it is None)."""
    query = "SELECT run_id, status FROM runs_meta WHERE label = ?"
    args: tuple = (label,)
    if status is not None:
        query, args = query + " AND status = ?", (label, status)
    row = sqlite3.connect(DB).execute(
        query + " ORDER BY started_at DESC LIMIT 1", args).fetchone()
    if row is None:
        pytest.fail(f"no {status or ''} run for {label!r}: the study's run set is incomplete")
    return row


def params(stem: str, run: Run) -> dict[str, float]:
    """The run's effective numeric parameters: the spec's defaults under its
    overrides (``method``, a string, left out)."""
    merged = {k: v.get("default") for k, v in spec(stem).parameters.items()}
    merged.update(run.params)
    return {k: float(v) for k, v in merged.items() if k != "method"}


def rel(q, exact) -> float:
    q, exact = np.asarray(q, dtype=float), np.asarray(exact, dtype=float)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(exact))):
        return math.inf
    assert np.all(exact != 0.0), "oracle reaches zero: the pointwise relative error is undefined"
    return float(np.max(np.abs(q - exact) / np.abs(exact)))


def errors(label: str) -> tuple[dict[str, float], float]:
    """Relative error per compared quantity, and the run's rtol."""
    stem, model = ANALYTIC[label]
    run = Run(DB, latest(label)[0])
    p = params(stem, run)
    t = run.observable("time")
    # global_time is a float sum of intervals (fifty 0.1 steps end at 4.999999999999998);
    # the oracles are evaluated at the emitted times themselves
    assert t[0] == 0.0 and math.isclose(t[-1], T_END, rel_tol=1e-12) and np.all(np.diff(t) > 0), \
        f"{label}: time does not run 0..{T_END}: {t}"
    if model == "logistic":
        N0, K, r = p["N0"], p["K"], p["r"]
        out = {"N": rel(run.observable("N"), K / (1 + (K - N0) / N0 * np.exp(-r * t)))}
    elif model == "mm":
        S0, Km, Vmax = p["S0"], p["Km"], p["Vmax"]
        S, P = run.observable("S"), run.observable("P")
        exact_S = Km * wrightomega(np.log(S0 / Km) + (S0 - Vmax * t) / Km).real
        out = {"S": rel(S, exact_S), "S+P": rel(S + P, np.full_like(t, S0))}
    else:
        a, b, d, g = p["a"], p["b"], p["d"], p["g"]
        x, y = run.observable("x"), run.observable("y")
        reference = solve_ivp(lambda _, u: [a * u[0] - b * u[0] * u[1], d * u[0] * u[1] - g * u[1]],
                              (0.0, t[-1]), [p["x0"], p["y0"]], method="DOP853",
                              t_eval=t, rtol=1e-13, atol=1e-15)
        assert reference.success, reference.message
        V = d * x - g * np.log(x) + b * y - a * np.log(y)
        V0 = d * p["x0"] - g * math.log(p["x0"]) + b * p["y0"] - a * math.log(p["y0"])
        out = {"x": rel(x, reference.y[0]), "y": rel(y, reference.y[1]),
               "V": rel(V, np.full_like(t, V0))}
    return out, p["rtol"]


@pytest.mark.parametrize("label", ANALYTIC)
def test_error_within_the_rtol_band(label):
    """H1 (default rtol: <= 1e-6) and H2's band: every compared quantity's
    relative error <= 100 * rtol."""
    errs, rtol = errors(label)
    bad = {k: e for k, e in errs.items() if not e <= BAND * rtol}
    assert not bad, f"{label}: {bad} exceed {BAND} * rtol = {BAND * rtol:g}"


@pytest.mark.parametrize("model", RTOL_LADDERS)
def test_error_grows_with_rtol(model):
    """H2: err(tight) <= err(default) <= err(loose), worst quantity per run."""
    worst = [max(errors(label)[0].values()) for label in RTOL_LADDERS[model]]
    assert worst[0] <= worst[1] <= worst[2], f"{model}: tight/default/loose errors {worst}"


@pytest.mark.parametrize("label", BLOWUP)
def test_blowup_run_is_recorded_failed_at_the_singularity(label):
    """H3: the latest recorded run of each blow-up label failed, its history
    ends in the interval that contains t* = 1/x0 (none stalled or completed
    past the singularity), and its last state matches x0 / (1 - x0 t)."""
    run_id, status = latest(label, status=None)
    assert status == "failed"
    run = Run(DB, run_id)
    p = params("quadratic_blowup", run)
    interval = spec("quadratic_blowup").state["blowup"]["interval"]
    t, x = run.observable("time"), run.observable("x")
    assert t[-1] < 1 / p["x0"] <= t[-1] + interval + 1e-12, \
        f"{label}: recorded history ends at t={t[-1]}, singularity at {1 / p['x0']}"
    assert rel(x, p["x0"] / (1 - p["x0"] * t)) <= BAND * p["rtol"]


@pytest.mark.parametrize("label, overrides", BLOWUP.items())
def test_blowup_raises_max_abs_at_the_singularity(label, overrides):
    """H3: a real run raises the max_abs RuntimeError in the interval that
    contains t* = 1/x0, and the state at the last completed time still matches
    x0 / (1 - x0 t) within the band."""
    blowup_spec = spec("quadratic_blowup")
    document = blowup_spec.to_document(overrides=overrides)
    composite = Composite(document, core=build_core())
    with pytest.raises(RuntimeError, match="max_abs"):
        composite.run(T_END)
    p = {k: v.get("default") for k, v in blowup_spec.parameters.items()} | overrides
    x0, rtol = float(p["x0"]), float(p["rtol"])
    interval = document["state"]["blowup"]["interval"]
    reached = composite.state["global_time"]
    assert reached < 1 / x0 <= reached + interval + 1e-12, \
        f"{label}: raised after t={reached}, singularity at {1 / x0}"
    assert rel(composite.state["stores"]["x"], x0 / (1 - x0 * reached)) <= BAND * rtol
