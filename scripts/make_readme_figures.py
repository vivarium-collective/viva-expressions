"""Regenerate the figures embedded in README.md (docs/img/*.png).

    uv run python scripts/make_readme_figures.py

Every curve is computed here from the package's own code, not drawn by hand:
``oracles.png`` integrates the four study baselines with OdeProcess and compares
them to their closed forms; ``synthesis.png`` runs three of the systems/examples
specs through the synthesis pipeline (about three minutes in total, dominated by
the repressor search).
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wrightomega

from viva_expressions.composites import ode_document, run_document
from viva_expressions.synthesize import load_spec, synthesize
from viva_expressions.synthesize.fit import Problem

warnings.filterwarnings("ignore", message="The following arguments have no effect")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def logistic(t, r=0.5, K=100.0, N0=5.0):
    return K / (1 + (K - N0) / N0 * np.exp(-r * t))


def michaelis_menten(t, Km, Vmax=1.0, S0=10.0):
    """S(t) = Km * W((S0/Km) exp((S0 - Vmax t)/Km)); wrightomega(z) = W(exp(z)) never overflows."""
    return Km * wrightomega(np.log(S0 / Km) + (S0 - Vmax * t) / Km)


def trajectory(result, spec):
    """The fitted model's states on the spec grid, re-simulated so that an infeasible
    candidate (whose stored series may be empty) is still drawn as what it does."""
    problem = Problem(result.model, spec)
    Y = problem.simulate(np.array([result.theta[p] for p in problem.names]), t_eval=spec.time.grid)
    return None if Y is None else dict(zip(spec.state_names, Y))


def draw(ax, spec, result, var, style, **kw):
    """Plot ``var`` of a fitted candidate; one that diverges past max_abs is labeled, not drawn."""
    traj = trajectory(result, spec)
    label = f"{result.model.name[:34]}  feasible={result.feasible}"
    if traj is None:
        ax.plot([], [], style, label=label + " (diverges)", **kw)
    else:
        ax.plot(spec.time.grid, traj[var], style, label=label, **kw)


def oracles():
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    # (a) logistic: relative error against the closed form, per solver setting
    a = ax[0, 0]
    for label, kw in [("LSODA  rtol 1e-8 (default)", {}),
                      ("RK45   rtol 1e-8", {"method": "RK45"}),
                      ("Radau  rtol 1e-8", {"method": "Radau"}),
                      ("LSODA  rtol 1e-4", {"rtol": 1e-4, "atol": 1e-6})]:
        doc = ode_document({"N": "r*N*(1 - N/K)"}, {"r": 0.5, "K": 100.0}, {"N": 5.0},
                           interval=0.5, **kw)
        t, s = run_document(doc, 20.0)
        a.semilogy(t[1:], np.abs(s["N"][1:] - logistic(t[1:])) / logistic(t[1:]), label=label)
    a.axhline(1e-6, color="k", ls=":", lw=1)
    a.text(0.3, 1.4e-6, "study hypothesis: 1e-6", fontsize=8)
    a.set(title="Logistic growth: relative error vs closed form", xlabel="time",
          ylabel="|N - N_exact| / N_exact")
    a.legend(fontsize=7, loc="lower right")

    # (b) stiff Michaelis-Menten: Km/S0 = 1e-5, substrate collapses in a boundary layer
    b = ax[0, 1]
    Km = 1e-4
    doc = ode_document({"S": "-Vmax*S/(Km + S)", "P": "Vmax*S/(Km + S)"},
                       {"Vmax": 1.0, "Km": Km}, {"S": 10.0, "P": 0.0},
                       interval=0.5, method="Radau")
    t, s = run_document(doc, 12.0)
    tt = np.linspace(0, 12, 1200)
    b.plot(tt, michaelis_menten(tt, Km), "k-", lw=1.5, label="closed form (Lambert W)")
    b.plot(t, s["S"], "o", ms=4, label="OdeProcess, Radau")
    b.set(title="Michaelis-Menten, Km = 1e-4 (stiff as S -> 0)", xlabel="time",
          ylabel="substrate S")
    b.legend(fontsize=8)

    # (c) Lotka-Volterra: the first integral V is conserved along the exact flow
    c = ax[1, 0]
    a_, b_, d_, g_ = 1.0, 0.1, 0.075, 1.5
    doc = ode_document({"x": "a*x - b*x*y", "y": "d*x*y - g*y"},
                       {"a": a_, "b": b_, "d": d_, "g": g_}, {"x": 10.0, "y": 5.0}, interval=0.1)
    t, s = run_document(doc, 50.0)
    V = d_ * s["x"] - g_ * np.log(s["x"]) + b_ * s["y"] - a_ * np.log(s["y"])
    c.plot(s["x"], s["y"], lw=1)
    c.set(title=f"Lotka-Volterra orbit; max |V - V0| = {np.max(np.abs(V - V[0])):.1e}",
          xlabel="prey x", ylabel="predator y")

    # (d) x' = x^2 blows up at t* = 1/x0; the guard must raise, not stall
    d = ax[1, 1]
    tt = 1 - np.logspace(0, -12, 600)  # approach t* = 1 geometrically
    d.semilogy(tt, 1 / (1 - tt), "k-", label="exact  x0/(1 - x0 t),  x0 = 1")
    d.axvline(1.0, color="r", ls="--", lw=1)
    d.axhline(1e12, color="r", ls=":", lw=1)
    d.text(0.02, 2e12, "max_abs = 1e12 (OdeProcess raises here)", color="r", fontsize=8)
    d.text(1.005, 3, "t* = 1", color="r", fontsize=8)
    d.set(title="dx/dt = x**2: finite-time singularity", xlabel="time", ylabel="x", xlim=(0, 1.08))
    d.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT / "oracles.png", dpi=130)
    plt.close(fig)


def synthesis():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    examples = ROOT / "systems" / "examples"

    # logistic: qualitative spec only; the rival grows without bound
    spec = load_spec(examples / "logistic.yml")
    res = synthesize(spec)
    a = ax[0]
    for r, style in zip(res.ranked[:2], ["-", "--"]):
        draw(a, spec, r, "N", style)
    a.axhspan(4.9, 5.1, color="g", alpha=0.2, label="steady_state 5 +/- 0.1")
    a.set(title="logistic.yml: behavior only", xlabel="time", ylabel="N", ylim=(0, 10))
    a.legend(fontsize=7, loc="upper left")

    # repressor: a period-20 oscillator; low cooperativity cannot do it
    spec = load_spec(examples / "repressor.yml")
    res = synthesize(spec)
    a = ax[1]
    for r, style in zip(res.ranked[:2], ["-", "--"]):
        draw(a, spec, r, "r", style, lw=1.2)
    a.set(title="repressor.yml: period ~ 20", xlabel="time", ylabel="repressor r")
    a.legend(fontsize=7, loc="upper right")

    # Lotka-Volterra: structure and parameters recovered from noise-free data
    spec = load_spec(examples / "lotka_volterra.yml")
    res = synthesize(spec)
    a = ax[2]
    w = res.ranked[0]
    for var, col in [("x", "C0"), ("y", "C1")]:
        a.plot(spec.data.t[::8], spec.data.columns[var][::8], "o", ms=3, color=col,
               label=f"data: {'prey' if var == 'x' else 'predator'}")
        a.plot(w.t, w.series[var], "-", color=col, lw=1)
    a.set(title="lotka_volterra.yml: fit to lv.csv", xlabel="time", ylabel="population")
    a.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "synthesis.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    oracles()
    synthesis()
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
