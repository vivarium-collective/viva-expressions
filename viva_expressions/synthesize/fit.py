"""Fit a candidate model's parameters to data residuals and behavior residuals.

The objective is one residual vector: data residuals (each variable normalized
by its observed spread, the whole block by ``sqrt(n)`` so its squared norm is a
mean) followed by behavior residuals (``|r| <= 1`` at tolerance). Variable
``bounds`` from the spec join the behavior as implicit ``Bounds`` features.

Without data the features are the targets, so fits are centered on them. With
data, the data defines the optimum and features are constraints: only the part
of each residual beyond tolerance is penalized, so a satisfied feature can't
pull the fit off the data. Feasibility always uses the raw residuals.
A failed or diverging simulation scores a finite penalty, never NaN.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares

from viva_expressions.expressions import CompiledRHS, compile_rhs
from viva_expressions.synthesize.behavior import PENALTY, residuals, score
from viva_expressions.synthesize.motifs import Model
from viva_expressions.synthesize.spec import Bounds, Spec

MAX_ABS = 1e12          # divergence bound, matching OdeProcess's default
# Fitting tolerances: behavior tolerances are >= 1e-3 relative, and every
# winner is re-scored through the exported OdeProcess at its tighter defaults.
RTOL, ATOL = 1e-6, 1e-8
IDENTIFIABLE_RTOL = 1e-3  # singular values below this fraction of the largest are flat
# ...or below this absolute value: residuals are in tolerance units (1 = at
# tolerance) and Jacobian columns per relative parameter change, so a 100%
# parameter change moving residuals by < 1% of a tolerance is flat even when
# every direction is (e.g. a single parameter the spec doesn't constrain).
IDENTIFIABLE_ATOL = 1e-2
POPSIZE = 10
# Stop differential evolution once the best loss improves by less than this
# (relative, with an absolute floor) over this many generations. Penalized
# members keep the population's spread wide, so DE's own variance criterion
# rarely fires once some members fit and others still fail.
# DE only needs to find the basin; least squares polishes from there.
STALL_GENERATIONS = 20
STALL_RTOL, STALL_ATOL = 1e-2, 1e-9


@dataclass
class FitResult:
    model: Model
    theta: dict[str, float]
    loss: float
    data_rss: float | None      # sum of squared normalized data errors
    n_data: int
    behavior: list[dict]        # behavior.score() on the fitted trajectory
    feasible: bool
    non_identifiable: list[str]
    t: np.ndarray               # behavior grid
    series: dict[str, np.ndarray] | None  # fitted trajectory on ``t``; None if it failed

    @property
    def k(self) -> int:
        return len(self.theta)

    @property
    def aicc(self) -> float | None:
        """Corrected AIC from the data residuals; ``None`` without data."""
        n, k = self.n_data, self.k
        if not n or self.data_rss is None or n - k - 1 <= 0:
            return None
        rss = max(self.data_rss, 1e-300)
        return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def merge_times(grid: np.ndarray, data_t: np.ndarray, tol: float):
    """Sorted union of two time arrays, treating points within ``tol`` as one.

    Returns ``(t, grid_idx, data_idx)`` with ``t[grid_idx] == grid`` and
    ``t[data_idx]`` within ``tol`` of ``data_t`` (CSV round-trips perturb
    times in their last digits, which a plain union would duplicate).
    """
    near = np.searchsorted(grid, data_t).clip(1, max(len(grid) - 1, 1))
    left, right = grid[near - 1], grid[near.clip(max=len(grid) - 1)]
    snapped = np.where(np.abs(data_t - left) <= np.abs(right - data_t), left, right)
    extra = data_t[np.abs(snapped - data_t) > tol]
    t = np.union1d(grid, extra)
    data_on = np.where(np.abs(snapped - data_t) <= tol, snapped, data_t)
    return t, np.searchsorted(t, grid), np.searchsorted(t, data_on)


class Problem:
    """A model compiled against a spec, evaluating residuals for parameter vectors."""

    def __init__(self, model: Model, spec: Spec):
        self.model, self.spec = model, spec
        self.names = list(model.params)
        self.bounds = np.array([model.params[p] for p in self.names], dtype=float)
        self.rhs: CompiledRHS = compile_rhs(
            model.rhs, spec.state_names, spec.input_names, self.names)
        self.u = np.array([v.value for v in spec.inputs], dtype=float)
        self.y0 = np.array([v.value for v in spec.states], dtype=float)
        self.grid = spec.time.grid
        data_t = spec.data.t if spec.data is not None else np.empty(0)
        self.t_eval, self.grid_idx, self.data_idx = merge_times(
            self.grid, data_t, tol=1e-9 * spec.time.t_end)
        self.features = list(spec.behavior.features) + [
            Bounds(v.name, *v.bounds) for v in spec.states if v.bounds != (None, None)]
        self.n_feature_res = sum(
            len(residuals(f, self.grid, np.zeros_like(self.grid) + 1.0))
            for f in self.features)
        self.data_obs = (spec.data.columns if spec.data is not None else {})
        self.n_data = sum(len(y) for y in self.data_obs.values())
        # a (near-)constant column is scaled by 0.1% of its magnitude, not ~0
        self.sigma = {v: max(float(np.std(y)), 1e-3 * float(np.max(np.abs(y))), 1e-12)
                      for v, y in self.data_obs.items()}
        w = spec.fit.weights
        self.w_data, self.w_behavior = np.sqrt(w["data"]), np.sqrt(w["behavior"])

    def simulate(self, theta: np.ndarray, t_eval: np.ndarray | None = None) -> np.ndarray | None:
        """States at ``t_eval`` (default: the problem's grid plus data times), shape
        ``(n_states, n_t)``, or ``None`` on failure/divergence."""
        f, jac, u = self.rhs.f, self.rhs.jac, self.u
        t_eval = self.t_eval if t_eval is None else t_eval

        def diverged(t, y):
            return MAX_ABS - np.max(np.abs(y))
        diverged.terminal = True

        for method in ("LSODA", "BDF"):
            with np.errstate(all="ignore"):
                sol = solve_ivp(lambda t, y: f(y, u, theta), (0.0, self.spec.time.t_end),
                                self.y0, method=method, t_eval=t_eval,
                                jac=lambda t, y: jac(y, u, theta), rtol=RTOL, atol=ATOL,
                                events=diverged)
            if sol.status == 0 and sol.y.shape[1] == len(t_eval) \
                    and np.all(np.isfinite(sol.y)):
                return sol.y
            if sol.status == 1:  # diverged; another method won't help
                return None
        return None

    def series(self, Y: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
        return {v: Y[i, idx] for i, v in enumerate(self.spec.state_names)}

    def data_residuals(self, Y: np.ndarray | None) -> np.ndarray:
        if not self.n_data:
            return np.empty(0)
        if Y is None:
            return np.full(self.n_data, PENALTY / np.sqrt(self.n_data))
        sim = self.series(Y, self.data_idx)
        return np.concatenate([(sim[v] - obs) / self.sigma[v]
                               for v, obs in self.data_obs.items()]) / np.sqrt(self.n_data)

    def behavior_residuals(self, Y: np.ndarray | None) -> np.ndarray:
        if not self.features:
            return np.empty(0)
        if Y is None:
            return np.full(self.n_feature_res, PENALTY)
        s = self.series(Y, self.grid_idx)
        r = np.concatenate([residuals(f, self.grid, s[f.var]) for f in self.features])
        if self.n_data:
            r = np.sign(r) * np.maximum(np.abs(r) - 1.0, 0.0)
        return r

    def residuals(self, theta: np.ndarray) -> np.ndarray:
        Y = self.simulate(theta)
        return np.concatenate([self.w_data * self.data_residuals(Y),
                               self.w_behavior * self.behavior_residuals(Y)])

    def loss(self, theta: np.ndarray) -> float:
        return float(np.sum(self.residuals(theta) ** 2))


def non_identifiable(problem: Problem, theta: np.ndarray) -> list[str]:
    """Parameters lying along flat directions of the residuals at ``theta``.

    Columns of the central-difference Jacobian are scaled by each parameter's
    magnitude (or its bound width, near zero), so sensitivities compare across
    units. Directions with singular value below ``IDENTIFIABLE_RTOL`` of the
    largest, or below ``IDENTIFIABLE_ATOL``, are flat; a parameter weighing
    > 0.3 in any of them is flagged.
    """
    lo, hi = problem.bounds.T
    scale = np.maximum(np.abs(theta), 1e-3 * (hi - lo))
    J = np.empty((len(problem.residuals(theta)), len(theta)))
    for i in range(len(theta)):
        h = 1e-4 * scale[i]
        up, dn = theta.copy(), theta.copy()
        up[i], dn[i] = min(theta[i] + h, hi[i]), max(theta[i] - h, lo[i])
        J[:, i] = (problem.residuals(up) - problem.residuals(dn)) / (up[i] - dn[i]) * scale[i]
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    if s[0] == 0:
        return list(problem.names)
    flat = Vt[s < max(IDENTIFIABLE_RTOL * s[0], IDENTIFIABLE_ATOL)]
    return [p for i, p in enumerate(problem.names) if np.any(np.abs(flat[:, i]) > 0.3)]


def fit_model(model: Model, spec: Spec) -> FitResult:
    """Fit ``model``'s parameters, then polish locally with least squares.

    A model carrying ``initial`` values (e.g. SINDy coefficients, already in the
    right basin) is polished from them directly; otherwise differential
    evolution searches the full bounds first.
    """
    problem = Problem(model, spec)
    lo, hi = problem.bounds.T
    if model.initial:
        theta = np.clip([model.initial[p] for p in problem.names], lo, hi)
        loss = problem.loss(theta)
    else:
        best: list[float] = []

        def stalled(intermediate_result) -> bool:
            best.append(float(intermediate_result.fun))
            if len(best) <= STALL_GENERATIONS:
                return False
            before = best[-1 - STALL_GENERATIONS]
            return before - best[-1] < max(STALL_RTOL * abs(before), STALL_ATOL)

        de = differential_evolution(problem.loss, problem.bounds, rng=spec.fit.seed,
                                    maxiter=spec.fit.maxiter, popsize=POPSIZE, polish=False,
                                    init="sobol", callback=stalled)
        theta, loss = de.x, float(de.fun)
    ls = least_squares(problem.residuals, np.clip(theta, lo, hi), bounds=(lo, hi),
                       x_scale="jac", max_nfev=200 * len(theta))
    ls_loss = float(np.sum(ls.fun ** 2))
    if ls_loss < loss:
        theta, loss = ls.x, ls_loss

    Y = problem.simulate(theta)
    data_r = problem.data_residuals(Y) * np.sqrt(problem.n_data or 1)
    behavior = (score(problem.features, problem.grid, problem.series(Y, problem.grid_idx))
                if Y is not None else
                [{"feature": f, "residuals": np.array([PENALTY]), "worst": PENALTY,
                  "satisfied": False} for f in problem.features])
    return FitResult(
        model=model,
        theta={p: float(x) for p, x in zip(problem.names, theta)},
        loss=loss,
        data_rss=float(np.sum(data_r ** 2)) if problem.n_data else None,
        n_data=problem.n_data,
        behavior=behavior,
        feasible=Y is not None and all(b["satisfied"] for b in behavior),
        # flat penalties around a failed simulation would flag every parameter
        non_identifiable=non_identifiable(problem, theta) if theta.size and Y is not None else [],
        t=problem.grid,
        series=problem.series(Y, problem.grid_idx) if Y is not None else None,
    )
