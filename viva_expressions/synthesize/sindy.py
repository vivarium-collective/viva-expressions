"""Discover polynomial candidate structures from data with PySINDy.

Sweeps polynomial degree and an STLSQ threshold grid (scaled to each degree's
largest unthresholded coefficient), keeps each distinct sparsity pattern once,
and drops patterns that zero out a state that varies. Every surviving pattern
becomes a Model with SINDy's coefficients as initial values; each is polished
against the integrated trajectory and AICc decides among them and the other
candidates (Mangan et al. 2017). No pre-ranking: AIC on the derivative
regression, or on unpolished trajectories, favors dense patterns whose extra
terms absorb finite-difference error, and drops the true sparse structure.
"""
import warnings

import numpy as np

from viva_expressions.synthesize.motifs import Model
from viva_expressions.synthesize.spec import Spec

DEGREES = (1, 2, 3)
THRESHOLD_FRACTIONS = np.logspace(-3, -0.5, 8)


class SindyUnavailable(Exception):
    """SINDy cannot run for this spec; the message says why."""


def _monomial(states: tuple[str, ...], powers) -> tuple[str, str]:
    """``(expression, identifier-safe name)`` for one library term."""
    parts = [(v, int(p)) for v, p in zip(states, powers) if p]
    if not parts:
        return "1", "const"
    expr = "*".join(v if p == 1 else f"{v}**{p}" for v, p in parts)
    name = "_".join(v if p == 1 else f"{v}{p}" for v, p in parts)
    return expr, name


def discover(spec: Spec) -> list[Model]:
    """Distinct viable SINDy sparsity patterns as Models, sparsest first."""
    try:
        import pysindy as ps
    except ImportError as e:
        raise SindyUnavailable(
            "pysindy is not installed; install the extra with `uv sync --extra sindy`") from e
    if spec.data is None:
        raise SindyUnavailable("no data")
    states = spec.state_names
    missing = [v for v in states if v not in spec.data.columns]
    if missing:
        raise SindyUnavailable(f"data lacks columns for states {missing}; "
                               f"SINDy needs every state observed")
    X = np.column_stack([spec.data.columns[v] for v in states])
    t = spec.data.t
    varies = np.std(X, axis=0) > 0
    # one shared derivative estimate, so every pattern is scored against the same target
    x_dot = ps.FiniteDifference()(X, t)

    patterns: dict[frozenset, tuple[float, int, np.ndarray, np.ndarray]] = {}
    for degree in DEGREES:
        def fitted(threshold, degree=degree):
            m = ps.SINDy(optimizer=ps.STLSQ(threshold=threshold),
                         feature_library=ps.PolynomialLibrary(degree=degree))
            with warnings.catch_warnings():
                # the sweep's upper thresholds empty some equations by design;
                # those patterns are dropped below
                warnings.filterwarnings("ignore", message="Sparsity parameter is too big")
                m.fit(X, t=t, x_dot=x_dot, feature_names=list(states))
            return m
        cmax = float(np.max(np.abs(fitted(0.0).coefficients())))
        for frac in THRESHOLD_FRACTIONS:
            m = fitted(frac * cmax)
            C, P = m.coefficients(), m.feature_library.powers_
            if np.any(varies & ~np.any(C != 0, axis=1)):
                continue
            key = frozenset((i, tuple(P[j])) for i, j in zip(*np.nonzero(C)))
            if key in patterns:
                continue
            patterns[key] = (degree, C, P)

    models = sorted((_model(states, *p) for p in patterns.values()),
                    key=lambda m: len(m.params))
    return [Model(f"sindy#{i + 1}({m.name})", m.rhs, m.params, m.initial, m.origin)
            for i, m in enumerate(models)]


def _model(states: tuple[str, ...], degree: int, C: np.ndarray, P: np.ndarray) -> Model:
    rhs, params, initial = {}, {}, {}
    for i, v in enumerate(states):
        terms = []
        for j in np.flatnonzero(C[i]):
            expr, name = _monomial(states, P[j])
            p, c = f"c_{v}_{name}", float(C[i, j])
            params[p] = (min(0.0, 2 * c), max(0.0, 2 * c))  # sign-preserving, +/-100%
            initial[p] = c
            terms.append(p if expr == "1" else f"{p}*{expr}")
        rhs[v] = " + ".join(terms)
    return Model(f"degree {degree}, {len(params)} terms", rhs, params,
                 initial=initial, origin="sindy")
