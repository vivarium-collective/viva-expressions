"""Score trajectories against behavioral features.

Every feature maps to a residual vector scaled so that ``|r| == 1`` means
"exactly at tolerance": a trajectory satisfies a feature when all of its
residuals have ``|r| <= 1``. Measurements a feature depends on but that are
undefined for the trajectory (no oscillation to take a period of) score
``PENALTY``, which is finite so optimizers keep a usable gradient-free signal.
"""
from itertools import pairwise

import numpy as np
from scipy.signal import find_peaks

from viva_expressions.synthesize.spec import (
    Bounds,
    Feature,
    Monotonic,
    Oscillation,
    SettlingTime,
    SteadyState,
)

PENALTY = 10.0
# Features without a user tolerance (monotonic, bounds) allow violations up to
# this fraction of the trajectory's scale, absorbing integrator noise.
SLACK = 1e-3
# Peaks must stand out by this fraction of the trajectory's range.
PROMINENCE = 0.01
# Ranges below this fraction of the trajectory's magnitude count as flat.
FLAT = 1e-9


def _scale(y: np.ndarray) -> float:
    return max(float(np.max(np.abs(y))), 1e-12)


def settling_time(t: np.ndarray, y: np.ndarray, band: float) -> float:
    """First time after which ``y`` stays within ``band * |y_end - y_0|`` of ``y_end``."""
    step = abs(y[-1] - y[0])
    outside = np.flatnonzero(np.abs(y - y[-1]) > band * step)
    if step == 0 or outside.size == 0:
        return float(t[0])
    last = outside[-1]
    if last + 1 >= len(t):
        return float(t[-1])
    # interpolate the band crossing between the last outside and first inside sample
    d0 = abs(y[last] - y[-1]) - band * step
    d1 = abs(y[last + 1] - y[-1]) - band * step
    return float(t[last] + (t[last + 1] - t[last]) * d0 / (d0 - d1))


def oscillation_stats(t: np.ndarray, y: np.ndarray, window: float | None = None) -> dict | None:
    """Period, amplitude and decay of ``y``; ``None`` only if ``y`` is flat.

    ``period`` is the mean spacing of prominent peaks, or, with fewer than two
    peaks, the dominant FFT period (``t`` must then be uniform). ``amplitude``
    is half the peak-to-peak height in the last full ``window`` (default: the
    period) and ``decay`` is ``1 - last / second-to-last`` window amplitude: 0
    when sustained at the end of the run (start-up transients settling onto a
    limit cycle don't count against it), toward 1 when damped, 1 when fewer
    than two windows fit. A last window whose oscillation is below the peak
    prominence floor isn't a real oscillation, so decay rises continuously to
    1 as it shrinks below that floor (two windows of integrator noise after a
    damped transient would otherwise read as sustained). Both
    are defined, and vary smoothly, even for trajectories that never cycle, so
    an optimizer can climb toward oscillation instead of facing a flat penalty.
    """
    span = float(np.ptp(y))
    if span <= FLAT * _scale(y):
        return None
    peaks, _ = find_peaks(y, prominence=PROMINENCE * span)
    if len(peaks) >= 2:
        period = float(np.mean(np.diff(_refine(t, y, peaks))))
    else:
        power = np.abs(np.fft.rfft(y - y.mean()))
        k = 1 + np.argmax(power[1:])
        period = float(len(y) * (t[1] - t[0]) / k)
    window = window or period
    n = int((t[-1] - t[0]) // window)
    edges = t[0] + window * np.arange(n + 1)
    amps = [float(np.ptp(y[(t >= a) & (t <= b)])) / 2 for a, b in pairwise(edges)]
    if not amps:
        amps = [span / 2]
    decay = 1 - amps[-1] / amps[-2] if len(amps) >= 2 and amps[-2] > 0 else 1.0
    floor = PROMINENCE * span / 2
    decay = max(decay, 1 - amps[-1] / floor)
    return {"period": period, "amplitude": amps[-1], "decay": float(decay),
            "n_peaks": len(peaks)}


def _refine(t: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Sub-sample peak times by fitting a parabola through each peak's neighbors."""
    out = t[idx].astype(float)
    for k, i in enumerate(idx):
        if 0 < i < len(y) - 1:
            a, b, c = y[i - 1], y[i], y[i + 1]
            denom = a - 2 * b + c
            if denom != 0:
                out[k] = t[i] + 0.5 * (a - c) / denom * (t[i + 1] - t[i - 1]) / 2
    return out


def residuals(feature: Feature, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Scaled residuals of ``y`` (the series of ``feature.var``) against ``feature``."""
    if isinstance(feature, SteadyState):
        tail = y[t >= t[-1] - 0.1 * (t[-1] - t[0])]
        return np.array([(y[-1] - feature.value) / feature.tol,
                         float(np.ptp(tail)) / feature.tol])
    if isinstance(feature, SettlingTime):
        return np.array([(settling_time(t, y, feature.band) - feature.value) / feature.tol])
    if isinstance(feature, Monotonic):
        steps = np.diff(y) if feature.direction == "increasing" else -np.diff(y)
        reverse = float(-steps[steps < 0].sum())
        return np.array([reverse / (SLACK * _scale(y))])
    if isinstance(feature, Bounds):
        below = 0.0 if feature.min is None else max(feature.min - float(y.min()), 0.0)
        above = 0.0 if feature.max is None else max(float(y.max()) - feature.max, 0.0)
        return np.array([(below + above) / (SLACK * _scale(y))])
    if isinstance(feature, Oscillation):
        n = 2 + (feature.amplitude is not None) + feature.sustained
        stats = oscillation_stats(t, y, window=feature.period)
        if stats is None:
            return np.full(n, PENALTY)
        r = [0.0 if stats["n_peaks"] >= 2 else PENALTY,
             (stats["period"] - feature.period) / (feature.tol * feature.period)]
        if feature.amplitude is not None:
            r.append((stats["amplitude"] - feature.amplitude) / (feature.tol * feature.amplitude))
        if feature.sustained:
            r.append(max(stats["decay"], 0.0) / feature.tol)
        return np.array(r)
    raise TypeError(f"unknown feature {feature!r}")


def score(features, t: np.ndarray, series: dict[str, np.ndarray]) -> list[dict]:
    """Per-feature residuals and pass/fail, for fitting and reporting."""
    out = []
    for f in features:
        r = residuals(f, t, series[f.var])
        out.append({"feature": f, "residuals": r,
                    "worst": float(np.max(np.abs(r))),
                    "satisfied": bool(np.all(np.abs(r) <= 1.0))})
    return out
