from itertools import pairwise

import numpy as np
import pytest

from viva_expressions.synthesize.behavior import (
    PENALTY,
    oscillation_stats,
    residuals,
    score,
    settling_time,
)
from viva_expressions.synthesize.spec import (
    Bounds,
    Monotonic,
    Oscillation,
    SettlingTime,
    SteadyState,
)

T = np.linspace(0, 40, 4001)


def test_sine_period_and_amplitude():
    stats = oscillation_stats(T, np.sin(T))
    assert stats["period"] == pytest.approx(2 * np.pi, rel=1e-4)
    assert stats["amplitude"] == pytest.approx(1.0, rel=1e-4)
    assert abs(stats["decay"]) < 1e-3


def test_damped_oscillation_reports_decay():
    # y(t + P) = exp(-0.05 P) y(t) for P = pi, so each period-wide window
    # shrinks by exactly that factor
    stats = oscillation_stats(T, np.exp(-0.05 * T) * np.cos(2 * T))
    assert stats["period"] == pytest.approx(np.pi, rel=1e-3)
    assert stats["decay"] == pytest.approx(1 - np.exp(-0.05 * np.pi), rel=1e-2)


def test_exponential_settling_time_matches_analytic():
    k, band = 0.5, 0.02
    t = np.linspace(0, 60, 60001)  # long enough that y_end ~ 0
    assert settling_time(t, np.exp(-k * t), band) == pytest.approx(np.log(1 / band) / k, rel=1e-3)


def test_flat_line_under_oscillation_scores_penalty():
    f = Oscillation("x", period=6.28, tol=0.05, amplitude=1.0, sustained=True)
    np.testing.assert_array_equal(residuals(f, T, np.full_like(T, 3.0)), [PENALTY] * 4)
    assert oscillation_stats(T, 1 + 1e-15 * np.sin(T)) is None


def test_non_oscillating_trajectory_fails_but_is_graded():
    # Regression: a flat penalty for "no oscillation" stalled differential
    # evolution on the damped oscillator. Less damping must score strictly
    # better, all the way from overdamped to sustained.
    f = Oscillation("x", 2 * np.pi, 0.05, 1.0, sustained=True)
    losses = []
    for rate in [2.0, 1.0, 0.5, 0.2, 0.05, 0.0]:
        r = residuals(f, T, np.exp(-rate * T) * np.cos(T))
        losses.append(float(np.sum(r ** 2)))
        if rate >= 1.0:
            assert r[0] == PENALTY  # fewer than two peaks: never satisfied
    assert all(a > b for a, b in pairwise(losses)), losses
    assert losses[-1] < 1


@pytest.mark.parametrize("feature, good, bad", [
    (SteadyState("x", 5.0, 0.1),
     5 - 5 * np.exp(-T), 4.5 - 4.5 * np.exp(-T)),
    (SteadyState("x", 0.0, 0.1),
     np.exp(-T), 0.5 * np.sin(T)),            # ends near 0 but still moving
    (SettlingTime("x", np.log(50) / 0.5, 0.02, 0.5),
     np.exp(-0.5 * T), np.exp(-0.2 * T)),
    (Monotonic("x", "increasing"),
     1 - np.exp(-T), np.sin(T)),
    (Monotonic("x", "decreasing"),
     np.exp(-T), np.exp(-T) + 0.1 * np.sin(T)),
    (Bounds("x", 0.0, None),
     np.exp(-T), np.cos(T)),
    (Bounds("x", None, 1.0),
     np.sin(T), 2 * np.sin(T)),
    (Oscillation("x", 2 * np.pi, 0.05, 1.0, True),
     np.sin(T), np.sin(1.2 * T)),
    (Oscillation("x", 2 * np.pi, 0.05, 1.0, True),
     np.sin(T), 1.5 * np.sin(T)),
    (Oscillation("x", 2 * np.pi, 0.05, None, True),
     np.sin(T), np.exp(-0.05 * T) * np.sin(T)),  # damped but sustained required
])
def test_satisfying_trajectory_within_tolerance_violating_outside(feature, good, bad):
    assert np.all(np.abs(residuals(feature, T, good)) <= 1), residuals(feature, T, good)
    assert np.any(np.abs(residuals(feature, T, bad)) > 1), residuals(feature, T, bad)


def test_damping_allowed_when_not_sustained():
    f = Oscillation("x", 2 * np.pi, 0.05, None, sustained=False)
    assert np.all(np.abs(residuals(f, T, np.exp(-0.05 * T) * np.sin(T))) <= 1)


def test_residual_is_exactly_one_at_tolerance():
    f = SteadyState("x", 5.0, 0.1)
    r = residuals(f, T, np.full_like(T, 5.1))
    assert r[0] == pytest.approx(1.0)


def test_score_reports_per_feature():
    feats = [SteadyState("x", 1.0, 0.1), Bounds("y", 0.0, None)]
    out = score(feats, T, {"x": np.ones_like(T), "y": -np.ones_like(T)})
    assert [s["satisfied"] for s in out] == [True, False]


def test_noise_level_tail_after_damped_transient_is_not_sustained():
    # Regression: a damped oscillation that collapses onto a fixed point leaves
    # two final windows of integrator noise whose amplitude ratio is ~1. The
    # early cycles still supply peaks and a period, so only the sustain check
    # can reject it. (Here the last two windows are pure constant-amplitude
    # noise, so the plain window ratio reads decay == 0.)
    t = np.linspace(0, 200, 2001)
    y = (50 * (1 - np.exp(-t / 5)) + 40 * np.exp(-0.15 * t) * np.cos(2 * np.pi * t / 20)
         + 1e-6 * np.sin(2 * np.pi * t / 20))
    stats = oscillation_stats(t, y, window=20)
    assert stats["n_peaks"] >= 2 and stats["period"] == pytest.approx(20, rel=0.1)
    assert stats["decay"] > 0.99
    f = Oscillation("x", 20.0, 0.1, None, sustained=True)
    assert np.abs(residuals(f, t, y))[-1] > 1
