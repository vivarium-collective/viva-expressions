"""Regenerate lv.csv: Lotka-Volterra dx = x - 0.5xy, dy = -y + 0.25xy from (2, 1).

Run with ``uv run systems/examples/make_lv_csv.py``.
"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

t = np.linspace(0.0, 20.0, 401)
sol = solve_ivp(lambda _, z: [z[0] - 0.5 * z[0] * z[1], -z[1] + 0.25 * z[0] * z[1]],
                (0.0, 20.0), [2.0, 1.0], t_eval=t, method="DOP853", rtol=1e-11, atol=1e-12)
np.savetxt(Path(__file__).with_name("lv.csv"), np.column_stack([t, sol.y.T]),
           delimiter=",", header="t,prey,predator", comments="", fmt="%.10g")
