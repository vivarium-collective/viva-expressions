import numpy as np
from process_bigraph import Process
from scipy.integrate import solve_ivp

from viva_expressions.expressions import compile_rhs


class OdeProcess(Process):
    """Integrate ``d(state)/dt = rhs(state, inputs, params)`` over each interval.

    Inputs are held constant across an interval. Outputs are deltas on plain
    ``float`` ports, the process-bigraph convention for Processes.

    A terminal event stops integration once any ``|state|`` exceeds ``max_abs``
    and the update raises: LSODA otherwise stalls indefinitely at a finite-time
    singularity (e.g. ``dx/dt = x**2``) instead of reporting failure.

    ``time_var`` names a symbol in ``rhs`` that stands for the model time
    (e.g. a time-dependent SBML model). It is an input port, to be wired to
    the composite's ``global_time``, and is evaluated exactly as interval start
    plus solver time, never integrated, so a switch in time (a piecewise at
    t = 30) happens at exactly t = 30.
    """
    config_schema = {
        "rhs":        "map[string]",           # {"N": "r*N*(1 - N/K)"}
        "params":     "map[float]",            # named constants
        "state_vars": "list[string]",          # integrated variables, in order
        "input_vars": "list[string]",          # exogenous variables, held per interval
        "method":     {"_type": "string", "_default": "LSODA"},
        "rtol":       {"_type": "float", "_default": 1e-8},
        "atol":       {"_type": "float", "_default": 1e-10},
        "max_abs":    {"_type": "float", "_default": 1e12},  # divergence bound
        "time_var":   {"_type": "string", "_default": ""},   # model-time symbol (optional)
    }

    def initialize(self, config):
        cfg = self.config
        self.state_vars = list(cfg["state_vars"])
        self.input_vars = list(cfg.get("input_vars", []))
        params = cfg.get("params", {})
        self.time_var = cfg.get("time_var") or None
        self.rhs = compile_rhs(
            cfg["rhs"], self.state_vars, self.input_vars, list(params), time=self.time_var)
        self.p = np.array([params[k] for k in self.rhs.params], dtype=float)

    def inputs(self):
        ports = {v: "float" for v in self.state_vars + self.input_vars}
        if self.time_var:
            ports[self.time_var] = "float"
        return ports

    def outputs(self):
        return {v: "float" for v in self.state_vars}

    def update(self, state, interval):
        y0 = np.array([state[v] for v in self.state_vars], dtype=float)
        u = np.array([state[v] for v in self.input_vars], dtype=float)
        t0 = float(state[self.time_var]) if self.time_var else 0.0
        f, jac, p = self.rhs.f, self.rhs.jac, self.p
        max_abs = self.config["max_abs"]

        def diverged(t, y):
            return max_abs - np.max(np.abs(y))
        diverged.terminal = True

        sol = solve_ivp(
            lambda t, y: f(y, u, p, t0 + t), (0.0, interval), y0,
            method=self.config["method"],
            jac=None if jac is None else (lambda t, y: jac(y, u, p, t0 + t)),
            rtol=self.config["rtol"], atol=self.config["atol"], events=diverged)
        y_end = sol.y[:, -1]
        if sol.status == 1:
            raise RuntimeError(
                f"OdeProcess integration failed over interval {interval}: "
                f"|state| exceeded max_abs={max_abs:g} at t={sol.t[-1]:g}; "
                f"rhs={self.config['rhs']}, state={dict(zip(self.state_vars, y0))}")
        if not sol.success or not np.all(np.isfinite(y_end)):
            raise RuntimeError(
                f"OdeProcess integration failed over interval {interval}: "
                f"{sol.message}; rhs={self.config['rhs']}, "
                f"state={dict(zip(self.state_vars, y0))}")
        return {v: float(y_end[i] - y0[i]) for i, v in enumerate(self.state_vars)}
