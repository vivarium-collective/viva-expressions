"""Load and validate a system-synthesis spec (see ``systems/template.yml``).

Validation is strict and exhaustive: every problem is collected and raised
together in one ``SpecError``, each prefixed with the field path it concerns.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from viva_expressions.expressions import identifiers


class SpecError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid spec:\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass(frozen=True)
class Variable:
    name: str
    value: float
    bounds: tuple[float | None, float | None] = (None, None)


@dataclass(frozen=True)
class Time:
    t_end: float
    n_points: int

    @property
    def grid(self) -> np.ndarray:
        return np.linspace(0.0, self.t_end, self.n_points)


@dataclass(frozen=True)
class Data:
    path: Path
    t: np.ndarray
    columns: dict[str, np.ndarray]  # state variable -> observed series


@dataclass(frozen=True)
class SteadyState:
    var: str
    value: float
    tol: float  # absolute, in units of ``var``


@dataclass(frozen=True)
class SettlingTime:
    var: str
    value: float
    band: float  # relative band around the final value, e.g. 0.02
    tol: float   # absolute, in time units


@dataclass(frozen=True)
class Monotonic:
    var: str
    direction: str  # "increasing" | "decreasing"


@dataclass(frozen=True)
class Bounds:
    var: str
    min: float | None
    max: float | None


@dataclass(frozen=True)
class Oscillation:
    var: str
    period: float
    tol: float  # relative, applied to period and amplitude
    amplitude: float | None = None
    sustained: bool = True


Feature = SteadyState | SettlingTime | Monotonic | Bounds | Oscillation


@dataclass(frozen=True)
class Behavior:
    description: str = ""
    features: tuple[Feature, ...] = ()


@dataclass(frozen=True)
class MotifCandidate:
    motif: str
    bind: dict[str, str]
    params: dict[str, tuple[float, float]] = field(default_factory=dict)  # bound overrides
    prefix: str = ""  # prepended to the motif's parameter names


@dataclass(frozen=True)
class CustomCandidate:
    name: str
    rhs: dict[str, str]
    params: dict[str, tuple[float, float]]


Candidate = MotifCandidate | CustomCandidate


@dataclass(frozen=True)
class Fit:
    weights: dict[str, float] = field(
        default_factory=lambda: {"data": 1.0, "behavior": 1.0})
    seed: int = 0
    maxiter: int = 100


@dataclass(frozen=True)
class Spec:
    name: str
    time: Time
    inputs: tuple[Variable, ...]
    states: tuple[Variable, ...]
    behavior: Behavior
    candidates: tuple[Candidate, ...]
    fit: Fit
    data: Data | None = None
    constraints: tuple[str, ...] = ()

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.states)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.inputs)


# Required and optional keys per feature kind; value types are checked below.
FEATURE_KEYS = {
    "steady_state": ({"value", "tol"}, set()),
    "settling_time": ({"value", "band", "tol"}, set()),
    "monotonic": ({"direction"}, set()),
    "bounds": (set(), {"min", "max"}),
    "oscillation": ({"period", "tol"}, {"amplitude", "sustained"}),
}
FEATURE_TYPES = {"steady_state": SteadyState, "settling_time": SettlingTime,
                 "monotonic": Monotonic, "bounds": Bounds, "oscillation": Oscillation}
TOP_KEYS = {"name", "time", "variables", "constraints", "data", "behavior",
            "candidates", "fit"}


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SpecError([f"{path}: not valid YAML: {e}"]) from e
    return parse_spec(raw, base_dir=path.parent)


def parse_spec(raw: Any, base_dir: str | Path = ".") -> Spec:
    """Validate a spec mapping; relative ``data.path`` resolves against ``base_dir``."""
    v = _Validator()
    if not isinstance(raw, dict):
        raise SpecError(["spec: must be a mapping"])
    for k in sorted(set(raw) - TOP_KEYS):
        v.err(k, f"unknown key (allowed: {sorted(TOP_KEYS)})")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        v.err("name", "required non-empty string")

    time = v.time(raw.get("time"))
    variables = raw.get("variables")
    if not isinstance(variables, dict):
        v.err("variables", "required mapping with 'y' (states) and optional 'x' (inputs)")
        variables = {}
    inputs = v.variables(variables.get("x") or [], "variables.x")
    states = v.variables(variables.get("y"), "variables.y", required=True)
    states_set = {s.name for s in states}
    for dup in sorted(states_set & {i.name for i in inputs}):
        v.err("variables", f"{dup!r} declared as both input and state")

    constraints = raw.get("constraints") or []
    if not (isinstance(constraints, list) and all(isinstance(c, str) for c in constraints)):
        v.err("constraints", "must be a list of free-text strings")
        constraints = []

    data = v.data(raw.get("data"), Path(base_dir), states_set,
                  time.t_end if time is not None else None)
    behavior = v.behavior(raw.get("behavior"), states_set,
                          time.t_end if time is not None else None)
    candidates = v.candidates(raw.get("candidates"), states_set,
                              {i.name for i in inputs})
    fit = v.fit(raw.get("fit"))

    # Judge presence from the raw spec, so an invalid section is reported once
    # (above) rather than again as missing.
    raw_behavior = raw.get("behavior")
    has_data = raw.get("data") is not None
    has_features = isinstance(raw_behavior, dict) and bool(raw_behavior.get("features"))
    if not (has_data or has_features):
        v.err("spec", "needs 'data', 'behavior.features', or both")
    if not (has_data or raw.get("candidates")):
        v.err("candidates", "required when there is no 'data' to discover structure from")

    if v.errors:
        raise SpecError(v.errors)
    assert isinstance(name, str) and time is not None  # no errors => both validated
    return Spec(name=name, time=time, inputs=tuple(inputs), states=tuple(states),
                behavior=behavior, candidates=tuple(candidates), fit=fit,
                data=data, constraints=tuple(constraints))


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


class _Validator:
    def __init__(self):
        self.errors: list[str] = []

    def err(self, where: str, msg: str):
        self.errors.append(f"{where}: {msg}")

    def num(self, d: dict, key: str, where: str, *, positive=False, optional=False):
        if key not in d or d[key] is None:
            if not optional:
                self.err(f"{where}.{key}", "required number")
            return None
        x = d[key]
        if not _is_num(x):
            self.err(f"{where}.{key}", f"must be a finite number, got {x!r}")
            return None
        if positive and x <= 0:
            self.err(f"{where}.{key}", f"must be > 0, got {x!r}")
            return None
        return float(x)

    def keys(self, d: dict, where: str, required: set, optional: set):
        for k in sorted(required - set(d)):
            self.err(f"{where}.{k}", "required")
        for k in sorted(set(d) - required - optional):
            self.err(f"{where}.{k}", f"unknown key (allowed: {sorted(required | optional)})")

    def time(self, raw) -> Time | None:
        if not isinstance(raw, dict):
            self.err("time", "required mapping {t_end, n_points}")
            return None
        self.keys(raw, "time", {"t_end", "n_points"}, set())
        t_end = self.num(raw, "t_end", "time", positive=True)
        n = raw.get("n_points")
        if not (isinstance(n, int) and not isinstance(n, bool) and n >= 2):
            self.err("time.n_points", f"must be an integer >= 2, got {n!r}")
            return None
        return Time(t_end, n) if t_end is not None else None

    def variables(self, raw, where: str, required=False) -> list[Variable]:
        if not isinstance(raw, list) or (required and not raw):
            self.err(where, "required non-empty list" if required else "must be a list")
            return []
        out, seen = [], set()
        for i, item in enumerate(raw):
            w = f"{where}[{i}]"
            if not isinstance(item, dict):
                self.err(w, "must be a mapping {name, value[, bounds]}")
                continue
            self.keys(item, w, {"name", "value"}, {"bounds", "type"})
            name = item.get("name")
            if not (isinstance(name, str) and name.isidentifier()):
                self.err(f"{w}.name", f"must be an identifier, got {name!r}")
                continue
            if name in seen:
                self.err(f"{w}.name", f"duplicate {name!r}")
            seen.add(name)
            if item.get("type", "float") != "float":
                self.err(f"{w}.type", "only 'float' is supported")
            value = self.num(item, "value", w)
            bounds = self.bounds_pair(item.get("bounds"), f"{w}.bounds", allow_open=True)
            if value is None or bounds is None:
                continue
            lo, hi = bounds
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                self.err(f"{w}.value", f"{value} outside bounds {list(bounds)}")
            out.append(Variable(name, value, bounds))
        return out

    def bounds_pair(self, raw, where: str, allow_open=False):
        if raw is None and allow_open:
            return (None, None)
        if not (isinstance(raw, list) and len(raw) == 2):
            self.err(where, f"must be [lo, hi], got {raw!r}")
            return None
        lo, hi = raw
        for x in (lo, hi):
            if not (_is_num(x) or (allow_open and x is None)):
                self.err(where, f"bounds must be finite numbers{' or null' if allow_open else ''}, got {raw!r}")
                return None
        if lo is not None and hi is not None and lo >= hi:
            self.err(where, f"lo must be < hi, got {raw!r}")
            return None
        return (None if lo is None else float(lo), None if hi is None else float(hi))

    def data(self, raw, base_dir: Path, states: set[str], t_end: float | None) -> Data | None:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            self.err("data", "must be a mapping {path, time_column, columns}")
            return None
        self.keys(raw, "data", {"path", "time_column", "columns"}, set())
        cols = raw.get("columns")
        if not (isinstance(cols, dict) and cols):
            self.err("data.columns", "required mapping {state_variable: csv_column}")
            return None
        for var in sorted(set(cols) - states):
            self.err(f"data.columns.{var}", "not a declared state variable (variables.y)")
        if not isinstance(raw.get("path"), str) or not isinstance(raw.get("time_column"), str):
            return None
        path = Path(raw["path"])
        path = path if path.is_absolute() else base_dir / path
        if not path.is_file():
            self.err("data.path", f"file not found: {path}")
            return None
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
        header = table.dtype.names or ()
        needed = [raw["time_column"], *cols.values()]
        missing = [c for c in needed if c not in header]
        if missing:
            self.err("data", f"columns {missing} not in {path.name} header {list(header)}")
            return None
        t = np.atleast_1d(table[raw["time_column"]])
        series = {var: np.atleast_1d(table[c]) for var, c in cols.items() if var in states}
        if len(t) < 3:
            self.err("data.path", f"need at least 3 rows, got {len(t)}")
        elif not np.all(np.diff(t) > 0):
            self.err("data.time_column", "time must be strictly increasing")
        elif t[0] < 0 or (t_end is not None and t[-1] > t_end):
            self.err("data.time_column",
                     f"times must lie in [0, time.t_end={t_end}], got [{t[0]:g}, {t[-1]:g}]")
        for var, y in series.items():
            if not np.all(np.isfinite(y)):
                self.err(f"data.columns.{var}", "contains non-finite values")
        return Data(path, t, series)

    def behavior(self, raw, states: set[str], t_end: float | None) -> Behavior:
        if raw is None:
            return Behavior()
        if isinstance(raw, str):  # free-text only, features still to be derived
            return Behavior(description=raw)
        if not isinstance(raw, dict):
            self.err("behavior", "must be a string or a mapping {description, features}")
            return Behavior()
        self.keys(raw, "behavior", set(), {"description", "features"})
        feats = raw.get("features") or []
        if not isinstance(feats, list):
            self.err("behavior.features", "must be a list")
            feats = []
        out = [f for i, item in enumerate(feats)
               if (f := self.feature(item, f"behavior.features[{i}]", states, t_end)) is not None]
        return Behavior(str(raw.get("description") or ""), tuple(out))

    def feature(self, raw, where: str, states: set[str], t_end: float | None) -> Feature | None:
        if not isinstance(raw, dict) or raw.get("kind") not in FEATURE_KEYS:
            self.err(f"{where}.kind", f"must be one of {sorted(FEATURE_KEYS)}")
            return None
        kind = raw["kind"]
        required, optional = FEATURE_KEYS[kind]
        n_before = len(self.errors)
        self.keys(raw, where, required | {"kind", "var"}, optional)
        var = raw.get("var")
        if var not in states:
            self.err(f"{where}.var", f"must name a state variable {sorted(states)}, got {var!r}")
        def num(key, **kw):
            return self.num(raw, key, where, **kw)

        if kind == "steady_state":
            fields = {"value": num("value"), "tol": num("tol", positive=True)}
        elif kind == "settling_time":
            fields = {"value": num("value", positive=True),
                      "band": num("band", positive=True), "tol": num("tol", positive=True)}
            if fields["band"] is not None and fields["band"] >= 1:
                self.err(f"{where}.band", "must be a fraction in (0, 1)")
        elif kind == "monotonic":
            fields = {"direction": raw.get("direction")}
            if fields["direction"] not in ("increasing", "decreasing"):
                self.err(f"{where}.direction", "must be 'increasing' or 'decreasing'")
        elif kind == "bounds":
            fields = {"min": num("min", optional=True), "max": num("max", optional=True)}
            lo, hi = fields["min"], fields["max"]
            if lo is None and hi is None:
                self.err(where, "needs 'min', 'max', or both")
            elif lo is not None and hi is not None and lo >= hi:
                self.err(where, f"min must be < max, got {lo} >= {hi}")
        else:
            fields = {"period": num("period", positive=True), "tol": num("tol", positive=True),
                      "amplitude": num("amplitude", positive=True, optional=True),
                      "sustained": raw.get("sustained", True)}
            if fields["tol"] is not None and fields["tol"] >= 1:
                self.err(f"{where}.tol", "oscillation tol is relative; must be in (0, 1)")
            if not isinstance(fields["sustained"], bool):
                self.err(f"{where}.sustained", "must be true or false")
            if fields["period"] is not None and t_end is not None and t_end < 2 * fields["period"]:
                self.err(f"{where}.period",
                         f"time.t_end={t_end:g} must cover at least two periods")
        if len(self.errors) > n_before:
            return None
        return FEATURE_TYPES[kind](var=var, **fields)

    def candidates(self, raw, states: set[str], inputs: set[str]) -> list[Candidate]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            self.err("candidates", "must be a list")
            return []
        out, names = [], set()
        for i, item in enumerate(raw):
            w = f"candidates[{i}]"
            if not isinstance(item, dict) or ("motif" in item) == ("rhs" in item):
                self.err(w, "must have exactly one of {motif, bind} or {name, rhs, params}")
                continue
            if "motif" in item:
                self.keys(item, w, {"motif", "bind"}, {"params", "prefix"})
                prefix = item.get("prefix", "")
                if not (isinstance(prefix, str) and (prefix == "" or prefix.isidentifier())):
                    self.err(f"{w}.prefix", f"must be an identifier prefix, got {prefix!r}")
                    continue
                bind = item.get("bind")
                if not (isinstance(item.get("motif"), str) and isinstance(bind, dict)
                        and all(isinstance(k, str) and isinstance(b, str) for k, b in bind.items())):
                    self.err(w, "motif must be a string and bind a {role: variable} mapping")
                    continue
                for role, var in bind.items():
                    if var not in states | inputs:
                        self.err(f"{w}.bind.{role}", f"{var!r} is not a declared variable")
                overrides = item.get("params") or {}
                if not isinstance(overrides, dict):
                    self.err(f"{w}.params", "must be a mapping {param: [lo, hi]}")
                    continue
                bounds = {p: self.bounds_pair(b, f"{w}.params.{p}") for p, b in overrides.items()}
                if all(b is not None for b in bounds.values()):
                    out.append(MotifCandidate(item["motif"], dict(bind), bounds, prefix))
                continue
            self.keys(item, w, {"name", "rhs", "params"}, set())
            name, rhs, params = item.get("name"), item.get("rhs"), item.get("params")
            if not (isinstance(name, str) and name):
                self.err(f"{w}.name", "required non-empty string")
            elif name in names:
                self.err(f"{w}.name", f"duplicate candidate name {name!r}")
            names.add(name)
            if not isinstance(rhs, dict) or set(rhs) != states:
                self.err(f"{w}.rhs", f"must map exactly the state variables {sorted(states)}")
                continue
            if not isinstance(params, dict):
                self.err(f"{w}.params", "required mapping {param: [lo, hi]}")
                continue
            bounds = {p: self.bounds_pair(b, f"{w}.params.{p}") for p, b in params.items()}
            for p in sorted(set(params) & (states | inputs)):
                self.err(f"{w}.params.{p}", "parameter name clashes with a variable")
            allowed = states | inputs | set(params) | {"pi"}
            for var, expr in rhs.items():
                try:
                    unknown = identifiers(str(expr)) - allowed
                except ValueError as e:
                    self.err(f"{w}.rhs.{var}", str(e))
                    continue
                if unknown:
                    self.err(f"{w}.rhs.{var}", f"undeclared symbol(s) {sorted(unknown)}")
            if all(b is not None for b in bounds.values()):
                out.append(CustomCandidate(name, {k: str(e) for k, e in rhs.items()}, bounds))
        return out

    def fit(self, raw) -> Fit:
        if raw is None:
            return Fit()
        if not isinstance(raw, dict):
            self.err("fit", "must be a mapping {weights, seed, maxiter}")
            return Fit()
        self.keys(raw, "fit", set(), {"weights", "seed", "maxiter"})
        weights = dict(Fit().weights)
        w = raw.get("weights") or {}
        if not isinstance(w, dict) or set(w) - set(weights):
            self.err("fit.weights", f"must be a mapping with keys from {sorted(weights)}")
        else:
            for k in w:
                x = self.num(w, k, "fit.weights")
                if x is not None and x < 0:
                    self.err(f"fit.weights.{k}", "must be >= 0")
                elif x is not None:
                    weights[k] = x
        ints = {}
        for k, lo in (("seed", 0), ("maxiter", 1)):
            x = raw.get(k, getattr(Fit(), k))
            if not (isinstance(x, int) and not isinstance(x, bool) and x >= lo):
                self.err(f"fit.{k}", f"must be an integer >= {lo}, got {x!r}")
                x = getattr(Fit(), k)
            ints[k] = x
        return Fit(weights, **ints)
