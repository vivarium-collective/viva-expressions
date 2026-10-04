"""Library of interpretable ODE motifs, and resolution of candidates into models.

A motif's ``rhs`` is written over role names. ``state_roles`` each get an
equation; ``input_roles`` are only referenced, and may bind to either an input
or a state that another equation governs. Binding renames roles to the spec's
variables; parameters keep their names unless a ``prefix`` is given.
"""
import re
from dataclasses import dataclass, field

from viva_expressions.expressions import compile_rhs
from viva_expressions.synthesize.spec import CustomCandidate, MotifCandidate, Spec


@dataclass(frozen=True)
class Model:
    """A fully named candidate structure: ``d(state)/dt = rhs`` with bounded params."""
    name: str
    rhs: dict[str, str]
    params: dict[str, tuple[float, float]]
    initial: dict[str, float] = field(default_factory=dict)  # optional starting guess
    origin: str = "custom"  # "custom" | "motif:<name>" | "sindy"


@dataclass(frozen=True)
class Motif:
    name: str
    description: str
    state_roles: tuple[str, ...]
    rhs: dict[str, str]
    params: dict[str, tuple[float, float]]
    input_roles: tuple[str, ...] = ()

    @property
    def roles(self) -> tuple[str, ...]:
        return self.state_roles + self.input_roles


MOTIFS = {m.name: m for m in [
    Motif("exponential_decay", "first-order loss: x' = -k x",
          ("x",), {"x": "-k*x"}, {"k": (0.0, 10.0)}),
    Motif("exponential_growth", "unbounded growth: x' = r x",
          ("x",), {"x": "r*x"}, {"r": (0.0, 10.0)}),
    Motif("logistic", "growth saturating at capacity K: x' = r x (1 - x/K)",
          ("x",), {"x": "r*x*(1 - x/K)"}, {"r": (0.0, 10.0), "K": (1e-6, 1e3)}),
    Motif("hill_activation", "x produced when s is high, cooperatively; first-order loss",
          ("x",), {"x": "beta*s**n/(Kd**n + s**n) - gamma*x"},
          {"beta": (0.0, 100.0), "Kd": (1e-6, 1e3), "n": (1.0, 4.0), "gamma": (0.0, 10.0)},
          input_roles=("s",)),
    Motif("hill_repression", "x produced when s is low, cooperatively; first-order loss",
          ("x",), {"x": "beta/(1 + (s/Kd)**n) - gamma*x"},
          {"beta": (0.0, 100.0), "Kd": (1e-6, 1e3), "n": (1.0, 4.0), "gamma": (0.0, 10.0)},
          input_roles=("s",)),
    Motif("michaelis_menten", "saturable consumption of substrate s",
          ("s",), {"s": "-Vmax*s/(Km + s)"}, {"Vmax": (0.0, 100.0), "Km": (1e-6, 1e3)}),
    Motif("harmonic_oscillator", "undamped oscillation at angular frequency omega",
          ("x", "v"), {"x": "v", "v": "-omega**2*x"}, {"omega": (0.0, 10.0)}),
    Motif("damped_oscillator", "oscillation at omega, damped by ratio zeta",
          ("x", "v"), {"x": "v", "v": "-omega**2*x - 2*zeta*omega*v"},
          {"omega": (0.0, 10.0), "zeta": (0.0, 2.0)}),
    Motif("lotka_volterra", "predator-prey cycles",
          ("prey", "predator"),
          {"prey": "a*prey - b*prey*predator", "predator": "-c*predator + d*prey*predator"},
          {"a": (0.0, 5.0), "b": (0.0, 5.0), "c": (0.0, 5.0), "d": (0.0, 5.0)}),
    Motif("goodwin", "three-stage negative-feedback (repressor) oscillator; needs n > 8",
          ("m", "p", "r"),
          {"m": "alpha/(1 + (r/Ki)**n) - k*m", "p": "k*m - k*p", "r": "k*p - k*r"},
          {"alpha": (0.0, 100.0), "Ki": (1e-3, 1e2), "n": (1.0, 20.0), "k": (0.0, 5.0)}),
]}


def instantiate(motif: Motif, bind: dict[str, str], prefix: str = "",
                overrides: dict[str, tuple[float, float]] | None = None) -> Model:
    """Rename ``motif``'s roles to bound variables (and params by ``prefix``)."""
    if set(bind) != set(motif.roles):
        raise ValueError(f"motif {motif.name!r} needs bind for roles {list(motif.roles)}, "
                         f"got {sorted(bind)}")
    overrides = overrides or {}
    unknown = set(overrides) - set(motif.params)
    if unknown:
        raise ValueError(f"motif {motif.name!r} has no params {sorted(unknown)} "
                         f"(has {list(motif.params)})")
    rename = {**bind, **{p: prefix + p for p in motif.params}}
    clash = set(bind.values()) & {prefix + p for p in motif.params}
    if clash:
        raise ValueError(f"variables {sorted(clash)} clash with motif {motif.name!r} "
                         f"parameter names; give the candidate a `prefix`")
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, rename)) + r")\b")
    rhs = {bind[role]: pattern.sub(lambda m: rename[m.group(1)], expr)
           for role, expr in motif.rhs.items()}
    params = {prefix + p: overrides.get(p, b) for p, b in motif.params.items()}
    return Model(f"{motif.name}({', '.join(f'{r}={v}' for r, v in bind.items())})",
                 rhs, params, origin=f"motif:{motif.name}")


def resolve(candidate: MotifCandidate | CustomCandidate, spec: Spec) -> Model:
    """Turn a spec candidate into a Model whose equations cover exactly the spec's states."""
    if isinstance(candidate, MotifCandidate):
        if candidate.motif not in MOTIFS:
            raise ValueError(f"unknown motif {candidate.motif!r}; known: {sorted(MOTIFS)}")
        model = instantiate(MOTIFS[candidate.motif], candidate.bind,
                            prefix=candidate.prefix, overrides=candidate.params)
    else:
        model = Model(candidate.name, dict(candidate.rhs), dict(candidate.params))
    states = set(spec.state_names)
    if set(model.rhs) != states:
        raise ValueError(f"candidate {model.name!r} defines equations for {sorted(model.rhs)} "
                         f"but the spec's states are {sorted(states)}")
    # compiling (as fitting will) catches undeclared symbols and any name
    # shared between states, inputs and parameters
    compile_rhs(model.rhs, spec.state_names, spec.input_names, list(model.params))
    return model


def describe() -> str:
    """Human-readable motif listing (for ``--list-motifs``)."""
    lines = []
    for m in MOTIFS.values():
        roles = ", ".join(m.state_roles) + (
            f" | inputs: {', '.join(m.input_roles)}" if m.input_roles else "")
        lines.append(f"{m.name}  [{roles}]  {m.description}")
        for v, e in m.rhs.items():
            lines.append(f"    d{v}/dt = {e}")
        lines.append("    params: " + ", ".join(f"{p} in [{lo:g}, {hi:g}]"
                                                for p, (lo, hi) in m.params.items()))
    return "\n".join(lines)
