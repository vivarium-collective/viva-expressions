"""Import an SBML model as OdeProcess inputs (``rhs``, ``params``, ``initial``).

The translation follows the SBML Level 3 Version 2 Core semantics (which
Level 2 models share for everything handled here):

* A kinetic law is the reaction's extent per unit time. A species changed by
  reactions accumulates ``d(amount)/dt = sum(stoichiometry * rate * conversion
  factor)``.
* A species identifier in math stands for its concentration
  (``amount / compartment size``) unless ``hasOnlySubstanceUnits`` is true,
  when it stands for its amount. Each species is integrated in the units its
  symbol has, so concentration species get ``d[S]/dt = d(amount)/dt / V``.
* Boundary-condition or constant species are not changed by reactions.
* Rate rules give a variable's derivative directly, in its symbol's units.
* Assignment rules are substituted into every derivative (resolved in
  dependency order). They are also kept as named ``assignments``, so derived
  quantities a model declares stay observable.
* The ``time`` csymbol becomes ``OdeModel.time_var``, a symbol OdeProcess binds
  to the exact model clock (``global_time`` plus solver time). It is never
  integrated: an integrated clock drifts by ~1e-13, which puts a piecewise
  switch that falls on a sample point on the wrong branch.

libsbml's own converters first expand function definitions and initial
assignments and promote local parameters to global ones. Constructs with no
faithful mapping (events, delays, algebraic rules, fast reactions, variable
stoichiometry, species in a compartment whose size changes) raise
``UnsupportedSBML``; nothing is dropped silently.
"""
import cmath
import keyword
import math
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import libsbml
import numpy as np
import sympy as sp

from viva_expressions.expressions import parse
from viva_expressions.sbml.math import RESERVED, UnsupportedSBML, from_ast, to_text

CONVERTERS = ("promoteLocalParameters", "expandFunctionDefinitions",
              "expandInitialAssignments")


@dataclass(frozen=True)
class OdeModel:
    """An SBML model as OdeProcess inputs, plus what's needed to compare it.

    ``kinds`` says what each state, assignment or parameter is in SBML terms
    (``concentration``, ``amount``, ``parameter`` or ``compartment``), and
    ``sbml_ids`` maps each of those names back to its SBML identifier. Names
    equal SBML identifiers except where one isn't a valid expression identifier
    (a Python keyword, a reserved function name, or a clash with ``time_var``).
    ``time_var`` names the model-time symbol for time-dependent models, else None.
    """
    model_id: str
    rhs: dict[str, str]
    params: dict[str, float]
    initial: dict[str, float]
    assignments: dict[str, str] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    sbml_ids: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    time_var: str | None = None


def _read(source: str | Path) -> libsbml.SBMLDocument:
    text = str(source)
    if text.lstrip().startswith("<"):
        doc = libsbml.readSBMLFromString(text)
    else:
        if not Path(text).is_file():
            raise FileNotFoundError(f"SBML file not found: {text}")
        doc = libsbml.readSBMLFromFile(text)
    if doc.getModel() is None:
        raise ValueError(f"no SBML model in {text[:80]!r}: {_errors(doc) or 'empty'}")
    return doc


def _errors(doc: libsbml.SBMLDocument) -> str:
    log = doc.getErrorLog()
    return "; ".join(
        f"line {e.getLine()}: {e.getShortMessage()} {e.getMessage().strip()}"
        for e in (log.getError(i) for i in range(log.getNumErrors()))
        if e.getSeverity() >= libsbml.LIBSBML_SEV_ERROR)


def _validate(doc: libsbml.SBMLDocument) -> None:
    """Raise on any SBML error; unit consistency isn't checked (no units here)."""
    doc.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
    doc.checkConsistency()
    if doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) + doc.getNumErrors(libsbml.LIBSBML_SEV_FATAL):
        raise ValueError(f"invalid SBML: {_errors(doc)}")


def _convert(doc: libsbml.SBMLDocument) -> None:
    for option in CONVERTERS:
        props = libsbml.ConversionProperties()
        props.addOption(option, True)
        if doc.convert(props) != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise UnsupportedSBML(f"libsbml converter {option!r} failed: {_errors(doc)}")


def _reject(m: libsbml.Model) -> None:
    problems = []
    if m.getNumEvents():
        problems.append(f"{m.getNumEvents()} event(s)")
    if m.getNumInitialAssignments():
        ids = [a.getSymbol() for a in m.getListOfInitialAssignments()]
        problems.append(f"initial assignments libsbml could not evaluate: {ids}")
    if any(r.getTypeCode() == libsbml.SBML_ALGEBRAIC_RULE for r in m.getListOfRules()):
        problems.append("algebraic rule(s) (a DAE)")
    for r in m.getListOfReactions():
        if r.isSetFast() and r.getFast():
            problems.append(f"fast reaction {r.getId()!r}")
        if not r.isSetKineticLaw() or not r.getKineticLaw().isSetMath():
            problems.append(f"reaction {r.getId()!r} has no kinetic law")
        for ref in (*r.getListOfReactants(), *r.getListOfProducts()):
            if ref.isSetStoichiometryMath() or (ref.isSetId() and m.getRule(ref.getId())):
                problems.append(f"variable stoichiometry in reaction {r.getId()!r}")
            elif m.getLevel() >= 3 and not ref.isSetStoichiometry():
                problems.append(f"unset stoichiometry in reaction {r.getId()!r}")
    ruled = {r.getVariable() for r in m.getListOfRules() if r.getTypeCode() != libsbml.SBML_ALGEBRAIC_RULE}
    for s in m.getListOfSpecies():
        if s.getCompartment() in ruled:
            problems.append(f"species {s.getId()!r} lives in compartment "
                            f"{s.getCompartment()!r}, whose size changes")
    if problems:
        raise UnsupportedSBML(
            f"model {m.getId()!r} has no faithful OdeProcess mapping: " + "; ".join(problems))


# The namespace lambdify compiles into (modules="numpy"): an argument with one
# of these names would shadow a function the printed code calls.
NUMPY_NAMES = frozenset(dir(np))


def _reserved(name: str) -> bool:
    return keyword.iskeyword(name) or name in RESERVED or name in NUMPY_NAMES


def _namer(m: libsbml.Model, uses_time: bool):
    """SBML id -> expression name: identity unless the id can't be one.

    An id is renamed (suffixed ``_``) when it is a Python keyword, not a Python
    identifier, a name the printed expressions use for a function or constant
    (``RESERVED``), or a name in the numpy namespace that compiled expressions
    run in (``NUMPY_NAMES``: e.g. a parameter ``select`` would shadow the
    function a Piecewise compiles to). The clock is renamed if an id is ``time``.
    """
    ids = {e.getId() for lst in (m.getListOfSpecies(), m.getListOfParameters(),
                                 m.getListOfCompartments(), m.getListOfReactions())
           for e in lst}
    taken = set(ids)
    time_name = "time"
    if uses_time:
        while time_name in taken:
            time_name += "_"
        taken.add(time_name)
    names = {}
    for i in sorted(ids):
        name = i
        if _reserved(i) or not i.isidentifier():
            while name in taken or _reserved(name):
                name += "_"
            taken.add(name)
        names[i] = name
    return names, time_name


def _uses_time(m: libsbml.Model) -> bool:
    def walk(n: libsbml.ASTNode) -> bool:
        return n.getType() == libsbml.AST_NAME_TIME or any(
            walk(n.getChild(i)) for i in range(n.getNumChildren()))
    maths = [r.getMath() for r in m.getListOfRules()] + [
        r.getKineticLaw().getMath() for r in m.getListOfReactions()]
    return any(walk(x) for x in maths if x is not None)


def _resolve(exprs: dict[sp.Symbol, sp.Basic]) -> dict[sp.Symbol, sp.Basic]:
    """Fully substitute definitions that reference each other, in dependency order."""
    graph = {k: v.free_symbols & set(exprs) for k, v in exprs.items()}
    try:
        order = list(TopologicalSorter(graph).static_order())
    except CycleError as e:
        raise ValueError(f"cyclic assignment rules: {e.args[1]}") from e
    full: dict[sp.Symbol, sp.Basic] = {}
    for k in order:
        full[k] = exprs[k].xreplace(full)
    return full


def _number(x: float) -> sp.Expr:
    """An exact sympy number: an Integer when integral, else the same double."""
    return sp.Integer(int(x)) if float(x).is_integer() else sp.Float(x)


def _species_value(s: libsbml.Species, size: float) -> float:
    """A species' initial value in the units its symbol has."""
    if s.isSetInitialAmount():
        amount = s.getInitialAmount()
        return amount if s.getHasOnlySubstanceUnits() else amount / size
    if s.isSetInitialConcentration():
        conc = s.getInitialConcentration()
        return conc * size if s.getHasOnlySubstanceUnits() else conc
    raise UnsupportedSBML(f"species {s.getId()!r} has no initial value")


def read_sbml(source: str | Path) -> OdeModel:
    """Translate an SBML file (or SBML text) into an ``OdeModel``."""
    doc = _read(source)
    _validate(doc)
    _convert(doc)
    m = doc.getModel()
    _reject(m)

    uses_time = _uses_time(m)
    names, time_name = _namer(m, uses_time)
    sym = {i: sp.Symbol(n) for i, n in names.items()}
    time_sym = sp.Symbol(time_name)

    def symbol(i: str) -> sp.Symbol:
        if i not in sym:
            raise UnsupportedSBML(f"math references {i!r}, which is not a species, "
                                  "parameter, compartment or reaction")
        return sym[i]

    def math(node: libsbml.ASTNode) -> sp.Basic:
        return from_ast(node, symbol, lambda: time_sym)

    comp_size = {c.getId(): c.getSize() for c in m.getListOfCompartments()}
    assign, rate = {}, {}
    for r in m.getListOfRules():
        target = sym[r.getVariable()]
        (assign if r.isAssignment() else rate)[target] = math(r.getMath())
    # L3: a reaction id in math stands for its rate; resolve it like an assignment
    rates = {sym[r.getId()]: math(r.getKineticLaw().getMath()) for r in m.getListOfReactions()}
    definitions = _resolve({**rates, **assign})

    model_cf = m.getConversionFactor() if m.isSetConversionFactor() else ""
    flux = {s.getId(): sp.Integer(0) for s in m.getListOfSpecies()}
    for r in m.getListOfReactions():
        v = sym[r.getId()]
        for sign, refs in ((-1, r.getListOfReactants()), (1, r.getListOfProducts())):
            for ref in refs:
                s = m.getSpecies(ref.getSpecies())
                if s.getBoundaryCondition() or s.getConstant():
                    continue
                cf = s.getConversionFactor() if s.isSetConversionFactor() else model_cf
                term = sign * _number(ref.getStoichiometry()) * v
                flux[s.getId()] += term * (sym[cf] if cf else 1)

    rhs, initial, kinds, notes = {}, {}, {}, []
    params = {}
    for s in m.getListOfSpecies():
        sid, x = s.getId(), sym[s.getId()]
        kind = "amount" if s.getHasOnlySubstanceUnits() else "concentration"
        if x in assign:     # an assignment rule defines it: no initial value needed
            kinds[x.name] = kind
            continue
        value = _species_value(s, comp_size[s.getCompartment()])
        if x in rate:
            rhs[x], initial[x.name], kinds[x.name] = rate[x], value, kind
        elif flux[sid] != 0:
            scale = 1 if s.getHasOnlySubstanceUnits() else 1 / sym[s.getCompartment()]
            rhs[x], initial[x.name], kinds[x.name] = flux[sid] * scale, value, kind
        else:
            params[x.name], kinds[x.name] = value, kind
    for kind, elements in (("compartment", m.getListOfCompartments()),
                           ("parameter", m.getListOfParameters())):
        for e in elements:
            x = sym[e.getId()]
            value = e.getSize() if kind == "compartment" else e.getValue()
            if x in assign:
                kinds[x.name] = kind
                continue
            if not (e.isSetSize() if kind == "compartment" else e.isSetValue()):
                raise UnsupportedSBML(f"{kind} {e.getId()!r} has no value")
            if x in rate:
                rhs[x], initial[x.name], kinds[x.name] = rate[x], value, kind
            else:
                params[x.name], kinds[x.name] = value, kind
    if uses_time:
        notes.append(f"time-dependent model: {time_name!r} is the exact model clock "
                     "(global_time), starting at 0")
    if m.getNumConstraints():
        notes.append(f"{m.getNumConstraints()} constraint(s) ignored: they assert "
                     "conditions during simulation but do not change the dynamics")

    allowed = [*initial, *params, *([time_name] if uses_time else [])]
    rhs_text = {x.name: _emit(e.xreplace(definitions), allowed, x.name)
                for x, e in rhs.items()}
    assign_names = [x.name for x in assign]
    reaction_rates = {k: definitions[k] for k in rates}
    assignments = {x.name: _emit(e.xreplace(reaction_rates), allowed + assign_names, x.name)
                   for x, e in assign.items()}
    reverse = {n: i for i, n in names.items()}
    sbml_ids = {n: reverse.get(n, n) for n in [*rhs_text, *assignments, *params]}
    return OdeModel(model_id=m.getId(), rhs=rhs_text, params=params, initial=initial,
                    assignments=assignments, kinds=kinds, sbml_ids=sbml_ids,
                    notes=tuple(notes), time_var=time_name if uses_time else None)


# Identity test on seeded sample points: a wide box (positive, negative and
# log-spaced values) plus every numeric constant of the expression and its
# neighbours, so Piecewise thresholds (t > 30) are straddled. Agreement is to
# double-precision rounding. Seeded, so imports are deterministic.
_RNG_SEED = 0
_N_POINTS = 24
_RTOL = 1e-12


def _probe_rows(expr: sp.Basic, n_symbols: int) -> np.ndarray:
    rng = np.random.default_rng(_RNG_SEED)
    constants = sorted({float(c) for c in expr.atoms(sp.Number)
                        if math.isfinite(float(c))})[:64]
    special = np.array([v for c in constants
                        for v in (c, c * (1 + 1e-3), c * (1 - 1e-3), c + 1e-3, c - 1e-3, -c)]
                       or [1.0])
    box = np.concatenate([rng.uniform(0.5, 2.0, 64), rng.uniform(-50.0, 50.0, 64),
                          10.0 ** rng.uniform(-3, 3, 64), special])
    return rng.choice(box, size=(_N_POINTS, n_symbols))


def _names(expr: sp.Basic) -> set:
    """The symbols and named constants (pi, E) an expression refers to, tagged
    by kind: a symbol named ``pi`` and the constant pi print alike (and shadow
    each other in lambdified code), so only the tag tells them apart."""
    return ({("symbol", s.name) for s in expr.free_symbols}
            | {("constant", str(c)) for c in expr.atoms(sp.NumberSymbol)})


def _same_function(a: sp.Basic, b: sp.Basic) -> bool:
    """Whether ``a`` and ``b`` agree at the sample points (at least one of them
    must be evaluable; an expression undefined at every point can't be checked)."""
    symbols = sorted(a.free_symbols | b.free_symbols, key=lambda x: x.name)
    fa, fb = (sp.lambdify(symbols, e, modules="numpy") for e in (a, b))
    checked = 0
    with np.errstate(all="ignore"):
        for row in _probe_rows(a, len(symbols)):
            va, vb = complex(fa(*row)), complex(fb(*row))
            if cmath.isnan(va) and cmath.isnan(vb):    # outside both domains
                continue
            if not (math.isclose(va.real, vb.real, rel_tol=_RTOL, abs_tol=_RTOL)
                    and math.isclose(va.imag, vb.imag, rel_tol=_RTOL, abs_tol=_RTOL)):
                return False
            checked += 1
    if not checked:
        raise UnsupportedSBML(f"cannot verify {a}: undefined at every sample point")
    return True


def _emit(expr: sp.Basic, allowed: list[str], name: str) -> str:
    """Print ``expr`` in the OdeProcess grammar, checked to mean exactly ``expr``.

    OdeProcess parses the text with sympy, which re-applies its own algebraic
    normalization (e.g. it distributes a leading minus sign over a sum), so the
    reparsed expression need not be structurally identical to ``expr``. Instead:
    the reparsed expression must refer to the same names and named constants (a
    printed name read back as something else, such as a parameter ``pi`` read as
    the constant, is caught) and agree with ``expr`` at seeded random points.
    The returned text is sympy's own canonical form, a print/parse fixed point,
    so it is exactly what OdeProcess will compile. Numbers outside double range,
    before or after reparsing, are rejected: sympy folds e.g.
    ``exp(2412.0 - 80*t)`` into ``1e1047*exp(-80*t)``, which evaluates to
    ``inf`` and silently turns trajectories into ``nan``.
    """
    try:
        parsed = parse(to_text(expr), allowed)
    except (ValueError, TypeError, SyntaxError) as e:
        raise UnsupportedSBML(f"expression for {name!r} does not reparse: {e}") from e
    text = to_text(parsed)
    for e in (expr, parsed):
        for number in e.atoms(sp.Number):
            try:
                finite = math.isfinite(float(number))
            except OverflowError:
                finite = False
            if not finite:
                raise UnsupportedSBML(f"expression for {name!r} has a number outside "
                                      f"double range ({number}): {text}")
    if _names(parsed) != _names(expr):
        raise UnsupportedSBML(f"expression for {name!r} reads back with different names: "
                              f"{sorted(_names(expr))} vs {sorted(_names(parsed))}")
    try:
        same = _same_function(expr, parsed)
    except UnsupportedSBML:
        raise
    except Exception as e:      # an expression numpy can't evaluate is unsupported
        raise UnsupportedSBML(f"expression for {name!r} cannot be evaluated: {e}") from e
    if not same:
        raise UnsupportedSBML(f"expression for {name!r} reads back as a different "
                              f"function: {to_text(expr)!r} vs {text!r}")
    if to_text(parse(text, allowed)) != text:
        raise UnsupportedSBML(f"expression for {name!r} has no stable printed form: {text!r}")
    return text
