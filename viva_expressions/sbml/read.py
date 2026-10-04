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
* The ``time`` csymbol becomes an integrated ``time`` variable with
  ``d(time)/dt = 1`` from 0: the standard autonomization, since OdeProcess
  integrates autonomous right-hand sides.

libsbml's own converters first expand function definitions and initial
assignments and promote local parameters to global ones. Constructs with no
faithful mapping (events, delays, algebraic rules, fast reactions, variable
stoichiometry, species in a compartment whose size changes) raise
``UnsupportedSBML``; nothing is dropped silently.
"""
import keyword
import math
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import libsbml
import sympy as sp

from viva_expressions.expressions import parse
from viva_expressions.sbml.math import RESERVED, UnsupportedSBML, from_ast, to_text

CONVERTERS = ("promoteLocalParameters", "expandFunctionDefinitions",
              "expandInitialAssignments")


@dataclass(frozen=True)
class OdeModel:
    """An SBML model as OdeProcess inputs, plus what's needed to compare it.

    ``kinds`` says what each state or assignment is in SBML terms
    (``concentration``, ``amount``, ``parameter``, ``compartment`` or ``time``),
    and ``sbml_ids`` maps each of those names back to its SBML identifier.
    Names equal SBML identifiers except where one isn't a valid expression
    identifier (a Python keyword, or a clash with the added ``time`` variable).
    """
    model_id: str
    rhs: dict[str, str]
    params: dict[str, float]
    initial: dict[str, float]
    assignments: dict[str, str] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    sbml_ids: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


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


def _namer(m: libsbml.Model, uses_time: bool):
    """SBML id -> expression name: identity unless the id can't be one.

    An id is renamed (suffixed ``_``) when it is a Python keyword, not a Python
    identifier, or a name the printed expressions use for a function or
    constant (``RESERVED``), and the clock is renamed if an id is ``time``.
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
        if keyword.iskeyword(i) or not i.isidentifier() or i in RESERVED:
            while name in taken or keyword.iskeyword(name) or name in RESERVED:
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
        rhs[time_sym], initial[time_name], kinds[time_name] = sp.Integer(1), 0.0, "time"
        notes.append(f"time-dependent model: added state {time_name!r} with "
                     f"d{time_name}/dt = 1 from {time_name} = 0")
    if m.getNumConstraints():
        notes.append(f"{m.getNumConstraints()} constraint(s) ignored: they assert "
                     "conditions during simulation but do not change the dynamics")

    allowed = [*initial, *params]
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
                    notes=tuple(notes))


def _doubles(expr: sp.Basic) -> sp.Basic:
    """``expr`` with every float at double precision (the precision it runs at)."""
    return expr.xreplace({f: sp.Float(float(f)) for f in expr.atoms(sp.Float)})


def _emit(expr: sp.Basic, allowed: list[str], name: str) -> str:
    """Print ``expr`` and check it reparses, in the OdeProcess grammar, to ``expr``.

    The check compares expressions, not text, so a printed name that reparses
    as something else (a constant read as a symbol) is caught. It also rejects
    numbers outside double range, before or after reparsing: sympy folds e.g.
    ``exp(2412.0 - 80*t)`` into ``1e1047*exp(-80*t)``, which evaluates to
    ``inf`` and silently turns trajectories into ``nan``.
    """
    text = to_text(expr)
    try:
        parsed = parse(text, allowed)
    except (ValueError, TypeError, SyntaxError) as e:
        raise UnsupportedSBML(f"expression for {name!r} does not reparse: {e}") from e
    for e in (expr, parsed):
        for number in e.atoms(sp.Number):
            try:
                finite = math.isfinite(float(number))
            except OverflowError:
                finite = False
            if not finite:
                raise UnsupportedSBML(f"expression for {name!r} has a number outside "
                                      f"double range ({number}): {text}")
    if _doubles(parsed) != _doubles(expr):
        raise UnsupportedSBML(f"expression for {name!r} does not reparse to itself: "
                              f"{text!r} reads back as {to_text(parsed)!r}")
    return text
