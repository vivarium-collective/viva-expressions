"""Export an OdeProcess model to SBML Level 3 Version 2.

Each state variable becomes a non-constant parameter whose ``RateRule`` is its
right-hand side, and each parameter or input becomes a constant parameter.
Rate rules state exactly the ODEs OdeProcess integrates. Emitting reactions
instead would mean splitting each summed right-hand side into reactions, and
that split is not unique, so it would invent structure the model doesn't have.
Derived quantities (``assignments``) become ``AssignmentRule``\\ s. OdeProcess
is unitless, so no units are declared, and unit consistency is not checked.

libsbml writes every double (attribute or MathML ``<cn>``) to 15 significant
digits and has no option for more, so exported constants are exact to a
relative 5e-15, which is below double precision's 15.95 digits.

Expressions are parsed with the same grammar OdeProcess compiles
(``viva_expressions.expressions.parse``), so the exported math is the
integrated math.
"""
from importlib.metadata import version

import libsbml

from viva_expressions.expressions import parse
from viva_expressions.sbml.math import UnsupportedSBML, to_ast
from viva_expressions.sbml.read import _validate

ODE_SUFFIX = "OdeProcess"
MATH_SUFFIX = "MathExpressionStep"


def _sid(name: str) -> str:
    if not libsbml.SyntaxChecker.isValidSBMLSId(name):
        raise UnsupportedSBML(f"{name!r} is not a valid SBML identifier")
    return name


def _parameter(model: libsbml.Model, name: str, value: float | None, constant: bool):
    p = model.createParameter()
    p.setId(_sid(name))
    p.setConstant(constant)
    if value is not None:
        p.setValue(float(value))


def write_sbml(
    rhs: dict[str, str],
    params: dict[str, float],
    initial: dict[str, float],
    inputs: dict[str, float] | None = None,
    assignments: dict[str, str] | None = None,
    model_id: str = "model",
) -> str:
    """SBML text whose rate rules are ``d(state)/dt = rhs[state]``.

    ``inputs`` (OdeProcess's exogenous variables, held constant per interval)
    are exported at their given values as constants, so the SBML model is the
    system with its inputs held at those values.
    """
    inputs, assignments = inputs or {}, assignments or {}
    if set(rhs) != set(initial):
        raise ValueError(f"rhs and initial must name the same states: "
                         f"{sorted(set(rhs) ^ set(initial))}")
    allowed = [*initial, *inputs, *params]
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId(_sid(model_id))
    model.setNotes(
        '<body xmlns="http://www.w3.org/1999/xhtml"><p>Exported by viva-expressions '
        f'{version("viva-expressions")}. Each state is a parameter with a rate rule '
        "(the OdeProcess right-hand side); no reactions or units are implied.</p></body>")
    for name, value in {**params, **inputs}.items():
        _parameter(model, name, value, constant=True)
    for name, value in initial.items():
        _parameter(model, name, value, constant=False)
        rule = model.createRateRule()
        rule.setVariable(name)
        rule.setMath(to_ast(parse(rhs[name], allowed)))
    for name, expr in assignments.items():
        _parameter(model, name, None, constant=False)
        rule = model.createAssignmentRule()
        rule.setVariable(name)
        rule.setMath(to_ast(parse(expr, allowed + list(assignments))))
    _validate(doc)
    return libsbml.writeSBMLToString(doc)


def _resolve_wire(scope: dict, path: list[str]):
    node = scope
    for key in path:
        if key == "..":
            raise UnsupportedSBML(f"wire {path} leaves the process's scope")
        node = node[key]
    return node


def ode_from_state(state: dict) -> dict:
    """``write_sbml`` arguments from a document state holding one OdeProcess.

    Initial values and inputs are read from the stores the process's input
    wires point at (relative to the process's own scope, as process-bigraph
    resolves them). A ``MathExpressionStep`` reading those stores becomes
    ``assignments``. Emitters are ignored. Any other process or step raises,
    since SBML export can't carry it. ``${param}`` templates must already be
    substituted.
    """
    nodes = [(scope, node) for scope, node in _walk(state)
             if isinstance(node, dict) and node.get("_type") in ("process", "step")]
    odes = [(sc, n) for sc, n in nodes if str(n.get("address", "")).endswith(ODE_SUFFIX)]
    if len(odes) != 1:
        raise ValueError(f"expected exactly one OdeProcess in the document, found {len(odes)}")
    scope, node = odes[0]
    cfg, wires = node["config"], node["inputs"]
    params = {k: float(x) for k, x in cfg.get("params", {}).items()}
    value = {v: float(_resolve_wire(scope, wires[v])) for v in
             [*cfg["state_vars"], *cfg.get("input_vars", [])]}
    assignments = {}
    for sc, n in nodes:
        address = str(n.get("address", ""))
        if n is node or "Emitter" in address:
            continue
        if not address.endswith(MATH_SUFFIX):
            raise UnsupportedSBML(f"SBML export can't carry {address!r}; only one "
                                  "OdeProcess and MathExpressionSteps are exported")
        for port, path in n.get("inputs", {}).items():
            source = sc is scope and path == wires.get(port)
            if not (source or port in n.get("outputs", {})):
                raise UnsupportedSBML(f"MathExpressionStep input {port!r} is not an "
                                      "OdeProcess variable")
        for k, x in n["config"].get("params", {}).items():
            if params.get(k) != float(x):
                raise UnsupportedSBML(f"MathExpressionStep parameter {k!r} differs from "
                                      "the OdeProcess's")
        assignments.update({e["out"]: e["expr"] for e in n["config"]["expressions"]})
    return {"rhs": dict(cfg["rhs"]), "params": params,
            "initial": {v: value[v] for v in cfg["state_vars"]},
            "inputs": {v: value[v] for v in cfg.get("input_vars", [])},
            "assignments": assignments}


def _walk(scope: dict):
    for node in scope.values():
        yield scope, node
        if isinstance(node, dict) and node.get("_type") not in ("process", "step"):
            yield from _walk(node)
