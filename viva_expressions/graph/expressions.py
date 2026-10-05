"""The expression layer of a viva-expressions system, as a view on its document graph.

``expression_view`` adds, for every ``OdeProcess`` and ``MathExpressionStep``
link (classes resolved through the core, as realization does):

* ``symbol`` nodes, one per declared name (``role``: state, input, param,
  time for OdeProcess; output, input, param for MathExpressionStep);
* ``depends`` edges ``x -> y`` when ``x`` appears in ``y``'s expression
  (``relation`` ``d/dt`` for an OdeProcess rhs, ``=`` for a step output,
  including output -> output). Edges are **syntactic**: the names in the
  exact text (``viva_expressions.expressions.identifiers``, an AST walk, no
  sympy printing). ``compiled`` says whether the name survives parsing
  (``parse(expr, allowed).free_symbols``): ``x - x`` has an edge from ``x``
  that is not compiled;
* ``binds`` edges between each port and its symbol (input port -> symbol,
  symbol -> output port);
* an annotation per link: ``process``, ``error`` (text that does not parse:
  recorded, never a failed conversion) and ``unwired`` (names the process
  reads that have no input port).

``state_dependency_graph`` projects that onto stores: ``x -> y`` when the
store wired to ``y`` changes with the store wired to ``x``.
"""
import networkx as nx

from viva_expressions.core import build_core
from viva_expressions.expressions import CONSTANTS, identifiers, parse
from viva_expressions.graph.document import VIEWS as DOCUMENT_VIEWS
from viva_expressions.graph.document import (
    add_views,
    annotate,
    child,
    derived_id,
    is_port,
    link_class,
    to_graph,
    wire_edges,
)
from viva_expressions.processes.math_expression import MathExpressionStep
from viva_expressions.processes.ode import OdeProcess

VIEW = "expressions"


def _ode(config):
    rhs = config["rhs"]
    state, inputs = list(config["state_vars"]), list(config.get("input_vars", []))
    time = config.get("time_var") or None
    roles = {**{n: "state" for n in state}, **{n: "input" for n in inputs},
             **{n: "param" for n in config.get("params", {})},
             **({time: "time"} if time else {})}
    exprs = {y: rhs[y] for y in state}
    return roles, exprs, "d/dt", list(roles), state + inputs + ([time] if time else []), state


def _math(config):
    params = config.get("params", {})
    exprs = {e["out"]: e["expr"] for e in config["expressions"]}
    names = set().union(*map(identifiers, exprs.values())) - CONSTANTS
    inputs = sorted(names - set(exprs) - set(params))
    roles = {**{n: "output" for n in exprs}, **{n: "input" for n in inputs},
             **{n: "param" for n in params}}
    return roles, exprs, "=", None, inputs, list(exprs)


def expression_view(G, core=None):
    """Add the expression view to ``G`` in place (see the module docstring)."""
    core = core if core is not None else build_core()
    for link, attrs in list(G.nodes(data=True)):
        if attrs.get("kind") != "link":
            continue
        fields = attrs.get("fields", {})
        cls, _ = link_class(fields, core)
        if not (isinstance(cls, type) and issubclass(cls, (OdeProcess, MathExpressionStep))):
            continue
        note = {"process": cls.__name__, "error": None, "unwired": []}
        annotate(G, link, VIEW, note)
        try:
            roles, exprs, relation, allowed, reads, writes = (
                (_ode if issubclass(cls, OdeProcess) else _math)(fields["config"]))
            edges = []
            for out, text in exprs.items():
                compiled = {s.name for s in parse(text, allowed).free_symbols}
                names = identifiers(text) - (CONSTANTS - set(roles))
                edges += [(x, out, x in compiled) for x in sorted(names)]
        except Exception as e:   # malformed config or unparseable text: recorded on the link
            note["error"] = f"{type(e).__name__}: {e}"
            continue

        for name, role in roles.items():
            G.add_node(derived_id(VIEW, link, name), kind="symbol", view=VIEW,
                       name=name, role=role, link=link)
        for x, y, compiled in edges:
            G.add_edge(derived_id(VIEW, link, x), derived_id(VIEW, link, y), key="depends",
                       view=VIEW, relation=relation, compiled=compiled)
        compiled_reads = {x for x, _, c in edges if c and roles[x] in ("input", "state", "time")}
        inputs, outputs = child(G, link, "inputs"), child(G, link, "outputs")
        for name in reads:
            port = child(G, inputs, name) if inputs else None
            if port is not None and is_port(G.nodes[port]):
                G.add_edge(port, derived_id(VIEW, link, name), key="binds", view=VIEW)
            elif issubclass(cls, OdeProcess) or name in compiled_reads:
                note["unwired"].append(name)
        for name in writes:
            port = child(G, outputs, name) if outputs else None
            if port is not None and is_port(G.nodes[port]):
                G.add_edge(derived_id(VIEW, link, name), port, key="binds", view=VIEW)


VIEWS = {**DOCUMENT_VIEWS, VIEW: expression_view}


def system_graph(document, core=None):
    """``to_graph(document)`` with every view, the expression view included."""
    core = core if core is not None else build_core()
    return add_views(to_graph(document), views=VIEWS, core=core)


def _store(G, link, side, name):
    """The store wired exactly to ``link``'s ``side`` port ``name``, or ``None``."""
    group = child(G, link, side)
    port = child(G, group, name) if group else None
    if port is None or not is_port(G.nodes[port]):
        return None
    (u, v, d), = wire_edges(G, port)
    if d["error"] is not None or d["subpath"]:
        return None
    return v if u == port else u


def state_dependency_graph(G) -> nx.DiGraph:
    """Store -> store dependencies from the expression view's ``depends`` edges
    (an OdeProcess ``x -> y`` becomes store(x) -> store(dy/dt)); parameters,
    and names not wired exactly to an existing store, drop out. Each edge
    lists the ``links`` and ``relations`` it comes from."""
    H = nx.DiGraph()
    for x, y, k, d in G.edges(keys=True, data=True):
        if k != "depends" or d.get("view") != VIEW:
            continue
        sx, sy = G.nodes[x], G.nodes[y]
        if sx["role"] == "param":
            continue
        link = sx["link"]
        source = _store(G, link, "outputs" if sx["role"] == "output" else "inputs", sx["name"])
        target = _store(G, link, "outputs", sy["name"])
        if source is None or target is None:
            continue
        if not H.has_edge(source, target):
            H.add_edge(source, target, links=[], relations=[])
        e = H.edges[source, target]
        if link not in e["links"]:
            e["links"].append(link)
        if d["relation"] not in e["relations"]:
            e["relations"].append(d["relation"])
    return H
