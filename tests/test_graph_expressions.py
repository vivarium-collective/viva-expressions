"""The expression view (viva_expressions.graph.expressions) on real documents.

Oracles come from the processes themselves: the Jacobian ``compile_rhs``
builds for OdeProcess, and the ``deps``/``input_names`` a real
``MathExpressionStep`` computes when it is constructed.
"""
import copy

import pytest
import sympy as sp
from graph_documents import COMPOSITES, REAL, WORKSPACE, oscillator, workspace_spec
from process_bigraph import allocate_core

from viva_expressions.composites import ode_document
from viva_expressions.core import build_core
from viva_expressions.expressions import compile_rhs
from viva_expressions.graph.document import (
    check_views,
    document_equal,
    dumps_graph,
    from_graph,
    loads_graph,
)
from viva_expressions.graph.expressions import VIEWS, state_dependency_graph, system_graph
from viva_expressions.processes.math_expression import MathExpressionStep


def depends(G, link=None):
    """{(x, y): (relation, compiled)} over the expression view's edges."""
    return {(G.nodes[x]["name"], G.nodes[y]["name"]): (d["relation"], d["compiled"])
            for x, y, k, d in G.edges(keys=True, data=True)
            if k == "depends" and (link is None or G.nodes[x]["link"] == link)}


def ode_links(doc):
    state = doc["state"]
    return {f"/state/{k}": v for k, v in state.items()
            if isinstance(v, dict) and v.get("_type") == "process"}


def test_oscillator_edges_are_exactly_its_equations():
    """Falsifies: the view finds exactly x' = v, v' = -omega**2 x (no more, no less)."""
    G = system_graph(oscillator())
    assert depends(G) == {("v", "x"): ("d/dt", True), ("x", "v"): ("d/dt", True),
                          ("omega", "v"): ("d/dt", True)}
    note = G.nodes["/state/ode"]["annotations"]["expressions"]
    assert note == {"process": "OdeProcess", "error": None, "unwired": []}
    assert G.nodes["/state/emitter"]["annotations"]["emitters"]["emitter"] is True


@pytest.mark.parametrize("name", [p.name for p in WORKSPACE] + ["oscillator",
                                                               "ode_document(assignments, time_var)"])
def test_jacobian_nonzeros_are_depends_edges(name):
    """Falsifies: every structural nonzero of the compiled Jacobian d f_i / d y_j
    (built by compile_rhs, the integrator's own) is a depends edge y_j -> y_i."""
    doc = REAL[name]
    G = system_graph(doc)
    for link, node in ode_links(doc).items():
        cfg = node["config"]
        rhs = compile_rhs(cfg["rhs"], cfg["state_vars"], cfg.get("input_vars", []),
                          list(cfg["params"]), time=cfg.get("time_var") or None)
        y = [sp.Symbol(v) for v in rhs.state]
        jac = sp.Matrix([rhs.exprs[v] for v in rhs.state]).jacobian(y)
        edges = depends(G, link)
        nonzero = {(rhs.state[j], rhs.state[i]) for i in range(len(y)) for j in range(len(y))
                   if jac[i, j] != 0}
        assert nonzero and nonzero <= set(edges)
        assert all(edges[e][1] for e in nonzero)                 # and they are compiled


def test_compiled_is_a_subset_of_the_syntactic_names():
    """Falsifies: an edge is flagged compiled when parsing cancels its name
    (x - x), or a syntactic name is dropped."""
    doc = ode_document(rhs={"x": "x - x + k*y", "y": "-y"}, params={"k": 2.0},
                       initial={"x": 1.0, "y": 1.0})
    edges = depends(system_graph(doc))
    assert edges == {("x", "x"): ("d/dt", False), ("k", "x"): ("d/dt", True),
                     ("y", "x"): ("d/dt", True), ("y", "y"): ("d/dt", True)}


def test_math_expression_edges_are_the_steps_own_deps():
    """Falsifies: the step's output -> output edges and inputs are what a real
    MathExpressionStep computes (deps, input_names), and stores project correctly."""
    assignments = {"a": "S*2", "b": "a + P", "c": "b*a - pi + c0", "d": "a - a"}
    doc = ode_document(rhs={"S": "-k*S", "P": "k*S"}, params={"k": 0.5, "c0": 1.0},
                       initial={"S": 1.0, "P": 0.0}, assignments=assignments)
    G = system_graph(doc)
    step = MathExpressionStep(copy.deepcopy(doc["state"]["assignments"]["config"]),
                              core=allocate_core())
    edges = depends(G, "/state/assignments")
    outs = set(assignments)
    compiled_out = {(x, y) for (x, y), (_, c) in edges.items() if c and x in outs}
    assert compiled_out == {(x, y) for y, xs in step.deps.items() for x in xs}
    roles = {G.nodes[n]["name"]: G.nodes[n]["role"] for n in G.nodes
             if G.nodes[n].get("link") == "/state/assignments"}
    assert sorted(n for n, r in roles.items() if r == "input") == step.input_names
    assert edges[("a", "d")] == ("=", False) and "pi" not in roles
    H = state_dependency_graph(G)
    assert H.has_edge("/state/a", "/state/b") and H.has_edge("/state/P", "/state/b")
    assert H.has_edge("/state/S", "/state/P")          # dP/dt = k*S, from the OdeProcess
    assert not any("k" in e for e in H.edges)           # parameters are not stores


def test_state_dependency_graph_of_lotka_volterra():
    """Falsifies: x -> dy/dt projects to store x -> store y for a workspace composite."""
    G = system_graph(workspace_spec(COMPOSITES / "lotka_volterra.composite.yaml"))
    H = state_dependency_graph(G)
    x, y = "/state/stores/x", "/state/stores/y"
    assert set(H.edges) == {(x, x), (y, x), (x, y), (y, y)}
    assert H.edges[x, y] == {"links": ["/state/predation"], "relations": ["d/dt"]}


def test_unparseable_and_unwired_are_reported_not_raised():
    """Falsifies: bad expression text fails the conversion, or a name read with
    no port goes unreported."""
    doc = ode_document(rhs={"x": "-x"}, params={}, initial={"x": 1.0})
    doc["state"]["ode"]["config"]["rhs"]["x"] = "-x +* 2"
    doc["state"]["calc"] = {
        "_type": "step",
        "address": "local:!viva_expressions.processes.math_expression.MathExpressionStep",
        "config": {"expressions": [{"out": "z", "expr": "q*2 + x"}]},
        "inputs": {"x": ["x"]}, "outputs": {"z": ["z"]}}
    G = system_graph(doc)
    assert "cannot parse" in G.nodes["/state/ode"]["annotations"]["expressions"]["error"]
    assert G.nodes["/state/calc"]["annotations"]["expressions"]["unwired"] == ["q"]
    assert document_equal(from_graph(G), doc)


@pytest.mark.parametrize("name", list(REAL))
def test_system_graph_views_survive_json_and_stay_fresh(name):
    """Falsifies: the full system graph (all views) is lossless through JSON,
    check_views accepts what system_graph produced (given the same core), and
    every expression link in these real documents parses."""
    doc = REAL[name]
    G = system_graph(doc)
    H = loads_graph(dumps_graph(G))
    assert document_equal(from_graph(H), doc)
    check_views(H, VIEWS, core=build_core())
    notes = [a["annotations"]["expressions"] for _, a in G.nodes(data=True)
             if "expressions" in a.get("annotations", {})]
    assert all(n["error"] is None and n["unwired"] == [] for n in notes)
