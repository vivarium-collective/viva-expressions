"""Lossless document <-> graph conversion (viva_expressions.graph.document).

Every test runs real documents: the four workspace composites, the
``/synthesize-system`` oscillator (``tests/data/graph/oscillator.composite.json``,
byte-identical to ``python -m viva_expressions.synthesize
systems/examples/oscillator.yml``), ``ode_document`` outputs, process-bigraph's
own nested-composite generator (``grow_divide_agent``, which has a bridge and
``..`` wires), and v2ecoli's published baseline document
(``tests/data/graph/v2ecoli.composites.baseline.json``, v2ecoli
``reports/composite-state/v2ecoli.composites.baseline.json`` at commit
2a2d775c: 1241 nodes, 732 wires). Re-runs are real ``Composite`` runs.
"""
import copy
import json
import math
import struct

import networkx as nx
import numpy as np
import pytest
from bigraph_schema.units import units as ureg
from graph_documents import (
    COMPOSITES,
    REAL,
    WORKSPACE,
    grow_divide_document,
    oscillator,
    workspace_spec,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from process_bigraph import Composite, Step, allocate_core, gather_emitter_results
from process_bigraph.composite_spec import CompositeSpec, substitute_parameters

from viva_expressions.composites import ode_document, run_document
from viva_expressions.core import build_core
from viva_expressions.graph import document as gd
from viva_expressions.graph.document import (
    GraphError,
    add_views,
    check_views,
    composite_document,
    composite_graph,
    document_equal,
    dumps_graph,
    from_graph,
    loads_graph,
    strip_views,
    to_graph,
)


def json_round_trip(G):
    return loads_graph(dumps_graph(G))


# --- 1, 2: identity on real documents, in memory and through JSON ------------

@pytest.mark.parametrize("name", list(REAL))
def test_round_trip_is_identity_on_real_documents(name):
    """Falsifies: from_graph(to_graph(d)) is d, value for value, on real documents."""
    doc = REAL[name]
    G = to_graph(doc)
    back = from_graph(G)
    assert document_equal(back, doc)
    assert sum(1 for *_, k in G.edges(keys=True) if k == "wire") > 0


@pytest.mark.parametrize("name", list(REAL))
def test_json_carrier_is_lossless_and_byte_stable(name):
    """Falsifies: node-link JSON through the native codec carries a document's
    graph losslessly (these documents hold no codec-lossy value) and re-dumps
    byte for byte."""
    doc = REAL[name]
    text = dumps_graph(to_graph(doc))
    H = loads_graph(text)
    assert document_equal(from_graph(H), doc)
    assert dumps_graph(H) == text


# --- 3: the codec's losses, and only those -----------------------------------

def _nan(sign, payload):
    return struct.unpack("<d", struct.pack("<Q", (sign << 63) | (0x7FF << 52) | payload))[0]


LOSSLESS = {
    "neg_zero": -0.0, "tiny": 5e-324, "max": 1.7976931348623157e308, "inf": -math.inf,
    "text": "Ünïcødé ✓  ", "array": np.arange(6, dtype=np.float32).reshape(2, 3),
    "quantity": ureg.Quantity(2.5, "femtogram"), "set": {1, 2}, "bytes": b"\x00\xff",
    "ints": [1, True, None],
}
LOSSY = {   # value, what the codec documents it becomes
    "tuple": ((1, "a"), [1, "a"]),
    "np_float": (np.float64(0.1), 0.1),
    "np_int": (np.int64(7), 7),
    "nan_signed_payload": (_nan(1, 0x1234), float("nan")),
    "int_key": ([{1: "a"}], [{"1": "a"}]),
}


def test_json_losses_are_exactly_the_codecs_documented_ones():
    """Falsifies: the JSON path loses nothing beyond the native codec's documented
    losses (tuple->list, numpy scalar->builtin, NaN sign/payload, non-string keys)."""
    store = {**LOSSLESS, **{k: v for k, (v, _) in LOSSY.items()}}
    doc = {"state": {"store": store, "link": {
        "_type": "step", "address": "local:Nothing", "config": dict(store),
        "inputs": {"p": ("store", "neg_zero")}, "outputs": {}}}}
    back = from_graph(json_round_trip(to_graph(doc)))
    for where in (back["state"]["store"], back["state"]["link"]["config"]):
        for k, v in LOSSLESS.items():
            assert document_equal(where[k], v), k
        for k, (v, becomes) in LOSSY.items():
            assert not document_equal(where[k], v), f"{k} was expected to be lossy"
            assert document_equal(where[k], becomes), k
    assert back["state"]["link"]["inputs"]["p"] == ["store", "neg_zero"]
    assert document_equal(from_graph(to_graph(doc)), doc)   # in memory: no loss at all


# --- 4: key order ------------------------------------------------------------

def test_dict_key_order_survives_including_interleaved_underscore_keys():
    """Falsifies: every dict's key order survives, with _-keys interleaved anywhere."""
    doc = {"state": {
        "z": {"_type": "float", "b": 1.0, "_default": 0.0, "a": 2.0},
        "_meta": "first-class",
        "proc": {"outputs": {"y": ["z", "b"], "_x": ["z", "a"]}, "config": {"_k": 1, "a": 2},
                 "_type": "process", "address": "local:Nothing", "interval": 1.0,
                 "inputs": {"b": ["z", "b"], "_a": ["z", "a"]}},
        "a": 3.0}}
    for back in (from_graph(to_graph(doc)), from_graph(json_round_trip(to_graph(doc)))):
        assert list(back["state"]) == ["z", "_meta", "proc", "a"]
        assert list(back["state"]["z"]) == ["_type", "b", "_default", "a"]
        assert list(back["state"]["proc"]) == list(doc["state"]["proc"])
        assert list(back["state"]["proc"]["outputs"]) == ["y", "_x"]
        assert list(back["state"]["proc"]["config"]) == ["_k", "a"]
        assert document_equal(back, doc)


# --- 5: wire grammar ---------------------------------------------------------

class GraphProbe(Step):
    """A real step with a flat and a nested input port."""
    config_schema = {}

    def inputs(self):
        return {"a": "float", "nested": {"b": "float"}}

    def outputs(self):
        return {"c": "float"}

    def update(self, state):
        return {"c": state["a"] + state["nested"]["b"]}


def wire_table(G):
    return {(v if k == "wire" and G.nodes[v]["kind"] == "port" else u):
            (u if G.nodes[v]["kind"] == "port" else v, d["resolved"], d["subpath"], d["error"])
            for u, v, k, d in G.edges(keys=True, data=True) if k == "wire"}


def _exists(state, path):
    for step in path:
        if not isinstance(state, dict) or step not in state:
            return False
        state = state[step]
    return True


def test_wire_grammar_resolves_as_process_bigraph_does():
    """Falsifies: our resolution (list, tuple, nested ports, '..') matches
    process-bigraph's: every resolved path exists in the realized Composite state.
    (A bare-string wire is covered statically below: process-bigraph 1.8.5's
    view_ports reads it as [str], but realization's port_merges rejects it.)"""
    doc = {"state": {
        "x": 1.0,
        "inner": {"y": 2.0, "probe": {
            "_type": "step", "address": "local:GraphProbe",
            "inputs": {"a": ["..", "x"], "nested": {"b": ["y"]}}, "outputs": {"c": ("out",)}}},
        "probe2": {"_type": "step", "address": "local:GraphProbe",
                   "inputs": {"a": ("x",), "nested": {"b": ["inner", "y"]}},
                   "outputs": {"c": ["z"]}}}}
    G = to_graph(doc)
    assert wire_table(G) == {
        "/state/inner/probe/inputs/a": ("/state/x", ["x"], [], None),
        "/state/inner/probe/inputs/nested/b": ("/state/inner/y", ["inner", "y"], [], None),
        "/state/inner/probe/outputs/c": ("/state/inner", ["inner", "out"], ["out"], None),
        "/state/probe2/inputs/a": ("/state/x", ["x"], [], None),
        "/state/probe2/inputs/nested/b": ("/state/inner/y", ["inner", "y"], [], None),
        "/state/probe2/outputs/c": ("/state", ["z"], ["z"], None),
    }
    core = allocate_core()
    core.register_link("GraphProbe", GraphProbe)
    composite = Composite(copy.deepcopy(doc), core=core)
    composite.run(1.0)
    for _, resolved, _, _ in wire_table(G).values():
        assert _exists(composite.state, resolved), resolved
    assert composite.state["z"] == 3.0 and composite.state["inner"]["out"] == 3.0


def test_bridge_and_nested_scope_wires_exist_in_both_realized_states():
    """Falsifies: bridge wires resolve from the inner scope root and a nested
    link's wires from its parent there, as process-bigraph realizes them."""
    doc = grow_divide_document()
    G = to_graph(doc)
    composite = Composite(copy.deepcopy(doc), core=allocate_core())
    composite.run(1.0)
    inner = composite.state["agents"]["0"]["instance"].state
    for port, (_, resolved, _, error) in wire_table(G).items():
        assert error is None
        scope = inner if port.startswith("/state/agents/0/config/") else composite.state
        assert _exists(scope, resolved), (port, resolved)
    # the bridge's own wires are relative to the inner state root, not the link
    assert wire_table(G)["/state/agents/0/config/bridge/inputs/mass"][1] == ["mass"]
    assert wire_table(G)["/state/agents/0/outputs/environment"][1] == []   # ['..'] from agents


def test_unresolvable_wires_are_kept_verbatim_with_an_error():
    """Falsifies: a wire that cannot resolve ('..' above the top, not a path) is
    still carried verbatim; '*' and integer steps stop the endpoint walk; a bare
    string is the one-step path view_ports reads it as."""
    doc = {"state": {"s": {"a": 1.0}, "p": {
        "_type": "step", "address": "local:Nothing",
        "inputs": {"up": ["..", ".."], "num": 5, "star": ["s", "*"], "idx": ["s", 0],
                   "text": "s"},
        "outputs": {}}}}
    G = to_graph(doc)
    t = wire_table(G)
    assert t["/state/p/inputs/up"][0] == "/state" and "above the top" in t["/state/p/inputs/up"][3]
    assert t["/state/p/inputs/num"] == ("/state", None, None, "not a wire: int 5")
    assert t["/state/p/inputs/star"] == ("/state/s", ["s", "*"], ["*"], None)
    assert t["/state/p/inputs/idx"] == ("/state/s", ["s", 0], [0], None)
    assert t["/state/p/inputs/text"] == ("/state/s", ["s"], [], None)    # view_ports: str -> [str]
    assert document_equal(from_graph(json_round_trip(G)), doc)


# --- 6: classification is annotation -----------------------------------------

def test_forced_misclassification_still_round_trips(monkeypatch):
    """Falsifies: losslessness depends on kinds (rebuilding reads none of them)."""
    kinds = ["link", "store", "ports", "port", "tree", "scope", "config", "bridge", None]

    def scrambled(parent_kind, slot, value):
        return kinds[sum(map(ord, slot)) % len(kinds)]

    for name in ("lotka_volterra.composite.yaml", "grow_divide_agent", "v2ecoli baseline"):
        doc = REAL[name]
        honest = {n: a["kind"] for n, a in to_graph(doc).nodes(data=True)}
        monkeypatch.setattr(gd, "child_kind", scrambled)
        G = to_graph(doc)
        assert {n: a["kind"] for n, a in G.nodes(data=True)} != honest
        assert document_equal(from_graph(G), doc)
        assert document_equal(from_graph(json_round_trip(G)), doc)
        monkeypatch.undo()


# --- 7: strictness -----------------------------------------------------------

def _lv_graph():
    return to_graph(workspace_spec(COMPOSITES / "lotka_volterra.composite.yaml"))


def _move_wire(G):
    u, v, k = next((u, v, k) for u, v, k in G.edges(keys=True) if k == "wire" and v == "/state/stores/x")
    d = G.edges[u, v, k]
    G.remove_edge(u, v, k)
    G.add_edge(u, "/state/stores/y", key=k, **d)


def _second_wire(G):
    G.add_edge("/state/stores/y", "/state/predation/inputs/x", key="wire", wire=["stores", "y"],
               resolved=["stores", "y"], subpath=[], error=None)


def _drop_wire(G):
    G.remove_edge("/state/stores/x", "/state/predation/inputs/x", key="wire")


def _second_parent(G):
    G.add_edge("/state/predation", "/state/stores/x", key="contains", slot="x")


def _rename(G):
    nx.relabel_nodes(G, {"/state/stores/x": "/state/stores/X"}, copy=False)


def _kind(G):
    G.nodes["/state/stores"]["kind"] = "link"


def _resolved(G):
    G.edges["/state/stores/x", "/state/predation/inputs/x", "wire"]["resolved"] = ["stores", "y"]


def _orphan(G):
    G.add_node("/state/ghost", kind="store", value=1.0)


def _format(G):
    del G.graph["format"]


def _unknown_edge(G):
    G.add_edge("/state/stores/x", "/state/stores/y", key="flows")


def _subpath_type(G):
    G.edges["/state", "/state/emitter/inputs/time", "wire"]["subpath"] = [b"global_time"]


def _field_extra(G):
    G.nodes["/state/predation"]["fields"].append(["hidden", 1.0])   # not in slots


def _fields_shape(G):
    G.nodes["/state/predation"]["fields"] = {"interval": 0.1}


TAMPER = {
    "wire to another endpoint": (_move_wire, "edge"),
    "port with two wires": (_second_wire, "2 wire edges"),
    "port without a wire": (_drop_wire, "0 wire edges"),
    "second parent (not a tree)": (_second_parent, "not one to_graph produces"),
    "id is not its pointer": (_rename, "JSON Pointer"),
    "kind changed": (_kind, "differs from to_graph's"),
    "resolved changed": (_resolved, "re-resolution"),
    "unreachable node": (_orphan, "not part of the document's graph"),
    "no format": (_format, "not a document graph"),
    "unknown core edge": (_unknown_edge, "not one to_graph produces"),
    "subpath of another type": (_subpath_type, "re-resolution"),
    "field outside the slots": (_field_extra, "differs from to_graph's"),
    "fields not pairs": (_fields_shape, "not a list of"),
}


@pytest.mark.parametrize("case", list(TAMPER))
def test_strict_rejects_graphs_to_graph_could_not_produce(case):
    """Falsifies: from_graph(strict=True) accepts only graphs to_graph produces."""
    tamper, message = TAMPER[case]
    G = _lv_graph()
    from_graph(G)                     # the untampered graph is accepted
    tamper(G)
    with pytest.raises(GraphError, match=message):
        from_graph(G)


def test_strict_compares_derived_wire_attributes_by_type():
    """Falsifies: strict mode accepts derived wire attributes that are == but not
    what to_graph produces (False / 0.0 for the index step 0)."""
    doc = {"state": {"s": {"a": 1.0}, "p": {"_type": "step", "address": "local:Nothing",
                                           "inputs": {"i": ["s", 0]}, "outputs": {}}}}
    for attr, value in (("resolved", ["s", False]), ("subpath", [0.0])):
        G = to_graph(doc)
        G.edges["/state/s", "/state/p/inputs/i", "wire"][attr] = value
        with pytest.raises(GraphError, match="re-resolution"):
            from_graph(G)


def test_non_strict_rebuilds_a_moved_wire_that_strict_rejects():
    """Falsifies: strictness, not the rebuild, is what rejects a re-pointed wire."""
    G = _lv_graph()
    _move_wire(G)
    assert document_equal(from_graph(G, strict=False),
                          workspace_spec(COMPOSITES / "lotka_volterra.composite.yaml"))


# --- 8: placeholders view ----------------------------------------------------

def _two_values(declared):
    t = declared.get("type")
    return {"float": (1.25, 2.5), "integer": (3, 4), "string": ("p", "q"),
            "boolean": (True, False)}[t]


@pytest.mark.parametrize("path", WORKSPACE, ids=lambda p: p.name)
def test_placeholder_view_matches_substitute_parameters(path):
    """Falsifies: the placeholders view's sites are where substitute_parameters
    puts each parameter (oracle: two typed overrides per parameter, not sentinels)."""
    spec = workspace_spec(path)
    first = next(iter(spec["parameters"]))
    spec["schema"] = {"probe": {"_type": "string", "_default": f"[${{{first}}}]"}}  # an inline site
    G = to_graph(spec)
    add_views(G, ["placeholders"])
    params = spec["parameters"]
    for name, declared in params.items():
        lo, hi = _two_values(declared)
        oracle = set()

        def diff(x, y, path):
            if isinstance(x, dict):
                for k in x:
                    diff(x[k], y[k], path + [k])
            elif isinstance(x, list):
                for i, (p, q) in enumerate(zip(x, y)):
                    diff(p, q, path + [i])
            elif x != y:
                oracle.add(tuple(path))
        for key in ("schema", "state"):
            diff(substitute_parameters(spec[key], params, {name: lo}),
                 substitute_parameters(spec[key], params, {name: hi}), [key])
        node = json.dumps(["placeholders", "", name])
        sites = {tuple(gd.pointer_tokens(v)) + tuple(d["subpath"])
                 for _, v, d in G.out_edges(node, data=True)}
        assert sites == oracle and oracle, name


def test_undeclared_placeholder_is_reported_not_raised():
    """Falsifies: a ${name} with no declaration fails the conversion."""
    spec = {"parameters": {"a": {"type": "float", "default": 1.0}},
            "state": {"x": "${a}", "y": "${missing}"}}
    G = to_graph(spec)
    add_views(G, ["placeholders"])
    assert "missing" in G.nodes[""]["annotations"]["placeholders"]["error"]
    assert document_equal(from_graph(G), spec)


# --- 9: bit-identical re-runs ------------------------------------------------

def _rows(spec):
    spec = {"id": spec["name"], **copy.deepcopy(spec)}
    composite = Composite(CompositeSpec.from_dict(spec).to_document(),
                          core=build_core())
    composite.run(0.5)
    return gather_emitter_results(composite)


@pytest.mark.parametrize("path", WORKSPACE, ids=lambda p: p.name)
def test_round_tripped_workspace_composite_reruns_bit_identically(path, tmp_path, monkeypatch):
    """Falsifies: a document rebuilt from its JSON graph runs differently from the original."""
    monkeypatch.chdir(tmp_path)    # JSONEmitter writes into the working directory
    spec = workspace_spec(path)
    back = from_graph(json_round_trip(to_graph(spec)))
    ours, theirs = _rows(spec), _rows(back)
    assert len(ours[("emitter",)]) > 1                     # it ran past t = 0
    assert document_equal(ours, theirs)


def test_round_tripped_oscillator_reruns_bit_identically_and_one_ulp_does_not():
    """Falsifies: bit-identity of re-runs is too coarse to see a one-ulp change
    (negative control: omega bumped by one ulp must change the trajectory)."""
    doc = oscillator()
    back = from_graph(json_round_trip(to_graph(doc)))
    t0, a = run_document(copy.deepcopy(doc), 2.0)
    t1, b = run_document(copy.deepcopy(back), 2.0)
    assert document_equal((t0, a), (t1, b))
    bumped = copy.deepcopy(back)
    omega = bumped["state"]["ode"]["config"]["params"]["omega"]
    bumped["state"]["ode"]["config"]["params"]["omega"] = float(np.nextafter(omega, np.inf))
    _, c = run_document(bumped, 2.0)
    assert not document_equal(a, c)


# --- 11: view isolation ------------------------------------------------------

def test_views_are_isolated_from_the_document():
    """Falsifies: derived elements influence from_graph, or check_views misses a stale view."""
    doc = grow_divide_document()
    G = add_views(to_graph(doc))
    assert G.graph["views"] == ["placeholders", "emitters", "bridges"]
    assert sum(1 for *_, k in G.edges(keys=True) if k == "crosses") == 3
    check_views(G)
    check_views(json_round_trip(G))
    assert document_equal(from_graph(G), doc)
    assert document_equal(from_graph(strip_views(G)), doc)
    stale = G.copy()
    stale.remove_edge(*next((u, v, k) for u, v, k in stale.edges(keys=True) if k == "crosses"))
    assert document_equal(from_graph(stale), doc)
    with pytest.raises(GraphError, match="stale"):
        check_views(stale)

    spec = workspace_spec(COMPOSITES / "lotka_volterra.composite.yaml")
    H = add_views(to_graph(spec), ["placeholders"])
    H.nodes["/state/stores/x"]["value"] = 10.0          # the document changed under the view
    with pytest.raises(GraphError, match="stale"):
        check_views(H)


# --- 12: generated documents -------------------------------------------------

KEYS = st.one_of(st.sampled_from(["/", "~", "~1", "", "..", "*", "a/b~", "_type", "inputs",
                                  "outputs", "config", "state", "bridge", "é✓"]),
                 st.text(max_size=4))
LEAVES = st.one_of(
    st.none(), st.booleans(), st.integers(),
    st.floats(allow_nan=False), st.sampled_from([-0.0, 5e-324, 1.7976931348623157e308]),
    st.text(max_size=6), st.lists(st.one_of(st.integers(), st.text(max_size=3)), max_size=3))
WIRES = st.one_of(st.sampled_from(["..", "*"]) | st.text(max_size=3),
                  st.lists(st.one_of(st.sampled_from(["..", "*"]), st.text(max_size=3),
                                     st.integers(-1, 2)), max_size=4))


def _link(children):
    return st.fixed_dictionaries(
        {"_type": st.sampled_from(["process", "step"]), "address": st.just("local:Nothing")},
        optional={"config": st.one_of(st.dictionaries(KEYS, LEAVES, max_size=3),
                                      st.fixed_dictionaries({"state": children})),
                  "inputs": st.dictionaries(KEYS, st.one_of(WIRES, st.dictionaries(KEYS, WIRES,
                                                                                   max_size=2)),
                                            max_size=3),
                  "outputs": st.dictionaries(KEYS, WIRES, max_size=3)})


STATES = st.recursive(LEAVES, lambda c: st.one_of(st.dictionaries(KEYS, c, max_size=4), _link(c)),
                      max_leaves=25)
DOCUMENTS = st.one_of(STATES, st.fixed_dictionaries(
    {"state": st.dictionaries(KEYS, STATES, max_size=4)},
    optional={"bridge": st.fixed_dictionaries({"inputs": st.dictionaries(KEYS, WIRES, max_size=2),
                                               "outputs": st.dictionaries(KEYS, WIRES, max_size=2)})}))


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(DOCUMENTS)
def test_generated_documents_round_trip(doc):
    """Falsifies: the conversion is lossless and strict-consistent beyond the
    hand-picked documents (hostile keys, unicode, -0.0, subnormals, max double)."""
    G = to_graph(doc)
    assert document_equal(from_graph(G), doc)
    text = dumps_graph(G)
    H = loads_graph(text)
    assert document_equal(from_graph(H), doc)
    assert dumps_graph(H) == text


def test_codec_tag_keys_on_a_link_survive_json():
    """Falsifies: a link key named like a codec tag (__set__) breaks the graph JSON
    (fields are [key, value] pairs, so the codec never reads link keys)."""
    doc = {"state": {"p": {"_type": "step", "address": "local:Nothing", "__set__": [1, 2],
                           "__numpy__": True, "inputs": {}, "outputs": {}},
                     "s": {"__set__": True, "data": [1]}}}
    assert document_equal(from_graph(json_round_trip(to_graph(doc))), doc)


def test_dict_subclass_values_are_exact_in_memory_and_rejected_by_strict_after_json():
    """Falsifies: a dict subclass (kept verbatim) loses its type in memory, or its
    JSON-flattened form passes strict mode as if to_graph had produced it."""
    from collections import OrderedDict
    doc = {"state": {"o": OrderedDict(a=1.0)}}
    assert document_equal(from_graph(to_graph(doc)), doc)
    H = json_round_trip(to_graph(doc))
    with pytest.raises(GraphError, match="missing"):
        from_graph(H)
    assert from_graph(H, strict=False) == {"state": {"o": {"a": 1.0}}}


def test_loads_graph_refuses_non_graph_json():
    """Falsifies: JSON that is not node-link graph data is half-read instead of refused."""
    for text in ("[]", "{}", '{"graph": {}, "nodes": []}'):
        with pytest.raises(GraphError, match="not node-link"):
            loads_graph(text)


def test_non_string_keys_are_refused_not_corrupted():
    """Falsifies: a dict key that has no JSON Pointer is silently mangled."""
    with pytest.raises(TypeError, match="not a string"):
        to_graph({"state": {1: 2.0}})


# --- 13: realized documents --------------------------------------------------

def test_composite_document_round_trips_and_is_not_the_authored_document():
    """Falsifies: a Composite's realized document round-trips (and is labelled as
    realized, because it is not the authored one)."""
    authored = ode_document(rhs={"x": "-k*x"}, params={"k": 1.0}, initial={"x": 1.0}, interval=0.5)
    composite = Composite(copy.deepcopy(authored), core=allocate_core())
    composite.run(1.0)
    realized = composite_document(composite)
    assert document_equal(from_graph(to_graph(realized)), realized)
    assert document_equal(from_graph(json_round_trip(to_graph(realized))), realized)
    assert not document_equal(realized, authored)
    assert realized["state"]["ode"]["config"]["max_abs"] == 1e12   # a filled default
    G = composite_graph(composite)
    assert G.graph["origin"] == "realized"
    assert G.nodes["/state/ode"]["kind"] == "link"                  # untyped, by address + ports


@pytest.mark.parametrize("store", [
    {"address": "12 Main St", "inputs": {"a": 1.0}},                  # ports are not wirings
    {"address": "12 Main St", "inputs": {"a": ["x"]}, "zip": "02139"},  # a key no link has
    {"address": 7, "outputs": {"a": ["x"]}},                          # not a link address
])
def test_a_store_shaped_like_an_untyped_link_stays_a_store(store):
    """Falsifies: an untyped dict counts as a link on ``address`` + ports alone
    (the earlier rule, which misread these stores' contents as wires)."""
    document = {"state": {"s": store}}
    G = to_graph(document)
    assert G.nodes["/state/s"]["kind"] == "store"
    assert not [k for *_, k in G.edges(keys=True) if k == "wire"]
    assert document_equal(from_graph(G), document)


def test_an_untyped_authored_link_is_still_a_link():
    """Falsifies: the narrowed rule drops a real untyped link (string address,
    wiring ports, only link fields)."""
    link = {"address": "local:RAMEmitter", "config": {"emit": {"x": "float"}},
            "inputs": {"x": ["x"]}, "_contract": {}}
    G = to_graph({"state": {"x": 1.0, "emitter": link}})
    assert G.nodes["/state/emitter"]["kind"] == "link"
    assert [k for *_, k in G.edges(keys=True) if k == "wire"] == ["wire"]
