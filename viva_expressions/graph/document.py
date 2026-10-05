"""Lossless conversion between a process-bigraph document and a directed graph.

General: it knows process-bigraph's document grammar, nothing about any one
process, and uses only public APIs of ``process_bigraph`` and
``bigraph_schema``, so it can move upstream unchanged.

The contract is a pure change of representation::

    document_equal(from_graph(to_graph(doc)), doc)

**Graph.** A ``networkx.MultiDiGraph``. Every node id is the RFC 6901 JSON
Pointer of what it stands for (``""`` is the root). One node per plain
``dict`` (``type(x) is dict``); every other value is kept verbatim, as the
same object, on a leaf node's ``value`` or a link node's ``fields``. Every
dict node keeps ``slots``, its keys in their original order. Lists are
values: a dict inside a list is part of that list's value.

* ``contains`` edges (key ``"contains"``, attribute ``slot``) form the tree.
* A link (process/step) keeps every key except dict ``inputs``/``outputs`` in
  ``fields``, verbatim (configs are opaque). Its ``inputs``/``outputs`` are
  ``ports`` nodes whose leaves are ``port`` nodes. A link whose ``config``
  holds a dict ``state`` (a nested composite) gets a ``config`` node instead,
  with a ``scope`` (its ``state``) and a ``bridge`` of port groups.
* ``wire`` edges (key ``"wire"``) join each port to the deepest existing node
  on its path, in dataflow direction: a link's input flows store -> port and
  its output port -> store; a bridge input flows port -> store and a bridge
  output (or conduit) store -> port. The edge carries the **verbatim**
  ``wire`` plus derived ``resolved`` (``bigraph_schema.schema.resolve_path``
  of the link's parent path + the wire, relative to the enclosing scope;
  ``()`` for a bridge), ``subpath`` (the part of ``resolved`` below the
  endpoint) and ``error`` (why it does not resolve, else ``None``). Port
  nodes are never endpoints.

``kind`` (``document``, ``scope``, ``store``, ``link``, ``config``,
``bridge``, ``ports``, ``port``, ``tree``) is annotation: rebuilding reads
only ``slots``, ``fields``, ``value``, ``contains`` and ``wire.wire``.

**Views** add derived elements: nodes and edges carrying ``view=<name>``
(derived node ids are ``json.dumps([view, pointer, name])``, never a
pointer), and per-node ``annotations[<name>]``. ``from_graph`` ignores them;
``check_views`` recomputes them and raises if they are stale.

**JSON.** ``dumps_graph``/``loads_graph`` use networkx node-link data
(``edges="edges"``) with process-bigraph's native codec,
``BigraphJSONEncoder``/``bigraph_json_hook``. The JSON path is exact up to
that codec's own losses: tuples load as lists, numpy scalars as Python
scalars, a NaN loses its sign and payload, non-string keys inside values
become strings, and a dict whose keys are the codec's tags (``__numpy__``,
``__pint__``, ``__set__``, ...) is read back as the tagged type.
"""
from __future__ import annotations

import json
import struct
import uuid
from collections.abc import Callable, Mapping

import networkx as nx
import numpy as np
import pint
from bigraph_schema.json_codec import BigraphJSONEncoder, bigraph_json_hook
from bigraph_schema.methods import load_protocol
from bigraph_schema.schema import normalize_address, resolve_path
from process_bigraph import Emitter, allocate_core
from process_bigraph.composite_spec import substitute_parameters

FORMAT = "process-bigraph-document-graph/1"
CONTAINS, WIRE = "contains", "wire"
LINK_TYPES = frozenset({"process", "step", "composite", "edge"})
KINDS = frozenset({"document", "scope", "store", "link", "config", "bridge",
                   "ports", "port", "tree"})
BRIDGE_SIDES = {"inputs": "inputs", "outputs": "outputs", "conduits": "outputs"}


class GraphError(ValueError):
    """A graph that ``to_graph`` could not have produced, or a stale view."""


# --- JSON Pointer (RFC 6901) -------------------------------------------------

def escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def pointer(tokens) -> str:
    """The JSON Pointer for a path of string keys."""
    return "".join("/" + escape(t) for t in tokens)


def pointer_tokens(ptr: str) -> list[str]:
    """The keys of a JSON Pointer; raises ``GraphError`` if it is not one."""
    if not isinstance(ptr, str) or (ptr and not ptr.startswith("/")):
        raise GraphError(f"node id {ptr!r} is not a JSON Pointer")
    if ptr == "":
        return []
    return [t.replace("~1", "/").replace("~0", "~") for t in ptr[1:].split("/")]


# --- classification (annotation only) ----------------------------------------

def is_link(value) -> bool:
    """A process-bigraph link node: a typed ``process``/``step``/... dict, or an
    untyped one with an ``address`` and ports (a realized serialization drops
    ``_type`` into the schema)."""
    if type(value) is not dict:
        return False
    if "_type" in value:
        return isinstance(value["_type"], str) and value["_type"] in LINK_TYPES
    return "address" in value and ("inputs" in value or "outputs" in value)


def root_kind(document) -> str:
    if type(document) is not dict:
        return "tree"
    return "document" if type(document.get("state")) is dict else "scope"


def child_kind(parent_kind: str, slot: str, value) -> str | None:
    """The kind of ``value`` under ``slot`` of a ``parent_kind`` node, or
    ``None`` for a link field (kept verbatim, not a node)."""
    plain = type(value) is dict
    if parent_kind in ("scope", "store"):
        return "link" if is_link(value) else "store"
    if parent_kind in ("document", "config"):
        if plain and slot == "state":
            return "scope"
        if plain and slot == "bridge":
            return "bridge"
        return "tree"
    if parent_kind == "link":
        if plain and slot in ("inputs", "outputs"):
            return "ports"
        if plain and slot == "config" and type(value.get("state")) is dict:
            return "config"
        return None
    if parent_kind == "bridge":
        return "ports" if plain and slot in BRIDGE_SIDES else "tree"
    if parent_kind == "ports":
        return "ports" if plain else "port"
    return "tree"


# --- document -> graph -------------------------------------------------------

def _is_core(attrs) -> bool:
    return "view" not in attrs


def _is_port(attrs) -> bool:
    return "slots" not in attrs and "value" not in attrs


def to_graph(document) -> nx.MultiDiGraph:
    """The directed graph of ``document`` (see the module docstring)."""
    G = nx.MultiDiGraph(format=FORMAT, views=[])
    ports = []
    _add(G, document, "", root_kind(document), ports)
    for port, wire in ports:
        _add_wire(G, port, wire)
    return G


def _add(G, value, ptr, kind, ports):
    if kind == "port":
        G.add_node(ptr, kind=kind)
        ports.append((ptr, value))
        return
    if type(value) is not dict:
        G.add_node(ptr, kind=kind, value=value)
        return
    for key in value:
        if type(key) is not str:
            raise TypeError(f"{ptr or '(root)'}: key {key!r} is not a string; "
                            "node ids are JSON Pointers, which need string keys")
    G.add_node(ptr, kind=kind, slots=list(value))
    children, fields = [], {}
    for key, child in value.items():
        ck = child_kind(kind, key, child)
        if ck is None:
            fields[key] = child
        else:
            children.append((key, child, ck))
    if fields:
        G.nodes[ptr]["fields"] = fields
    for key, child, ck in children:
        cptr = ptr + "/" + escape(key)
        G.add_edge(ptr, cptr, key=CONTAINS, slot=key)
        _add(G, child, cptr, ck, ports)


def _parent(G, node):
    for u, _, k, d in G.in_edges(node, keys=True, data=True):
        if k == CONTAINS and _is_core(d):
            return u, d["slot"]
    return None, None


def _child(G, node, slot):
    child = node + "/" + escape(slot)
    if child in G and G.has_edge(node, child, key=CONTAINS) and _is_core(G.nodes[child]):
        return child
    return None


def _enclosing_scope(G, node):
    while node is not None:
        if G.nodes[node].get("kind") == "scope":
            return node
        node, _ = _parent(G, node)
    return ""


def _resolution(G, port, wire):
    """``(endpoint, port_is_source, attrs)`` for ``port`` carrying ``wire``."""
    top, side = port, None
    parent, slot = _parent(G, top)
    while parent is not None and G.nodes[parent].get("kind") == "ports":
        top = parent
        parent, slot = _parent(G, top)
    owner_kind = None if parent is None else G.nodes[parent].get("kind")
    error = None
    if owner_kind == "link":
        side = slot
        scope = _enclosing_scope(G, parent)
        base = pointer_tokens(parent)[len(pointer_tokens(scope)):-1]
        port_is_source = side == "outputs"
    elif owner_kind == "bridge":
        side = BRIDGE_SIDES.get(slot)
        holder, _ = _parent(G, parent)
        scope = _child(G, holder, "state") if holder is not None else None
        if scope is None or G.nodes[scope].get("kind") != "scope":
            scope, error = _enclosing_scope(G, parent), "bridge has no state scope"
        base = []
        port_is_source = side == "inputs"
    else:
        scope, base, port_is_source = _enclosing_scope(G, port), [], True
        error = "port group is not under a link or a bridge"
    if side not in ("inputs", "outputs") and error is None:
        error = f"port group {slot!r} is neither inputs nor outputs"

    resolved = subpath = None
    if error is None:
        if isinstance(wire, str):
            steps = [wire]
        elif isinstance(wire, (list, tuple)):
            steps = list(wire)
        else:
            steps, error = None, f"not a wire: {type(wire).__name__} {wire!r}"
        if steps is not None:
            try:
                resolved = list(resolve_path(list(base) + steps))
            except Exception as e:   # resolve_path raises a bare Exception
                error = str(e)
    endpoint = scope
    if resolved is not None:
        depth = 0
        for step in resolved:
            if type(step) is not str or step == "*":
                break
            child = _child(G, endpoint, step)
            if child is None or _is_port(G.nodes[child]):
                break
            endpoint, depth = child, depth + 1
        subpath = resolved[depth:]
    return endpoint, port_is_source, {"resolved": resolved, "subpath": subpath, "error": error}


def _add_wire(G, port, wire):
    endpoint, port_is_source, attrs = _resolution(G, port, wire)
    u, v = (port, endpoint) if port_is_source else (endpoint, port)
    G.add_edge(u, v, key=WIRE, wire=wire, **attrs)


# --- graph -> document -------------------------------------------------------

def _wire_edges(G, port):
    return ([(u, v, d) for u, v, k, d in G.in_edges(port, keys=True, data=True)
             if k == WIRE and _is_core(d)]
            + [(u, v, d) for u, v, k, d in G.out_edges(port, keys=True, data=True)
               if k == WIRE and _is_core(d)])


def _build(G, node):
    attrs = G.nodes[node]
    if "slots" in attrs:
        fields = attrs.get("fields", {})
        kids = {d["slot"]: c for _, c, k, d in G.out_edges(node, keys=True, data=True)
                if k == CONTAINS and _is_core(d)}
        out = {}
        for slot in attrs["slots"]:
            if slot in fields:
                out[slot] = fields[slot]
            elif slot in kids:
                if kids[slot] != node + "/" + escape(slot):
                    raise GraphError(f"node {kids[slot]!r} under {node!r} slot {slot!r} "
                                     "is not at its JSON Pointer")
                out[slot] = _build(G, kids[slot])
            else:
                raise GraphError(f"node {node!r}: slot {slot!r} has no field and no child")
        return out
    if "value" in attrs:
        return attrs["value"]
    edges = _wire_edges(G, node)
    if len(edges) != 1:
        raise GraphError(f"port {node!r} has {len(edges)} wire edges, not exactly one")
    return edges[0][2]["wire"]


def from_graph(G: nx.MultiDiGraph, strict: bool = True):
    """The document ``G`` represents.

    ``strict`` (default) raises ``GraphError`` unless ``G``'s core (everything
    but view-derived elements) is exactly what ``to_graph`` produces for the
    rebuilt document: a tree of JSON Pointer ids, every port with exactly one
    wire, every endpoint and derived wire attribute equal to its
    re-resolution, every kind and field as classified.
    """
    if G.graph.get("format") != FORMAT:
        raise GraphError(f"not a document graph: format {G.graph.get('format')!r}, "
                         f"expected {FORMAT!r}")
    if "" not in G or not _is_core(G.nodes[""]):
        raise GraphError("no root node ''")
    document = _build(G, "")
    if strict:
        _check_canonical(G, to_graph(document))
    return document


def _core_nodes(G):
    return {n: a for n, a in G.nodes(data=True) if _is_core(a)}


def _core_edges(G):
    return {(u, v, k): d for u, v, k, d in G.edges(keys=True, data=True) if _is_core(d)}


def _same_node(a, b) -> bool:
    keys = {"kind", "slots", "fields", "value"}
    if {k for k in a if k in keys} != {k for k in b if k in keys} \
            or set(a) - keys - {"annotations"}:
        return False
    if a["kind"] != b["kind"] or a.get("slots") != b.get("slots"):
        return False
    if "value" in a and a["value"] is not b["value"]:
        return False
    fa, fb = a.get("fields", {}), b.get("fields", {})
    return set(fa) == set(fb) and all(fa[k] is fb[k] for k in fa)


def _same_edge(key, a, b) -> bool:
    if set(a) != set(b):
        return False
    if key == WIRE:
        return a["wire"] is b["wire"] and all(a[k] == b[k] for k in a if k != "wire")
    return a == b


def _check_canonical(G, canon):
    nodes, want = _core_nodes(G), _core_nodes(canon)
    for n in nodes.keys() - want.keys():
        raise GraphError(f"node {n!r} is not part of the document's graph "
                         "(not reachable from the root at its JSON Pointer)")
    for n in want.keys() - nodes.keys():
        raise GraphError(f"node {n!r} is missing")
    for n, a in nodes.items():
        if not _same_node(a, want[n]):
            raise GraphError(f"node {n!r} differs from to_graph's: "
                             f"kind {a.get('kind')!r} (expected {want[n]['kind']!r}), "
                             f"attributes {sorted(a)} (expected {sorted(want[n])})")
    edges, want_edges = _core_edges(G), _core_edges(canon)
    for e in edges.keys() - want_edges.keys():
        raise GraphError(f"edge {e!r} is not one to_graph produces "
                         "(not a tree edge, or a wire to the wrong endpoint or direction)")
    for e in want_edges.keys() - edges.keys():
        raise GraphError(f"edge {e!r} is missing")
    for e, d in edges.items():
        if not _same_edge(e[2], d, want_edges[e]):
            raise GraphError(f"edge {e!r} attributes differ from their re-resolution: "
                             f"{ {k: v for k, v in d.items() if k != 'wire'} }")


# --- equality ----------------------------------------------------------------

def document_equal(a, b) -> bool:
    """Exact equality: same types, same dict key order, floats bit for bit,
    arrays by dtype, shape and bytes, quantities by magnitude and units, any
    other value by identity or ``==``."""
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return (len(a) == len(b)
                and all(document_equal(x, y) for x, y in zip(a, b))
                and all(document_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(document_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.dtype.hasobject:
            return all(document_equal(x, y) for x, y in zip(a.ravel(), b.ravel()))
        return np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, float):
        return struct.pack("<d", a) == struct.pack("<d", b)
    if isinstance(a, complex):
        return struct.pack("<dd", a.real, a.imag) == struct.pack("<dd", b.real, b.imag)
    if isinstance(a, pint.Quantity):
        return document_equal(a.magnitude, b.magnitude) and a.units == b.units
    if a is b:
        return True
    eq = a == b
    if isinstance(eq, (bool, np.bool_)):
        return bool(eq)
    raise TypeError(f"cannot compare {type(a).__name__} values exactly")


# --- JSON carrier ------------------------------------------------------------

def dumps_graph(G: nx.MultiDiGraph, **kwargs) -> str:
    """Node-link JSON of ``G`` through process-bigraph's native codec."""
    data = nx.node_link_data(G, edges="edges")
    return json.dumps(data, cls=BigraphJSONEncoder, **kwargs)


def loads_graph(text: str) -> nx.MultiDiGraph:
    """Inverse of ``dumps_graph``."""
    data = json.loads(text, object_hook=bigraph_json_hook)
    return nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")


# --- views -------------------------------------------------------------------

def derived_id(view: str, ptr: str, name: str) -> str:
    return json.dumps([view, ptr, name])


def annotate(G, node, view, data):
    G.nodes[node].setdefault("annotations", {})[view] = data


def link_class(fields, core):
    """``(class, error)`` for a link, resolved the way realization does:
    ``normalize_address`` then the core's protocol through ``load_protocol``."""
    if "instance" in fields:
        return type(fields["instance"]), None
    address = normalize_address(fields.get("address", "local:edge"))
    if not (isinstance(address, dict) and "protocol" in address):
        return None, f"not an address: {address!r}"
    try:
        cls = load_protocol(core, core.access(address["protocol"]), address["data"])
    except Exception as e:   # any import/lookup failure is recorded, not raised
        return None, f"{type(e).__name__}: {e}"
    if cls is None:
        return None, f"no link at address {address['protocol']}:{address['data']}"
    return cls, None


def placeholders_view(G, core=None):
    """``param`` nodes for the root's ``parameters`` and ``references`` edges to
    every ``${name}`` site in its ``schema``/``state``.

    Sites are found from ``substitute_parameters``' own behavior, not a copy
    of its pattern: each parameter is substituted by a unique sentinel
    (declared without a type, so it is not coerced) and a site is any string
    whose substitution changed; the sentinels it contains name its parameters.
    """
    view = "placeholders"
    document = _build(G, "")
    params = document.get("parameters") if type(document) is dict else None
    if type(params) is not dict:
        return
    sentinels = {name: f"\x00{uuid.uuid4().hex}\x00" for name in params}
    for name in params:
        G.add_node(derived_id(view, "", name), kind="param", view=view, name=name,
                   declaration=pointer(["parameters", name]))
    try:
        substituted = {key: substitute_parameters(
            document[key], {n: {"default": s} for n, s in sentinels.items()})
            for key in ("schema", "state") if key in document}
    except KeyError as e:
        annotate(G, "", view, {"error": str(e.args[0])})
        return
    for key, new in substituted.items():
        for path, text in _changed_strings(document[key], new, [key]):
            node, depth = "", 0
            for step in path:
                child = _child(G, node, step) if type(step) is str else None
                if child is None:
                    break
                node, depth = child, depth + 1
            sub = path[depth:]
            for name, s in sentinels.items():
                if s in text:
                    G.add_edge(derived_id(view, "", name), node,
                               key="references" + "".join(f"/{escape(str(t))}" for t in sub),
                               view=view, subpath=sub)


def _changed_strings(old, new, path):
    if old is new:
        return
    if isinstance(old, dict):
        for k in old:
            yield from _changed_strings(old[k], new[k], path + [k])
    elif isinstance(old, list):
        for i, (o, n) in enumerate(zip(old, new)):
            yield from _changed_strings(o, n, path + [i])
    elif isinstance(old, str) and old != new:
        yield path, new


def emitters_view(G, core=None):
    """Annotate each link: ``emitter`` is whether its class, resolved through
    ``core``, subclasses ``process_bigraph.Emitter`` (``None`` if unresolved)."""
    core = core if core is not None else allocate_core()
    for node, attrs in list(G.nodes(data=True)):
        if attrs.get("kind") != "link":
            continue
        cls, error = link_class(attrs.get("fields", {}), core)
        annotate(G, node, "emitters", {
            "emitter": None if cls is None else (isinstance(cls, type) and issubclass(cls, Emitter)),
            "class": None if cls is None else f"{cls.__module__}.{cls.__qualname__}",
            "error": error})


def bridges_view(G, core=None):
    """``crosses`` edges pairing a nested composite's outer ports with its
    bridge's same-named ports, in dataflow direction (inputs outer -> inner,
    outputs and conduits inner -> outer)."""
    for node, attrs in list(G.nodes(data=True)):
        if attrs.get("kind") != "link":
            continue
        config = _child(G, node, "config")
        bridge = _child(G, config, "bridge") if config else None
        if bridge is None:
            continue
        for slot, side in BRIDGE_SIDES.items():
            outer, inner = _child(G, node, side), _child(G, bridge, slot)
            if outer and inner:
                _cross(G, outer, inner, side)


def _cross(G, outer, inner, side):
    for slot in G.nodes[outer].get("slots", []):
        o, i = _child(G, outer, slot), _child(G, inner, slot)
        if o is None or i is None:
            continue
        if _is_port(G.nodes[o]) and _is_port(G.nodes[i]):
            u, v = (o, i) if side == "inputs" else (i, o)
            G.add_edge(u, v, key="crosses", view="bridges", side=side)
        elif "slots" in G.nodes[o] and "slots" in G.nodes[i]:
            _cross(G, o, i, side)


VIEWS: dict[str, Callable] = {
    "placeholders": placeholders_view,
    "emitters": emitters_view,
    "bridges": bridges_view,
}


def add_views(G, names=None, views: Mapping[str, Callable] = VIEWS, core=None):
    """Apply ``views[name](G, core)`` for each name (default: all), in place."""
    for name in (list(views) if names is None else names):
        if name in G.graph["views"]:
            raise ValueError(f"view {name!r} is already applied")
        views[name](G, core)
        G.graph["views"].append(name)
    return G


def strip_views(G) -> nx.MultiDiGraph:
    """A copy of ``G`` without any view-derived element."""
    H = G.copy()
    H.remove_edges_from([(u, v, k) for u, v, k, d in H.edges(keys=True, data=True)
                         if not _is_core(d)])
    H.remove_nodes_from([n for n, a in H.nodes(data=True) if not _is_core(a)])
    for _, a in H.nodes(data=True):
        a.pop("annotations", None)
    H.graph["views"] = []
    return H


def _derived(G):
    return ({n: a for n, a in G.nodes(data=True) if not _is_core(a)},
            {(u, v, k): d for u, v, k, d in G.edges(keys=True, data=True) if not _is_core(d)},
            {n: a["annotations"] for n, a in G.nodes(data=True) if "annotations" in a})


def check_views(G, views: Mapping[str, Callable] = VIEWS, core=None):
    """Raise ``GraphError`` unless every applied view equals its recomputation."""
    unknown = [n for n in G.graph.get("views", []) if n not in views]
    if unknown:
        raise GraphError(f"unknown view(s) {unknown}")
    fresh = add_views(strip_views(G), list(G.graph["views"]), views, core)
    if _derived(fresh) != _derived(G):
        raise GraphError(f"stale view(s): {G.graph['views']} differ from their recomputation")
