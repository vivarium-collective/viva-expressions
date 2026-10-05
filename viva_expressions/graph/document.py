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

import struct

import numpy as np
import pint


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
