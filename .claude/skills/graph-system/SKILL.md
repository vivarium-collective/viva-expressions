---
name: graph-system
description: Convert a process-bigraph document (a workspace *.composite.yaml, a /synthesize-system composite.json, any composite document JSON) to its directed graph and back, losslessly — wires resolved to the stores they reach, nested composites and bridges, placeholders, emitters, and the expression-level dependencies of OdeProcess / MathExpressionStep models (which variable drives which derivative). Use when the user wants to see or analyze a model's structure as a graph, hand it to a graph tool, get adjacency/Laplacian matrices of it, check that a document survives a graph round trip, or ask what depends on what.
argument-hint: "to-graph <DOC> --out G.json | from-graph <G.json> --out <DOC> | check <DOC> [--run T]"
---

# /graph-system

A document and its graph are the same information: `from_graph(to_graph(d))`
equals `d` exactly (types, key order, floats bit for bit). **You** do the
judgment: which document, which view, what the structure means for the user.
**The code** does every conversion: `viva_expressions.graph`.

```
uv run python -m viva_expressions.graph to-graph DOC --out G.json [--bare]
uv run python -m viva_expressions.graph from-graph G.json --out DOC.yaml|DOC.json
uv run python -m viva_expressions.graph check DOC [--run T]
```
Exit codes: `0` lossless (and, with `--run`, bit-identical re-runs), `1` invalid
input (unreadable file, a graph `to_graph` could not have produced, a stale
view), `2` not lossless.

## What is in the graph

- **Nodes** are JSON Pointers into the document (`/state/stores/x`). Every dict is
  a node with its keys in order (`slots`); other values stay verbatim. A process
  or step is a `link` node whose config stays opaque (`fields`), except a nested
  composite, whose `state` (a `scope`) and `bridge` are expanded.
- **`wire` edges** join each port to the store it reaches, in dataflow direction,
  with the verbatim wire and its `resolved` path (`..` applied, relative to the
  scope). A wire to a store the document does not contain yet (e.g.
  `global_time`) ends at the deepest existing node, with the rest in `subpath`;
  one that cannot resolve carries an `error`.
- **Views** (derived, never part of the document): `placeholders` (`${p}` sites),
  `emitters` (which links are Emitters), `bridges` (outer port ↔ inner bridge
  port), `expressions` (symbols, `depends` edges x → y with relation `d/dt` or
  `=` and a `compiled` flag, `binds` port ↔ symbol, per-link `error`/`unwired`).

In Python:

```python
from viva_expressions.graph.document import to_graph, from_graph, dumps_graph, loads_graph
from viva_expressions.graph.expressions import system_graph, state_dependency_graph
from viva_expressions.graph.matrices import to_matrices

G = system_graph(doc)                    # to_graph + every view
H = state_dependency_graph(G)            # store -> store (x -> dy/dt)
m = to_matrices(G, kinds=("wire",))      # index, A, D, L (scipy.sparse)
```

## Steps

1. **Check first:** `check DOC` (add `--run T` for a model the user will run).
   Report the counts, any wire `error`, and every expression `error` or
   `unwired` name it prints. Those are findings about the model. Report them;
   don't fix them yourself.
2. **Convert** with `to-graph`; `--bare` when the user wants the document's
   structure without derived views.
3. **Back:** `from-graph` accepts only graphs `to_graph` produces. A hand
   edit that leaves the graph inconsistent (a wire re-pointed but not
   re-resolved, a node off its pointer, a stale view) is rejected rather than
   half-read; to change a model, edit the document and regenerate the graph.

## Rules

- The graph is lossless; the matrices are not. `to_matrices` keeps edge counts
  only; `index` maps rows back to node ids. Say so when you hand matrices over.
- The JSON file is exact up to process-bigraph's own codec, whose losses
  all fall inside values: tuples load as lists, numpy scalars as Python
  numbers, a NaN loses its sign and payload, non-string keys become strings,
  a dict subclass becomes a plain dict (strict `from-graph` then rejects the
  graph), a dict keyed by a codec tag (`__set__`, ...) loads as that type.
  `check` exits `2` if a document read from a file hits one of these.
- `depends` edges are syntactic (the names in the text). `compiled: false` means
  parsing cancels the name (`x - x`). Say which one you are quoting.
- An in-memory `Composite` converts only its **realized** document
  (`composite_graph`, labelled `origin: realized`), never the authored one.
- Keys must be strings (JSON Pointer ids); a document with other keys is refused,
  not mangled. A cyclic document (a self-referencing YAML anchor) is refused;
  shared anchors come back as copies.
- A link is what process-bigraph realizes as one: a dict whose own `_type`, or
  else the type the document's `schema` declares for its path (realized
  documents; followed through a map's `_value` and a struct's keys, not a
  tree's leaves), the core resolves to a `Link` (`process`, `step`,
  `composite`, `link`, ...; not `edge`). An untyped dict with no declared link
  type is a store; process-bigraph never runs it either.
