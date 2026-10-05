# Plan: lossless document ↔ directed-graph conversion

Status: **planned, decisions taken (2026-10-05), not started.** Planned by a read-only planning
agent; its load-bearing claims were spot-checked against source (see "Verified"). Paths:
`SP` = `.venv/lib/python3.12/site-packages`.

## Goal
- **(a)** Any viva-expressions system (OdeProcess / MathExpressionStep documents, workspace
  `*.composite.yaml`, `/synthesize-system` `composite.json`) → its directed graph, and back.
- **(b)** Any process-bigraph composite document (in memory or JSON) → its directed graph, and back.
- Lossless = pure representation change: `document_equal(from_graph(to_graph(d)), d)`.

## Decisions (user, 2026-10-05)
1. **(b) lives in viva-expressions for now** (`viva_expressions/graph/document.py`, general, not
   expression-specific) — we cannot contribute to process-bigraph for this project yet. Consequence:
   **public APIs only** from process-bigraph / bigraph-schema. No `_FULL_PLACEHOLDER`,
   `_node_is_emitter`, or other private helpers; where the plan used one, re-derive from public
   behavior (e.g. placeholders: `substitute_parameters` with sentinel overrides; emitters: resolve the
   address via the core and `issubclass(..., Emitter)`), and record each such choice in
   `docs/decisions.yaml`. Keep the module self-contained so it can move upstream later unchanged.
2. **Expression edges: syntactic + `compiled` flag.** Edges from `identifiers()` (AST over the exact
   rhs text, no sympy printing); each edge flagged `compiled = name ∈ parse(expr).free_symbols`.
3. **Configs are opaque verbatim fields**; only nested-composite (scope) configs are expanded.
4. **NaN: document the payload/sign loss on the JSON path** — stay aligned with process-bigraph's
   native codec (`BigraphJSONEncoder` / `bigraph_json_hook`); do not add a stricter codec.
5. (Planner default, not contested) In-memory `Composite` input converts its **realized**
   serialization only (`serialize_schema`/`serialize_state`), labelled as such — the authored document
   is unrecoverable (`SP/bigraph_schema/edge.py:68` stores `core.fill(config_schema, config)`).
6. (Planner default) bigraph-viz / loom consuming this graph: later, separate work.

## Verified against source (2026-10-05)
- networkx and hypothesis are absent from `uv.lock` and the venv (must be added).
- `SP/bigraph_schema/edge.py:68` — Edge stores the filled config.
- `SP/bigraph_viz/methods/generate_graph_dict.py:106-109` — drops non-str wire steps (lossy).
- `SP/vivarium_workbench/loom/src/convert.ts:304-310` — self-described "deliberately lossy and NOT parseable".
- `BigraphJSONEncoder`/`bigraph_json_hook` turn tuples into lists (probe).

## Graph schema
`networkx.MultiDiGraph`; interchange = node-link JSON (`edges="edges"` explicit) written with
`BigraphJSONEncoder`, read with `bigraph_json_hook`. GraphML rejected: raises on non-scalar attributes.
Attribute names avoid node-link reserved `id/source/target/key`.

- **Node ids**: RFC 6901 JSON Pointer into the input (`""` root). Derived nodes: `json.dumps([view, pointer, name])` (starts with `[`, never a pointer → injective).
- **Granularity**: one node per `type(x) is dict`; everything else verbatim (`value`/`fields`, same object).
- **Node kinds** (`kind` is annotation only; losslessness must not depend on it — test 6):
  `document`, `scope`, `store` (container or leaf), `link` (process/step; `fields` = all keys verbatim
  except dict `inputs`/`outputs`), `config` (scope links only), `ports`, `port` (wire leaf);
  derived: `param`, `symbol`.
- **Every dict node** keeps `slots` = all keys in original order (rebuild iterates `slots`).
- **Edges**: `contains` (parent→child, `slot`); `wire` (dataflow direction: inputs endpoint→port,
  outputs port→endpoint; bridge inputs port→endpoint, bridge outputs endpoint→port) carrying the
  **verbatim** `wire` plus derived `resolved`/`subpath`/`error`; endpoint = deepest existing node on
  `bigraph_schema.schema.resolve_path(base + wire)`, base = link's parent path (`SP/bigraph_schema/core.py:1078`) or `()` for bridges (`process_bigraph/composite.py:2270-2275`).
  Derived: `crosses` (outer port → inner bridge port), `references` (param → site), `depends`/`binds` (view a).
- **Rebuild reads only** `slots`, `fields`, `value`, `contains`, `wire.wire`. `from_graph(strict=True)`
  rejects anything `to_graph` could not have produced (non-tree, id≠pointer, port without exactly one
  wire, endpoint ≠ re-resolution).
- **Equality** `document_equal`: exact types; dict key order; floats bitwise; ndarray dtype+shape+bytes;
  pint magnitude+units; else `is` or `==`. JSON path exact up to the codec's own losses
  (tuple→list, numpy scalar→builtin, NaN payload/sign).

Wire grammar (source): port value = str (→`[str]`) | list/tuple path | dict of nested ports;
steps `..` (pop; above top raises), `*` Star, int Index, str Key
(`SP/bigraph_schema/core.py:996-1100`, `schema.py:335-365`). Placeholders `${p}` apply to `schema`
and `state` only (`process_bigraph/composite_spec.py:191-227, 349-351`).

## (a) expression layer
`expression_view(G, core=None)` adds derived `symbol` nodes and `depends` edges (x→y, relation `d/dt`
for OdeProcess, `=` for MathExpressionStep incl. out→out) and `binds` edges (port↔symbol).
`state_dependency_graph(G) -> nx.DiGraph` projects to store→store (x → dy/dt). Unparseable text →
`error` on the link, never a failed conversion. MathExpressionStep references without a port → `unwired`.

## Steps (atomic, each independently verifiable) — feature branch off `main`
1. `pyproject.toml`: `networkx>=3.4` (verify `edges=` kwarg at the pinned minimum), dev `hypothesis`; version bump.
2. `graph/document.py`: `document_equal` + its unit tests.
3. Tree layer: `to_graph`/`from_graph`, pointer ids, `slots`/`fields`/`value`, strict check.
4. Link/ports/wire layer + resolution via public `resolve_path`.
5. Scopes: scope links, `config` node, bridge port groups.
6. `dumps_graph`/`loads_graph`.
7. Views protocol + `placeholders`, `emitters`, `bridges` (public APIs only) + `check_views`.
8. `composite_document(c)` (realized).
9. `graph/expressions.py`: `expression_view`, `system_graph`, `state_dependency_graph`.
10. `graph/__main__.py` CLI: `to-graph`, `from-graph`, `check [--run T]`; exit 0 / 1 invalid / 2 not lossless; utf-8 I/O.
11. `.claude/skills/graph-system/SKILL.md`.
12. README section + `docs/decisions.yaml` entries; `/adversarial` before PR.

## Tests (proposition each falsifies)
1. In-memory round trip on every `viva_expressions/composites/*.composite.yaml`, `out/oscillator/composite.json`, `ode_document` outputs (assignments, `time_var`), process-bigraph test documents (nested wires, bridge), a v2ecoli-baseline fixture — "identity on real documents".
2. JSON round trip + byte-equal re-dump — "node-link + codec is a lossless carrier".
3. Codec-loss inventory — "the only losses are the codec's documented ones".
4. Key order incl. interleaved `_` keys — "dict order survives".
5. Wire grammar matrix; `resolved` paths exist in `Composite(doc).state` — "our resolution matches process-bigraph's".
6. Forced misclassification still round-trips — "losslessness doesn't depend on classification".
7. Strictness: tampered graphs raise named errors — "from_graph accepts only graphs to_graph could produce".
8. Placeholder view = sites `substitute_parameters` changes under sentinels.
9. **Bit-identical re-run** of each workspace composite and the oscillator after round trip; negative control: one-ulp `omega` bump must differ.
10. Expression view: oscillator edges exactly {v→x, x→v, omega→v}; `compiled` ⊆ AST names; Jacobian structural nonzeros ⊆ `depends`; MathExpressionStep edges = its `deps`.
11. View isolation: removing derived elements doesn't change `from_graph`; stale view raises under `check_views`.
12. Hypothesis-generated documents (hostile keys `/ ~ '' .. *`, unicode, `-0.0`, `5e-324`, max double).
13. `composite_document` round trip, and it differs from the authored document.

## Non-goals
Rendering/layout; YAML text/comments; normalization/validation; schema realization; aliasing/cycles;
recovering authored documents from a `Composite`; SBML; graph-editing UX.
