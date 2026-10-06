"""Command line for document <-> graph conversion.

    uv run python -m viva_expressions.graph to-graph DOC --out GRAPH.json [--bare]
    uv run python -m viva_expressions.graph from-graph GRAPH.json --out DOC
    uv run python -m viva_expressions.graph check DOC [--run T]

DOC is a process-bigraph document (``.json``, read with process-bigraph's
codec) or a workspace ``*.composite.yaml`` (``.yaml``/``.yml``). ``to-graph``
writes node-link JSON with every view (``--bare``: none). ``from-graph``
rejects a graph ``to_graph`` could not have produced and any stale view, and
writes the document as JSON or YAML by the suffix of ``--out``. ``check``
converts in memory and through JSON and compares exactly; ``--run T`` also
runs the original and the round-tripped document to ``T`` in a temporary
directory and compares every emitted value bit for bit.

Exit codes: 0 lossless (and, with ``--run``, bit-identical), 1 invalid input,
2 not lossless.
"""
import argparse
import contextlib
import copy
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml
from bigraph_schema.json_codec import BigraphJSONEncoder, bigraph_json_hook
from process_bigraph import Composite, gather_emitter_results
from process_bigraph.composite_spec import CompositeSpec

from viva_expressions.core import build_core
from viva_expressions.graph.document import (
    GraphError,
    check_views,
    document_equal,
    dumps_graph,
    from_graph,
    loads_graph,
    to_graph,
)
from viva_expressions.graph.expressions import VIEWS, system_graph

YAML_SUFFIXES = (".yaml", ".yml")


def read_document(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text, object_hook=bigraph_json_hook)
    if path.suffix in YAML_SUFFIXES:
        return yaml.safe_load(text)
    raise ValueError(f"{path}: expected .json, .yaml or .yml")


def write_document(document, path: Path) -> None:
    if path.suffix == ".json":
        text = json.dumps(document, cls=BigraphJSONEncoder, indent=2)
    elif path.suffix in YAML_SUFFIXES:
        text = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    else:
        raise ValueError(f"{path}: expected .json, .yaml or .yml")
    path.write_text(text, encoding="utf-8")


def _to_graph(args, core) -> int:
    document = read_document(args.document)
    G = to_graph(document) if args.bare else system_graph(document, core)
    args.out.write_text(dumps_graph(G, indent=1), encoding="utf-8")
    print(f"{args.out}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"views {G.graph['views']}")
    return 0


def _from_graph(args, core) -> int:
    G = loads_graph(args.graph.read_text(encoding="utf-8"))
    document = from_graph(G, strict=True)
    check_views(G, VIEWS, core)      # also rejects derived elements no applied view made
    write_document(document, args.out)
    print(args.out)
    return 0


def _emitted(document, spec_file: Path, written: bool, t_end: float, core):
    """Emitter results of ``document`` run to ``t_end`` in a temporary directory.

    A workspace spec (``name`` + ``state``) goes through
    ``CompositeSpec.from_file`` on a file under ``spec_file``'s name in that
    directory: a byte copy of ``spec_file``, or, when ``written``, ``document``
    written there. Both sides of a ``check --run`` thus load from the same
    place, so a difference comes from the documents, not from where they live.
    """
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
        if "name" in document and "state" in document:
            local = Path(tmp) / spec_file.name
            if written:
                write_document(document, local)
            else:
                shutil.copyfile(spec_file, local)
            config = CompositeSpec.from_file(local).to_document()
        else:
            config = copy.deepcopy(document)     # Composite fills its config in place
        composite = Composite(config, core=core)
        composite.run(t_end)
        return gather_emitter_results(composite)


def _check(args, core) -> int:
    document = read_document(args.document)
    G = system_graph(document, core)
    text = dumps_graph(G)
    H = loads_graph(text)
    check_views(H, VIEWS, core)
    in_memory = document_equal(from_graph(G), document)
    through_json = document_equal(from_graph(H), document) and dumps_graph(H) == text

    kinds = Counter(k.split("/")[0] for *_, k in G.edges(keys=True))   # references/<subpath>
    wires = [d for *_, k, d in G.edges(keys=True, data=True) if k == "wire"]
    print(f"{args.document}: {G.number_of_nodes()} nodes, "
          + ", ".join(f"{n} {k}" for k, n in sorted(kinds.items())))
    for d in wires:
        if d["error"]:
            print(f"wire {d['wire']!r}: {d['error']}")
    for node, attrs in G.nodes(data=True):
        note = attrs.get("annotations", {}).get("expressions")
        if note and (note["error"] or note["unwired"]):
            print(f"{node}: {note['process']} error={note['error']} unwired={note['unwired']}")
    print(f"lossless in memory: {in_memory}; through JSON: {through_json}")
    if not (in_memory and through_json):
        return 2

    if args.run is not None:
        ours = _emitted(document, args.document.resolve(), False, args.run, core)
        theirs = _emitted(from_graph(H), args.document, True, args.run, core)
        if not ours:
            print("nothing to compare: the document has no emitter", file=sys.stderr)
            return 1
        same = document_equal(ours, theirs)
        rows = sum(len(r) for r in ours.values())
        print(f"re-run to t={args.run}: {rows} emitted rows, bit-identical: {same}")
        if not same:
            return 2
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m viva_expressions.graph",
                                 description="process-bigraph document <-> directed graph")
    sub = ap.add_subparsers(dest="command", required=True)
    tg = sub.add_parser("to-graph", help="document -> node-link JSON graph")
    tg.add_argument("document", type=Path)
    tg.add_argument("--out", type=Path, required=True)
    tg.add_argument("--bare", action="store_true", help="no views, only the document's graph")
    fg = sub.add_parser("from-graph", help="node-link JSON graph -> document")
    fg.add_argument("graph", type=Path)
    fg.add_argument("--out", type=Path, required=True, help=".json, .yaml or .yml")
    ck = sub.add_parser("check", help="is the conversion lossless for this document?")
    ck.add_argument("document", type=Path)
    ck.add_argument("--run", type=float, metavar="T",
                    help="also re-run both documents to T and compare emitted values bit for bit")
    args = ap.parse_args(argv)
    command = {"to-graph": _to_graph, "from-graph": _from_graph, "check": _check}[args.command]
    try:
        return command(args, build_core())
    except (GraphError, ValueError, KeyError, TypeError, FileNotFoundError,
            yaml.YAMLError) as e:   # GraphError and JSONDecodeError are ValueErrors
        print(f"error: {e}", file=sys.stderr)
        return 1
    except RecursionError:
        print("error: the document is cyclic (e.g. a self-referencing YAML anchor) "
              "or nested deeper than Python's recursion limit", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
