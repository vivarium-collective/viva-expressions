"""Command line for SBML <-> expressions conversion.

    uv run python -m viva_expressions.sbml import MODEL.xml --name NAME [--out DIR] [--interval DT]
    uv run python -m viva_expressions.sbml export SOURCE --out MODEL.xml [--id MODEL_ID]

``import`` writes ``DIR/NAME.composite.yaml`` (a workspace composite spec, default
DIR = the package's ``composites/``) and ``DIR/NAME.import.json`` (the
translation: rhs, params, initial, assignments, kinds, SBML ids, notes).
``export`` reads a document (``.json``, e.g. a ``/synthesize-system``
``composite.json``) or a ``*.composite.yaml`` spec (``${param}`` defaults
substituted) holding one OdeProcess, and writes SBML L3V2.

Exit codes: 0 converted, 2 the model uses SBML with no faithful mapping,
1 invalid input.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml
from process_bigraph.composite_spec import substitute_parameters

from viva_expressions.sbml import UnsupportedSBML, ode_from_state, read_sbml, write_sbml
from viva_expressions.sbml.composite import composite_spec

COMPOSITES = Path(__file__).resolve().parent.parent / "composites"


def _import(args) -> None:
    model = read_sbml(args.model)
    out = args.out or COMPOSITES
    out.mkdir(parents=True, exist_ok=True)
    spec = composite_spec(model, args.name, args.interval, source=args.source or args.model.name)
    (out / f"{args.name}.composite.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    (out / f"{args.name}.import.json").write_text(
        json.dumps(asdict(model), indent=2), encoding="utf-8")
    print(f"{out / f'{args.name}.composite.yaml'}: {len(model.rhs)} states, "
          f"{len(model.params)} parameters, {len(model.assignments)} assignments")
    for note in model.notes:
        print(f"note: {note}")


def _export(args) -> None:
    raw = (json.loads(args.source.read_text(encoding="utf-8")) if args.source.suffix == ".json"
           else yaml.safe_load(args.source.read_text(encoding="utf-8")))
    state = raw.get("state", raw)
    if "parameters" in raw:
        state = substitute_parameters(state, raw["parameters"])
    model_id = args.id or raw.get("name") or args.source.name.split(".")[0]
    args.out.write_text(write_sbml(**ode_from_state(state), model_id=model_id),
                        encoding="utf-8")
    print(args.out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m viva_expressions.sbml",
                                 description="SBML <-> expressions conversion")
    sub = ap.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="SBML -> workspace composite")
    imp.add_argument("model", type=Path)
    imp.add_argument("--name", required=True, help="composite name (file stem)")
    imp.add_argument("--out", type=Path, help="output directory (default: package composites/)")
    imp.add_argument("--interval", type=float, default=0.1, help="OdeProcess interval")
    imp.add_argument("--source", help="provenance text for the description (e.g. a BioModels id)")
    exp = sub.add_parser("export", help="OdeProcess document/spec -> SBML")
    exp.add_argument("source", type=Path)
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument("--id", help="SBML model id (default: spec name or file stem)")
    args = ap.parse_args(argv)
    try:
        (_import if args.command == "import" else _export)(args)
    except UnsupportedSBML as e:
        print(f"unsupported: {e}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
