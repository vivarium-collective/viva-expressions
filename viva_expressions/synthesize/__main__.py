"""CLI: ``uv run python -m viva_expressions.synthesize SPEC [options]``.

Exit codes: 0 feasible and verified in a Composite, 2 no feasible (or no
verified) model, 1 invalid spec.
"""
import argparse
import sys
from pathlib import Path

from viva_expressions.synthesize.motifs import describe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m viva_expressions.synthesize",
        description="Fit and select process-bigraph ODE models from a behavior/data spec.")
    ap.add_argument("spec", nargs="?", type=Path, help="spec YAML (see systems/template.yml)")
    ap.add_argument("--out", type=Path, help="output directory (default: <spec dir>/<name>_out)")
    ap.add_argument("--check", action="store_true", help="validate the spec and exit")
    ap.add_argument("--list-motifs", action="store_true", help="print the motif library and exit")
    ap.add_argument("--interval", type=float, help="Composite interval (default: spec grid spacing)")
    ap.add_argument("--top", type=int, help="candidates listed in report.md (default: all)")
    args = ap.parse_args(argv)

    if args.list_motifs:
        print(describe())
        return 0
    if args.spec is None:
        ap.error("spec is required unless --list-motifs is given")

    # deferred so --list-motifs and --help stay fast
    from viva_expressions.synthesize.pipeline import candidate_models, synthesize
    from viva_expressions.synthesize.spec import SpecError, load_spec

    try:
        spec = load_spec(args.spec)
        if args.check:
            models, notes = candidate_models(spec)
            print(f"{args.spec}: valid ({len(models)} candidate model(s))")
            for n in notes:
                print(f"  {n}")
            return 0
        out = args.out or args.spec.parent / f"{spec.name}_out"
        result = synthesize(spec, out_dir=out, interval=args.interval, top=args.top)
    except SpecError as e:
        print(e, file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"{args.spec}: {e.strerror}", file=sys.stderr)
        return 1

    rep = result.report
    winner = rep["winner"]
    print(f"{spec.name}: {rep['status']}"
          + (f" — winner {winner['name']}, theta {winner['theta']}" if winner else ""))
    print(f"report: {out / 'report.md'}")
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
