"""Fetch BioModels SBML files and pin them with SHA-256 checksums.

Uses viva-biomodels' BioModels client (the workspace import), so the files are
exactly the ones that workspace's simulators run. Writes ``<ID>.xml`` per model
and a ``SHA256SUMS`` manifest (``sha256sum -c`` format) into ``--out``.

    uv run python scripts/fetch_biomodels.py BIOMD0000000012 ... --out tests/data/biomodels
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import biomodels
from viva_biomodels import run_biomodels as rb


def fetch(model_id: str, out: Path) -> Path:
    entry = rb.find_first_sbml(list(rb._iter_entry_files(biomodels.get_metadata(model_id))))
    if entry is None:
        raise SystemExit(f"{model_id}: no SBML file in the BioModels entry")
    dest = out / f"{model_id}.xml"
    shutil.copyfile(rb.fetch_biomodel_files_to_dir(entry, str(out)), dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for model_id in args.ids:
        print(fetch(model_id, args.out))
    sums = sorted(args.out.glob("*.xml"))
    (args.out / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sums), encoding="utf-8")


if __name__ == "__main__":
    main()
