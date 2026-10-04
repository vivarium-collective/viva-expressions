# Pinned BioModels test files

SBML files from [BioModels](https://www.ebi.ac.uk/biomodels/) (EMBL-EBI). Each was
fetched on 2026-10-04 through viva-biomodels' BioModels client by:

    uv run python scripts/fetch_biomodels.py BIOMD0000000001 BIOMD0000000005 \
        BIOMD0000000010 BIOMD0000000012 BIOMD0000000041 BIOMD0000000254 \
        BIOMD0000000400 BIOMD0000000700 BIOMD0000001000 --out tests/data/biomodels
    # added 2026-10-04 (regression for the time-clock fix):
    uv run python scripts/fetch_biomodels.py BIOMD0000000678 --out tests/data/biomodels
    # added 2026-10-04 (regression for the reparse-normalization fix):
    uv run python scripts/fetch_biomodels.py BIOMD0000000051 BIOMD0000000150 BIOMD0000000844 --out tests/data/biomodels

`SHA256SUMS` pins the exact bytes (`shasum -a 256 -c SHA256SUMS`, and
`test_pinned_files_match_checksums`). BioModels distributes its models under
CC0 1.0 (public domain dedication). See https://www.ebi.ac.uk/biomodels/faq.

`tests/test_sbml.py` (`IMPORTABLE`) records which SBML feature each model covers.
`BIOMD0000000001` has an event and is the rejection case.
