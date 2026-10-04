---
name: sbml-system
description: Convert models between SBML and viva-expressions — import an SBML model (a file, or a BioModels entry) as an OdeProcess workspace composite, or export an expression-defined model (a workspace composite or a /synthesize-system composite.json) as SBML Level 3. Use when the user wants to bring a BioModels/SBML model into the workspace, hand a model to an SBML tool (Tellurium, COPASI, PySCeS), or check that an expression model and an SBML model agree.
argument-hint: "import <MODEL.xml | BIOMD…> --name NAME | export <composite.yaml | composite.json>"
---

# /sbml-system

Two-way conversion between SBML and the expression layer (`OdeProcess` `rhs` /
`params` / `state_vars`). **You** do the judgment: choosing models,
naming, reading what was rejected and why, explaining it. **The code** does
every translation, and never guesses: `viva_expressions.sbml`.

```
uv run python -m viva_expressions.sbml import MODEL.xml --name NAME [--out DIR] [--interval DT] [--source TEXT]
uv run python -m viva_expressions.sbml export SOURCE --out MODEL.xml [--id MODEL_ID]
uv run python scripts/fetch_biomodels.py BIOMD… --out DIR      # BioModels via viva-biomodels
```
Exit codes: `0` converted, `2` the model uses SBML with no faithful mapping
(the message names it), `1` invalid input.

## Import (SBML → composite)

1. **Get the file.** For a BioModels id, fetch it with `scripts/fetch_biomodels.py`
   (it uses viva-biomodels' client and writes a `SHA256SUMS` manifest). Keep
   files that studies depend on under `workspace/references/` or test data, with
   their checksums.
2. **Run `import`.** By default it writes `NAME.composite.yaml` into
   `viva_expressions/composites/`, where the workbench registry finds it, and
   `NAME.import.json` (the full translation, including `kinds`, `sbml_ids` and
   `notes`). Pass `--source BIOMD…` so the description records provenance.
3. **Read the output and report it:** states, parameters, assignments, and every
   `note`. A time-dependent model reads time from the exact model clock (`time_var`, wired to `global_time`); its `--interval` must be exact in binary (0.5, 0.25, 0.125 = the default), and that note
   must be passed on to the user.
4. **On exit 2,** tell the user exactly which construct blocked it (events,
   delays, algebraic rules, fast reactions, variable stoichiometry, species in a
   changing compartment, piecewise without otherwise, constants outside double
   range, INF/NaN literals). Ids that clash with a keyword, a function name or `pi`
   are renamed with a trailing `_` and recorded in `sbml_ids`. Never edit the SBML to get it through. Changing the model is the
   user's call, never a fix you apply yourself.
5. **Verify** before calling a model usable: smoke-run the composite (`/viva-run`)
   and, for anything a study relies on, compare it with Tellurium on the original
   SBML (as `tests/test_sbml.py` does). Use per-variable
   `max|ours - ref| / max|ref|`, with the threshold fixed **before** looking at
   the results.

What the translation does (SBML L3V2 semantics): kinetic laws are extents per
time; species are integrated in the units their symbol has (concentration unless
`hasOnlySubstanceUnits`); boundary/constant species don't change by reactions; L3
conversion factors apply; libsbml expands function definitions and initial
assignments and promotes local parameters; assignment rules are substituted into
the derivatives **and** emitted as outputs (a `MathExpressionStep`).

## Export (expressions → SBML)

1. `export` takes a `*.composite.yaml` (parameter defaults are substituted) or a
   document JSON (e.g. `/synthesize-system`'s `composite.json`) with one OdeProcess.
2. Each state becomes a parameter with a **rate rule** carrying the exact rhs, and
   parameters and inputs become constants (inputs are held at their given values).
   A `MathExpressionStep` reading the OdeProcess's stores becomes assignment rules. Any other
   process or step raises, because SBML can't carry it, so it is never silently dropped.
   No reactions are invented: a summed rhs does not split into reactions uniquely.
   No units are declared, because OdeProcess has none.
3. libsbml writes doubles to 15 significant digits (a relative change of at most
   5e-15). State that if the user needs bit-exact constants.

## Rules

- Never drop, approximate, or "simplify away" an unsupported construct. Report it.
- Never hand-edit a generated composite or `import.json`. Rerun the import instead.
- Agreement claims need a real comparison (Tellurium/roadrunner on the original
  SBML), never a re-run of our own output against itself.
