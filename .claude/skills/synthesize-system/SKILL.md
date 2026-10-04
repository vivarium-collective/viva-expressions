---
name: synthesize-system
description: Build an executable process-bigraph ODE model of a dynamical system from its expected outputs — qualitative behavior (steady state, settling time, monotonicity, bounds, oscillation), time-series data, or both. Use when the user wants a model "that does X", wants to fit or discover equations from trajectories, or asks which mechanism (logistic, Hill, Michaelis-Menten, oscillator, predator-prey, Goodwin repressor, ...) explains a behavior.
argument-hint: "[spec.yml | free-text description of the system's behavior]"
---

# /synthesize-system

Turn "what the system should do" into a fitted, selected, Composite-verified
process-bigraph model. **You** do the judgment (interview, behavior → features,
candidate structures, revision). **The code** does everything numeric and never
calls an LLM: validation, fitting, SINDy discovery, selection, identifiability,
export, verification.

```
uv run python -m viva_expressions.synthesize SPEC [--out DIR] [--check] [--interval DT] [--top N]
uv run python -m viva_expressions.synthesize --list-motifs
```
Exit codes: `0` feasible and verified in a real Composite, `2` nothing feasible
(or not verified), `1` invalid spec (errors name the field path).

## Workflow

1. **Spec.** If given a spec path, read it. Otherwise interview the user and
   write one from `systems/template.yml` (examples in `systems/examples/`).
   Ask only for what you can't infer:
   - state variables and their initial values (`variables.y[].value`), any
     constant inputs (`variables.x`), physical bounds (e.g. `[0, null]`);
   - simulated horizon `time.t_end` (an oscillation needs ≥ 2 periods) and `n_points`;
   - data, if any: CSV path (relative to the spec), time column, state → column map.
     Data times must lie in `[0, t_end]`; integration starts at t = 0 from the
     declared initial values.
2. **Behavior → features.** Translate the free-text description into
   `behavior.features`, keeping the text in `behavior.description`. **Show the
   mapping and confirm it before fitting** — every number in it is a claim
   about the system. Tolerance semantics:
   - `steady_state.tol`: absolute, in the variable's units;
   - `settling_time.tol`: absolute time; `band` is a fraction of the total change;
   - `oscillation.tol`: relative (fraction) on period and amplitude; `sustained: true`
     means the last cycle keeps the previous cycle's amplitude;
   - `monotonic` and `bounds` have no tolerance (violations beyond integrator noise fail).
   Never invent a tolerance the user would reject; ask when it matters.
3. **Candidates.** Run `--list-motifs` (the library is the source of truth; do
   not rely on memory). Propose 2–6 structures, each with a one-line mechanistic
   justification, as `{motif, bind, params}` or custom `{name, rhs, params}`.
   - If a variable shares a name with a motif parameter (e.g. a state `K`
     with `logistic`), give the candidate a `prefix` (e.g. `prefix: g_`).
   - Set `params` bounds from the spec's scale (e.g. logistic `K` around the
     target steady state); defaults are wide and slow the search.
   - Include at least one simpler rival, so selection is a real comparison.
   - With data for every state, SINDy adds polynomial candidates automatically;
     it cannot express Hill or Michaelis-Menten terms, so propose those yourself.
4. **Validate** with `--check` and fix every reported field.
5. **Run.** Expect seconds per parameter; a 4-parameter oscillator can take ~1 min.
6. **Read `report.md`** (and `report.json` for exact numbers):
   - **verified**: present the winner's equations and parameters, the
     Composite re-score table, and the alternatives it beat and why (AICc with
     data; parsimony then loss without).
   - **Not identifiable** parameters: say plainly that the spec doesn't pin them
     down, and what extra behavior or data would (e.g. a settling time pins a rate).
   - **infeasible**: use each candidate's worst feature to revise the structures
     (not the tolerances) and rerun — at most 3 rounds, then report what was tried
     and what failed. A worst residual of exactly 10 means the simulation diverged
     or never produced the measured quantity (e.g. no oscillation at all).
7. **Hand off** `composite.json`: a self-contained process-bigraph document (one
   `OdeProcess`) that runs with `Composite(doc, core=allocate_core())`, or via
   `viva_expressions.composites.run_document(doc, t_end)`. To hand it to an SBML
   tool (Tellurium, COPASI, PySCeS), export it with `/sbml-system export composite.json`.

## Rules

- **Never loosen a tolerance, drop a feature, or widen a bound without the
  user's explicit agreement.** Those change the question, not the answer.
- Never hand-edit reports or `composite.json`; rerun instead.
- Never present an unverified winner as a result. `unverified` means the
  exported Composite either failed a behavior feature the fit satisfied, or
  diverged from the fitted trajectory by more than the agreement limit; the
  report's verification section says which. Report it as a defect.
- Constraints in `constraints:` are free text and are not scored; if one is
  scorable, turn it into a feature (with the user's agreement).
