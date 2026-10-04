---
name: new-work-candidates
description: New study and investigation proposals to seed the viva-expressions repo.
author: Alex(@AlexPatrie)
---

## What the workspace provides

- OdeProcess integrates d(state)/dt = rhs, with expressions written as strings. It uses scipy solve_ivp (default LSODA) and has a divergence guard at max_abs. It emits float deltas.
- MathExpressionStep compiles sympy expressions into a dependency-ordered DAG of derived values. It takes params as constants and reads its input ports from the free symbols in the expressions.
- Together they let you write a kinetic model as strings and compose it with other modules through plain float ports.
- Installed modules are viva-basic-processes and viva-emitters. Everything else in the catalog (tellurium, copasi, pysces, comets, nfsim, smoldyn, torch and others) needs /api/catalog-install.

---

## Candidates, strongest first

1. (CREATED:COMPLETE): **Single study: expression-defined kinetics against analytic solutions:**
This is the foundation for creating new models, and it needs no installs.
Baselines: logistic growth, Michaelis–Menten, and a Lotka–Volterra first integral, each with a closed-form or conserved quantity to check against.
Variants: method, rtol, and stiff parameter regimes.
Also test the max_abs guard on dx/dt = x².
Pass criteria are numeric error bands.

2. (IN PROGRESS) **Single study: gene-circuit dynamics (repressilator or toggle switch), written purely as expressions:**
Variants sweep Hill coefficient and degradation rate to find the oscillation-to-steady-state boundary.
MathExpressionStep supplies the derived observables (period, amplitude proxies, promoter activity).
It's still workspace-only, and it's the first genuinely biological model.

3. (PENDING): **Multi-study investigation: "Does the expression layer reproduce established simulators?:**
It compares OdeProcess with viva-tellurium, viva-copasi and viva-pysces on the same models from candidates 1 and 2, ideally taken from BioModels via viva-biomodels.
One study per model class: kinetics, oscillator, and a stiff case.
The shared question is trajectory agreement and cost. It also shows when users should write expressions and when they should wrap an SBML model.

4. (PENDING): **Multi-study investigation: composition with existing modules:**
Use expressions as glue: an OdeProcess controller, such as inducible gene expression or a substrate feed, driving viva-comets or spatio-flux (dFBA).
MathExpressionStep converts units and computes yield or growth-rate observables.
Studies: (a) the port and type contract with a one-way coupling, (b) closed-loop feedback, (c) sensitivity of the result to the coupling interval.
This most directly tests "compose with existing viva-* modules". It's also the riskiest, because I haven't verified the ports of the uninstalled modules.

5. (PENDING): **Single study: deterministic against stochastic validity:**
Compare OdeProcess with viva-smoldyn or viva-nfsim as copy number drops.
Variants sweep molecule count to find where the ODE approximation breaks down.
It needs the stochastic wrapper installed first.

---

## Suggested Order

Suggested order: 1 → 2 → 3, then 4 once 3 shows the ports line up. Candidates 1 and 2 can start now, and the investigation in 3 can reuse their baselines.

### Before building any of these confirm three things

a.) The remaining registry steps and processes.
b.) The port and type contracts of whichever modules you install.
c.) The viva-emitters outputs, so the studies have observables to check.