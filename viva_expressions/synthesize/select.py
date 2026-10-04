"""Rank fitted candidates: feasible before infeasible, then by evidence.

With data, feasible candidates rank by AICc (fit quality traded against
parameter count). Without data there is no likelihood, so they rank by
parsimony, then loss. Infeasible candidates rank by how badly their worst
behavior feature misses, for the revision loop.
"""
from viva_expressions.synthesize.fit import FitResult


def worst(result: FitResult) -> float:
    return max((b["worst"] for b in result.behavior), default=0.0)


def rank(results: list[FitResult]) -> list[FitResult]:
    feasible = [r for r in results if r.feasible]
    infeasible = [r for r in results if not r.feasible]
    if feasible and all(r.aicc is not None for r in feasible):
        feasible.sort(key=lambda r: (r.aicc, r.k))
    else:
        feasible.sort(key=lambda r: (r.k, r.loss))
    infeasible.sort(key=lambda r: (worst(r), r.loss))
    return feasible + infeasible
