"""End-to-end synthesis: candidates → fit → rank → export → Composite verification."""
from dataclasses import dataclass
from pathlib import Path

from viva_expressions.synthesize import export
from viva_expressions.synthesize.fit import FitResult, fit_model
from viva_expressions.synthesize.motifs import Model, resolve
from viva_expressions.synthesize.select import rank
from viva_expressions.synthesize.sindy import SindyUnavailable, discover
from viva_expressions.synthesize.spec import Spec, SpecError


@dataclass
class Synthesis:
    ranked: list[FitResult]
    verification: export.Verification | None
    document: dict | None
    report: dict

    @property
    def ok(self) -> bool:
        return self.verification is not None and self.verification.verified


def candidate_models(spec: Spec) -> tuple[list[Model], list[str]]:
    """Resolve the spec's candidates (any failure is a SpecError) plus SINDy's."""
    models, errors, notes = [], [], []
    for i, c in enumerate(spec.candidates):
        try:
            models.append(resolve(c, spec))
        except ValueError as e:
            errors.append(f"candidates[{i}]: {e}")
    if errors:
        raise SpecError(errors)
    if spec.data is not None:
        try:
            found = discover(spec)
            models += found
            notes.append(f"SINDy proposed {len(found)} polynomial structure(s).")
        except SindyUnavailable as e:
            notes.append(f"SINDy skipped: {e}")
    return models, notes


def synthesize(spec: Spec, out_dir: str | Path | None = None,
               interval: float | None = None, top: int | None = None) -> Synthesis:
    models, notes = candidate_models(spec)
    if not models:
        raise SpecError(["candidates: none to fit (no candidates and SINDy unavailable)"])
    ranked = rank([fit_model(m, spec) for m in models])
    winner = ranked[0] if ranked[0].feasible else None
    document = verification = None
    if winner is not None:
        document = export.to_document(winner, spec, interval)
        verification = export.verify(winner, spec, document)
    rep = export.report(spec, ranked, verification, notes)
    if out_dir is not None:
        export.write(out_dir, rep, document, top)
    return Synthesis(ranked, verification, document, rep)
