"""Synthesize process-bigraph models of dynamical systems from expected outputs and behavior."""
from viva_expressions.synthesize.pipeline import Synthesis, synthesize
from viva_expressions.synthesize.spec import Spec, SpecError, load_spec, parse_spec

__all__ = ["Spec", "SpecError", "Synthesis", "load_spec", "parse_spec", "synthesize"]
