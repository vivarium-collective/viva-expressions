"""Two-way conversion between SBML and viva-expressions OdeProcess models."""
from viva_expressions.composites import ode_document
from viva_expressions.sbml.math import UnsupportedSBML
from viva_expressions.sbml.read import OdeModel, read_sbml
from viva_expressions.sbml.write import ode_from_state, write_sbml

__all__ = ["OdeModel", "UnsupportedSBML", "document", "ode_from_state",
           "read_sbml", "write_sbml"]


def document(model: OdeModel, interval: float = 0.1, **solver) -> dict:
    """A runnable process-bigraph document for an imported model."""
    return ode_document(model.rhs, model.params, model.initial, interval=interval,
                        assignments=model.assignments, time_var=model.time_var, **solver)
