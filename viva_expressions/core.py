"""build_core(core=None) — the workspace core.

Cross-repo convention: compose the cores of imported repos (inheriting
their registered processes/types), then register THIS repo's own types,
processes, and composites — so a downstream importer gets everything by
calling this build_core.
"""
from process_bigraph import allocate_core
from viva_superpowers.core_compose import register_package_processes


def build_core(core=None):
    core = core if core is not None else allocate_core()
    # Register THIS repo's own process/step classes (OdeProcess,
    # MathExpressionStep) as first-class, browsable Registry entries; a
    # composite that instantiates a process directly does NOT auto-register
    # it by name, and an editable install is invisible to allocate_core().
    register_package_processes(core, "viva_expressions.processes")
    return core
