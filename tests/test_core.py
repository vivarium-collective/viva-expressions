"""build_core() must register this package's own processes.

The scaffold's tests/test_core_registration.py only scans top-level modules, so
it skips here (OdeProcess/MathExpressionStep live in viva_expressions.processes).
This asserts the real registry, so dropping the registration in core.py fails.
"""
import pytest

from viva_expressions.core import build_core


@pytest.mark.parametrize("name", ["OdeProcess", "MathExpressionStep"])
def test_workspace_process_registered(name):
    core = build_core()
    assert name in core.link_registry
    assert core.link_registry.get(name) is not None
