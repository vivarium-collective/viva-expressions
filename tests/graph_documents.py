"""Real documents shared by the graph tests (pytest puts tests/ on sys.path).

The ``/synthesize-system`` oscillator (``data/graph/oscillator.composite.json``)
is byte-identical to ``python -m viva_expressions.synthesize
systems/examples/oscillator.yml``; ``data/graph/v2ecoli.composites.baseline.json``
is v2ecoli's ``reports/composite-state/v2ecoli.composites.baseline.json`` at
commit 2a2d775c (1241 nodes, 732 wires).
"""
import json
from pathlib import Path

import yaml
from process_bigraph.processes.growth_division import grow_divide_agent

from viva_expressions.composites import ode_document

DATA = Path(__file__).parent / "data" / "graph"
COMPOSITES = Path(__file__).parent.parent / "viva_expressions" / "composites"
WORKSPACE = sorted(COMPOSITES.glob("*.composite.yaml"))


def workspace_spec(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def oscillator():
    return json.loads((DATA / "oscillator.composite.json").read_text(encoding="utf-8"))


def grow_divide_document():
    return {"state": {"agents": {"0": grow_divide_agent(
        config={}, state={"mass": 1.0}, path=["agents", "0"])}}}


def real_documents():
    docs = {p.name: workspace_spec(p) for p in WORKSPACE}
    docs["oscillator"] = oscillator()
    docs["ode_document(assignments, time_var)"] = ode_document(
        rhs={"S": "-k*S*Piecewise((0.0, t < t0), (1.0, True))",
             "P": "k*S*Piecewise((0.0, t < t0), (1.0, True))"},
        params={"k": 0.5, "t0": 2.0}, initial={"S": 1.0, "P": 0.0}, interval=0.25,
        assignments={"on": "Piecewise((0.0, t < t0), (1.0, True))", "total": "S + P"},
        time_var="t", method="Radau")
    docs["grow_divide_agent"] = grow_divide_document()
    docs["v2ecoli baseline"] = json.loads(
        (DATA / "v2ecoli.composites.baseline.json").read_text(encoding="utf-8"))
    return docs


REAL = real_documents()
