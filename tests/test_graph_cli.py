"""python -m viva_expressions.graph: to-graph, from-graph, check; exit codes 0/1/2."""
import json

import pytest
import yaml
from graph_documents import COMPOSITES, DATA, workspace_spec

from viva_expressions.graph.__main__ import main, read_document
from viva_expressions.graph.document import document_equal, dumps_graph, loads_graph

LV = COMPOSITES / "lotka_volterra.composite.yaml"
OSC = DATA / "oscillator.composite.json"


@pytest.mark.parametrize("source, suffix", [(LV, ".yaml"), (OSC, ".json")])
def test_to_graph_then_from_graph_gives_the_document_back(tmp_path, source, suffix):
    """Falsifies: the CLI's file round trip (graph JSON, then .yaml/.json) is lossless."""
    graph, back = tmp_path / "g.json", tmp_path / f"back{suffix}"
    assert main(["to-graph", str(source), "--out", str(graph)]) == 0
    assert loads_graph(graph.read_text(encoding="utf-8")).graph["views"] == [
        "placeholders", "emitters", "bridges", "expressions"]
    assert main(["from-graph", str(graph), "--out", str(back)]) == 0
    assert document_equal(read_document(back), read_document(source))


@pytest.mark.parametrize("source", [LV, OSC])
def test_check_reruns_bit_identically(source, capsys):
    """Falsifies: `check --run` passes a document whose round trip re-runs differently."""
    assert main(["check", str(source), "--run", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "lossless in memory: True; through JSON: True" in out
    assert "bit-identical: True" in out


def test_check_run_loads_both_sides_from_a_temporary_copy(monkeypatch, capsys):
    """Falsifies: `check --run` loads the original spec from where it lives
    while the round trip loads from a temporary directory."""
    from process_bigraph.composite_spec import CompositeSpec
    loaded, real = [], CompositeSpec.from_file.__func__
    monkeypatch.setattr(CompositeSpec, "from_file",
                        classmethod(lambda cls, path: loaded.append(path) or real(cls, path)))
    assert main(["check", str(LV), "--run", "0.5"]) == 0
    assert len(loaded) == 2
    assert all(p.name == LV.name and p.parent != LV.parent for p in loaded)


def test_check_exits_2_when_the_json_path_loses_something(tmp_path, capsys):
    """Falsifies: a document the JSON carrier cannot hold (an integer key inside a
    value, which JSON turns into a string) is reported lossless."""
    spec = workspace_spec(LV)
    spec["state"]["predation"]["config"]["tags"] = {1: "one"}
    path = tmp_path / "intkey.composite.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    assert main(["check", str(path)]) == 2
    assert "lossless in memory: True; through JSON: False" in capsys.readouterr().out


def test_invalid_inputs_exit_1(tmp_path, capsys):
    """Falsifies: a missing file, an unknown suffix, a tampered graph or a stale
    view is accepted instead of exiting 1."""
    assert main(["check", str(tmp_path / "missing.json")]) == 1
    odd = tmp_path / "doc.txt"
    odd.write_text("{}", encoding="utf-8")
    assert main(["check", str(odd)]) == 1

    graph = tmp_path / "g.json"
    assert main(["to-graph", str(LV), "--out", str(graph)]) == 0
    data = json.loads(graph.read_text(encoding="utf-8"))
    node = next(n for n in data["nodes"] if n["id"] == "/state/stores/x")
    node["value"] = "${y0}"                              # the document changed; views did not
    graph.write_text(json.dumps(data), encoding="utf-8")
    assert main(["from-graph", str(graph), "--out", str(tmp_path / "x.yaml")]) == 1
    assert "stale" in capsys.readouterr().err

    G = loads_graph(graph.read_text(encoding="utf-8"))
    G.nodes["/state/stores"]["kind"] = "link"
    graph.write_text(dumps_graph(G), encoding="utf-8")
    assert main(["from-graph", str(graph), "--out", str(tmp_path / "x.yaml")]) == 1
    assert "differs from to_graph's" in capsys.readouterr().err


def test_forged_derived_elements_and_cycles_exit_1(tmp_path, capsys):
    """Falsifies: from-graph accepts derived elements no applied view produced, or a
    cyclic YAML document escapes the exit-code contract with a traceback."""
    graph = tmp_path / "g.json"
    assert main(["to-graph", str(LV), "--out", str(graph), "--bare"]) == 0
    G = loads_graph(graph.read_text(encoding="utf-8"))
    G.add_node("forged", view="expressions")
    graph.write_text(dumps_graph(G), encoding="utf-8")
    assert main(["from-graph", str(graph), "--out", str(tmp_path / "x.yaml")]) == 1
    assert "stale" in capsys.readouterr().err

    cyclic = tmp_path / "cyclic.yaml"
    cyclic.write_text("a: &x\n  b: *x\n", encoding="utf-8")
    assert main(["check", str(cyclic)]) == 1
    assert "cyclic" in capsys.readouterr().err


@pytest.fixture
def ascii_locale():
    """The C locale: text I/O that relies on the locale default becomes ASCII
    (see tests/test_cli.py: native libraries call setlocale mid-session)."""
    import locale
    previous = locale.setlocale(locale.LC_CTYPE)
    locale.setlocale(locale.LC_CTYPE, "C")
    try:
        yield
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)


def test_files_are_utf8_under_an_ascii_locale(tmp_path, ascii_locale):
    """Falsifies: the CLI reads and writes non-ASCII documents without naming utf-8."""
    spec = workspace_spec(LV)
    spec["description"] = "Lotka–Volterra — prédateur/proie ✓"
    spec["state"]["stores"]["ñ"] = 1.0
    source = tmp_path / "unicode.composite.yaml"
    source.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    graph, back = tmp_path / "g.json", tmp_path / "back.yaml"
    assert main(["to-graph", str(source), "--out", str(graph)]) == 0
    assert main(["from-graph", str(graph), "--out", str(back)]) == 0
    assert document_equal(read_document(back), spec)
    assert "ñ" in back.read_text(encoding="utf-8")
