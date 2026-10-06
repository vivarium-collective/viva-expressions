"""Matrix projections (viva_expressions.graph.matrices) on real document graphs.

Oracles are independent of the projection: edge counts come from the
networkx edge list, connected components from a union-find written here.
"""
import numpy as np
import pytest
from graph_documents import REAL

from viva_expressions.graph.document import to_graph
from viva_expressions.graph.matrices import to_matrices

KINDS = [("wire",), ("contains",), ("contains", "wire")]
CASES = [(name, kinds) for name in REAL for kinds in KINDS]


def selected_edges(G, kinds):
    return [(u, v) for u, v, k, d in G.edges(keys=True, data=True)
            if k in kinds and "view" not in d]


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges:
        parent[find(u)] = find(v)
    return len({find(n) for n in nodes})


@pytest.mark.parametrize("name, kinds", CASES)
def test_adjacency_counts_every_selected_edge(name, kinds):
    """Falsifies: A.sum() equals the number of selected edges (parallel edges sum)."""
    G = to_graph(REAL[name])
    m = to_matrices(G, kinds)
    assert m.A.sum() == len(selected_edges(G, kinds)) > 0
    assert m.index == list(G.nodes)


@pytest.mark.parametrize("name, kinds", CASES)
@pytest.mark.parametrize("directed", [True, False])
def test_laplacian_rows_sum_to_zero(name, kinds, directed):
    """Falsifies: every row of L = D - A sums to 0 (D is the right degree)."""
    L = to_matrices(to_graph(REAL[name]), kinds, directed).L
    assert not np.any(L.sum(axis=1))


@pytest.mark.parametrize("name, kinds", CASES)
def test_undirected_laplacian_is_symmetric_psd_with_one_zero_per_component(name, kinds):
    """Falsifies: the undirected L is symmetric positive semidefinite and its
    number of zero eigenvalues is the number of connected components
    (counted independently by union-find over the edge list)."""
    G = to_graph(REAL[name])
    m = to_matrices(G, kinds, directed=False)
    L = m.L.toarray().astype(float)
    assert np.array_equal(L, L.T)
    eig = np.linalg.eigvalsh(L)
    assert eig.min() > -1e-9
    assert int(np.sum(np.abs(eig) < 1e-9)) == components(m.index, selected_edges(G, kinds))


def test_unknown_edge_kind_is_refused():
    """Falsifies: a kind that is not a core edge kind is silently projected as empty."""
    with pytest.raises(ValueError, match="unknown edge kind"):
        to_matrices(to_graph(REAL["oscillator"]), ("crosses",))
