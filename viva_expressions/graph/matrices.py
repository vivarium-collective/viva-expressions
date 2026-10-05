"""Matrix projections of a document graph: adjacency, degree, Laplacian.

**Lossy.** A/D/L keep only how many selected edges join each ordered pair of
nodes: no slots, values, wires, edge attributes or derived (view) elements.
The lossless representation is the node/edge graph
(``viva_expressions.graph.document``); ``index`` maps each matrix row and
column back to its node id there.
"""
from typing import NamedTuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_array, diags_array

from viva_expressions.graph.document import CONTAINS, WIRE

EDGE_KINDS = (CONTAINS, WIRE)


class Matrices(NamedTuple):
    index: list[str]     # node id of each row/column
    A: csr_array         # adjacency: count of selected edges u -> v
    D: csr_array         # degree (diagonal)
    L: csr_array         # Laplacian, D - A


def to_matrices(G: nx.MultiDiGraph, kinds=(WIRE,), directed: bool = True) -> Matrices:
    """Adjacency, degree and Laplacian over the non-derived edges of ``kinds``.

    Rows and columns are every non-derived node, in graph order. ``A[i, j]``
    counts the selected edges ``index[i] -> index[j]`` (parallel edges sum).
    Directed: ``D`` is the out-degree. Undirected: ``A`` is ``A + A.T`` and
    ``D`` its degree. ``L = D - A`` either way, so every row of ``L`` sums to 0.
    """
    unknown = set(kinds) - set(EDGE_KINDS)
    if unknown:
        raise ValueError(f"unknown edge kind(s) {sorted(unknown)}; choose from {EDGE_KINDS}")
    index = [n for n, a in G.nodes(data=True) if "view" not in a]
    pos = {n: i for i, n in enumerate(index)}
    pairs = [(pos[u], pos[v]) for u, v, k, d in G.edges(keys=True, data=True)
             if k in kinds and "view" not in d]
    rows, cols = (np.array(x, dtype=np.int64) for x in zip(*pairs)) if pairs else ([], [])
    n = len(index)
    A = csr_array((np.ones(len(pairs), dtype=np.int64), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    if not directed:
        A = (A + A.T).tocsr()
    D = diags_array(np.asarray(A.sum(axis=1)).ravel(), format="csr", dtype=np.int64)
    return Matrices(index, A, D, (D - A).tocsr())
