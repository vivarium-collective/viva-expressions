"""document_equal: the exact equality the lossless contract is stated in."""
import copy
import struct

import numpy as np
import pytest
from bigraph_schema.units import units as ureg

from viva_expressions.graph.document import document_equal


def _nan(sign, payload):
    return struct.unpack("<d", struct.pack("<Q", (sign << 63) | (0x7FF << 52) | payload))[0]


@pytest.mark.parametrize("a, b", [
    (0.0, -0.0), (1, 1.0), (True, 1), ((1,), [1]), ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
    (np.float64(1.0), 1.0), (np.zeros(2, np.float32), np.zeros(2, np.float64)),
    (np.zeros((2, 1)), np.zeros((1, 2))), (_nan(0, 1), _nan(0, 2)),
    (ureg.Quantity(1.0, "fg"), ureg.Quantity(1.0, "g")),
])
def test_document_equal_distinguishes(a, b):
    """Falsifies: document_equal conflates values that differ in type, order, bits or units."""
    assert not document_equal(a, b)
    assert document_equal(a, copy.deepcopy(a))


def test_document_equal_accepts_bitwise_equal_nan_and_nested_structures():
    """Falsifies: document_equal rejects values that are identical bit for bit."""
    nan = _nan(1, 7)
    doc = {"a": [nan, (1, "x")], "b": {"c": np.array([nan, 1.0]), "d": ureg.Quantity(3, "count")}}
    assert document_equal(doc, copy.deepcopy(doc))
    assert document_equal(nan, _nan(1, 7))
