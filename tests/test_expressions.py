import numpy as np
import pytest
import sympy as sp

from viva_expressions.expressions import compile_rhs, identifiers, parse, substitute


@pytest.mark.parametrize("name", ["N", "gamma", "beta", "E", "I", "S", "Q"])
def test_model_names_parse_as_plain_symbols(name):
    # regression: bare sp.sympify("N*x") / sp.sympify("gamma*x") raise TypeError
    expr = parse(f"{name}*x")
    assert expr == sp.Symbol(name) * sp.Symbol("x")
    assert substitute(expr, {name: 3.0, "x": 2.0}) == 6.0


def test_called_names_stay_functions_and_pi_stays_constant():
    assert identifiers("a*sin(b) + exp(c)") == {"a", "b", "c"}
    assert parse("sin(pi/2)") == 1
    assert parse("x^2") == sp.Symbol("x") ** 2
    # declaring pi makes it an ordinary symbol
    assert parse("pi*x", allowed=["pi", "x"]).free_symbols == {sp.Symbol("pi"), sp.Symbol("x")}


def test_undeclared_symbol_raises():
    with pytest.raises(ValueError, match=r"undeclared symbol\(s\) \['k'\]"):
        parse("k*x", allowed=["x"])


def test_compile_rhs_logistic_matches_numpy():
    rhs = compile_rhs({"N": "r*N*(1 - N/K)"}, state=["N"], params=["r", "K"])
    for n in [0.1, 2.5, 4.9, 7.0]:
        np.testing.assert_allclose(
            rhs.f(np.array([n]), np.array([]), np.array([0.8, 5.0])),
            [0.8 * n * (1 - n / 5.0)])
        np.testing.assert_allclose(
            rhs.jac(np.array([n]), np.array([]), np.array([0.8, 5.0])),
            [[0.8 - 2 * 0.8 * n / 5.0]])


def test_compile_rhs_lotka_volterra_with_input():
    rhs = compile_rhs(
        {"x": "a*x - b*x*y + u", "y": "-c*y + d*x*y"},
        state=["x", "y"], inputs=["u"], params=["a", "b", "c", "d"])
    y, u, p = np.array([2.0, 1.5]), np.array([0.3]), np.array([1.0, 0.5, 1.0, 0.25])
    x_, y_ = y
    np.testing.assert_allclose(
        rhs.f(y, u, p), [x_ - 0.5 * x_ * y_ + 0.3, -y_ + 0.25 * x_ * y_])
    np.testing.assert_allclose(
        rhs.jac(y, u, p), [[1 - 0.5 * y_, -0.5 * x_], [0.25 * y_, -1 + 0.25 * x_]])


def test_compile_rhs_constant_rhs_has_correct_shape():
    rhs = compile_rhs({"x": "k", "y": "0"}, state=["x", "y"], params=["k"])
    np.testing.assert_allclose(rhs.f(np.zeros(2), np.array([]), np.array([2.0])), [2.0, 0.0])
    assert rhs.jac(np.zeros(2), np.array([]), np.array([2.0])).shape == (2, 2)


def test_compile_rhs_rejects_mismatched_and_duplicate_names():
    with pytest.raises(ValueError, match="missing"):
        compile_rhs({"x": "1"}, state=["x", "y"])
    with pytest.raises(ValueError, match="more than once"):
        compile_rhs({"x": "k"}, state=["x"], params=["x", "k"])
