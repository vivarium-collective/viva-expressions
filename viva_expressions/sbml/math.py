"""Translate between libsbml MathML ASTs and sympy expressions.

Import walks the libsbml ``ASTNode`` tree directly (no formula strings), so
every construct is mapped explicitly and anything unmapped raises
``UnsupportedSBML``. Export walks the sympy tree back into ``ASTNode``\\ s.

``to_text`` prints an expression in the grammar ``viva_expressions.expressions``
parses: floats are printed with ``repr`` (the shortest string that round-trips
the IEEE double), so an imported constant keeps its exact value.
"""
import math
from collections.abc import Callable

import libsbml
import sympy as sp
from sympy.printing.str import StrPrinter

AVOGADRO = 6.02214179e23   # the value SBML Level 3 fixes for the avogadro csymbol


class UnsupportedSBML(ValueError):
    """The model uses an SBML construct with no faithful OdeProcess mapping."""


_UNARY = {
    libsbml.AST_FUNCTION_ABS: sp.Abs,
    libsbml.AST_FUNCTION_EXP: sp.exp,
    libsbml.AST_FUNCTION_LN: sp.log,
    libsbml.AST_FUNCTION_FLOOR: sp.floor,
    libsbml.AST_FUNCTION_CEILING: sp.ceiling,
    libsbml.AST_FUNCTION_FACTORIAL: sp.factorial,
    libsbml.AST_FUNCTION_SIN: sp.sin,
    libsbml.AST_FUNCTION_COS: sp.cos,
    libsbml.AST_FUNCTION_TAN: sp.tan,
    libsbml.AST_FUNCTION_SEC: sp.sec,
    libsbml.AST_FUNCTION_CSC: sp.csc,
    libsbml.AST_FUNCTION_COT: sp.cot,
    libsbml.AST_FUNCTION_SINH: sp.sinh,
    libsbml.AST_FUNCTION_COSH: sp.cosh,
    libsbml.AST_FUNCTION_TANH: sp.tanh,
    libsbml.AST_FUNCTION_SECH: sp.sech,
    libsbml.AST_FUNCTION_CSCH: sp.csch,
    libsbml.AST_FUNCTION_COTH: sp.coth,
    libsbml.AST_FUNCTION_ARCSIN: sp.asin,
    libsbml.AST_FUNCTION_ARCCOS: sp.acos,
    libsbml.AST_FUNCTION_ARCTAN: sp.atan,
    libsbml.AST_FUNCTION_ARCSEC: sp.asec,
    libsbml.AST_FUNCTION_ARCCSC: sp.acsc,
    libsbml.AST_FUNCTION_ARCCOT: sp.acot,
    libsbml.AST_FUNCTION_ARCSINH: sp.asinh,
    libsbml.AST_FUNCTION_ARCCOSH: sp.acosh,
    libsbml.AST_FUNCTION_ARCTANH: sp.atanh,
    libsbml.AST_FUNCTION_ARCSECH: sp.asech,
    libsbml.AST_FUNCTION_ARCCSCH: sp.acsch,
    libsbml.AST_FUNCTION_ARCCOTH: sp.acoth,
    libsbml.AST_LOGICAL_NOT: sp.Not,
}
_RELATIONAL = {
    libsbml.AST_RELATIONAL_EQ: sp.Eq,
    libsbml.AST_RELATIONAL_NEQ: sp.Ne,
    libsbml.AST_RELATIONAL_GT: sp.StrictGreaterThan,
    libsbml.AST_RELATIONAL_GEQ: sp.GreaterThan,
    libsbml.AST_RELATIONAL_LT: sp.StrictLessThan,
    libsbml.AST_RELATIONAL_LEQ: sp.LessThan,
}
_NARY = {
    libsbml.AST_LOGICAL_AND: sp.And,
    libsbml.AST_LOGICAL_OR: sp.Or,
    libsbml.AST_LOGICAL_XOR: sp.Xor,
    libsbml.AST_FUNCTION_MIN: sp.Min,
    libsbml.AST_FUNCTION_MAX: sp.Max,
}
_REJECTED = {
    libsbml.AST_FUNCTION_DELAY: "delay (a delay differential equation)",
    libsbml.AST_FUNCTION_RATE_OF: "rateOf",
    libsbml.AST_FUNCTION: "an unexpanded user-defined function call",
    libsbml.AST_LAMBDA: "a lambda outside a function definition",
}


def from_ast(node: libsbml.ASTNode, symbol: Callable[[str], sp.Basic],
             time: Callable[[], sp.Basic]) -> sp.Basic:
    """Convert a libsbml AST to sympy.

    ``symbol(sbml_id)`` returns the sympy object an identifier stands for, and
    ``time()`` the object for the ``time`` csymbol; the caller decides both, so
    renames and the autonomized time variable stay out of this module.
    """
    t = node.getType()
    kids = [from_ast(node.getChild(i), symbol, time)
            for i in range(node.getNumChildren())]
    if t in _REJECTED:
        raise UnsupportedSBML(f"unsupported math: {_REJECTED[t]} "
                              f"({libsbml.formulaToL3String(node)!r})")
    if t == libsbml.AST_INTEGER:
        return sp.Integer(node.getInteger())
    if t == libsbml.AST_RATIONAL:
        return sp.Rational(node.getNumerator(), node.getDenominator())
    if t in (libsbml.AST_REAL, libsbml.AST_REAL_E):
        if not math.isfinite(node.getReal()):
            raise UnsupportedSBML(f"non-finite number {node.getReal()} in math")
        return sp.Float(node.getReal())
    if t == libsbml.AST_NAME:
        return symbol(node.getName())
    if t == libsbml.AST_NAME_TIME:
        return time()
    if t == libsbml.AST_NAME_AVOGADRO:
        return sp.Float(AVOGADRO)
    if t == libsbml.AST_CONSTANT_PI:
        return sp.pi
    if t == libsbml.AST_CONSTANT_E:
        return sp.E
    if t == libsbml.AST_CONSTANT_TRUE:
        return sp.true
    if t == libsbml.AST_CONSTANT_FALSE:
        return sp.false
    if t == libsbml.AST_PLUS:
        return sp.Add(*kids)
    if t == libsbml.AST_TIMES:
        return sp.Mul(*kids)
    if t == libsbml.AST_MINUS:
        return -kids[0] if len(kids) == 1 else kids[0] - kids[1]
    if t == libsbml.AST_DIVIDE:
        return kids[0] / kids[1]
    if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        return kids[0] ** kids[1]
    if t == libsbml.AST_FUNCTION_ROOT:          # libsbml: [degree, radicand]
        return kids[1] ** (sp.Integer(1) / kids[0])
    if t == libsbml.AST_FUNCTION_LOG:           # libsbml: [base, argument]
        return sp.log(kids[1], kids[0])
    if t == libsbml.AST_FUNCTION_PIECEWISE:     # [value, cond]* [otherwise]
        pieces = [(kids[i], kids[i + 1]) for i in range(0, len(kids) - 1, 2)]
        if len(kids) % 2:
            pieces.append((kids[-1], sp.true))
        else:
            raise UnsupportedSBML(
                "piecewise without <otherwise> is undefined where no piece "
                f"applies ({libsbml.formulaToL3String(node)!r})")
        return sp.Piecewise(*pieces)
    if t in _UNARY:
        return _UNARY[t](kids[0])
    if t in _RELATIONAL:
        if len(kids) != 2:
            raise UnsupportedSBML(
                f"n-ary relational {libsbml.formulaToL3String(node)!r}")
        return _RELATIONAL[t](kids[0], kids[1])
    if t in _NARY:
        return _NARY[t](*kids)
    raise UnsupportedSBML(
        f"unsupported math node {node.getName() or t!r} "
        f"in {libsbml.formulaToL3String(node)!r}")


_TO_UNARY = {f: t for t, f in _UNARY.items() if f is not sp.log}
_TO_RELATIONAL = {f: t for t, f in _RELATIONAL.items()}
_TO_NARY = {f: t for t, f in _NARY.items()}


def _node(type_, *children: libsbml.ASTNode) -> libsbml.ASTNode:
    n = libsbml.ASTNode(type_)
    for c in children:
        n.addChild(c)
    return n


def to_ast(expr: sp.Basic, time: str | None = None) -> libsbml.ASTNode:
    """Convert a sympy expression to a libsbml AST (MathML content).

    A symbol named ``time`` becomes SBML's ``time`` csymbol.
    """
    if expr.is_Symbol and time is not None and expr.name == time:
        n = libsbml.ASTNode(libsbml.AST_NAME_TIME)
        n.setName("time")
        return n
    if expr.is_Symbol:
        n = libsbml.ASTNode(libsbml.AST_NAME)
        n.setName(expr.name)
        return n
    if expr is sp.pi:
        return libsbml.ASTNode(libsbml.AST_CONSTANT_PI)
    if expr is sp.E:
        return libsbml.ASTNode(libsbml.AST_CONSTANT_E)
    if expr is sp.true or expr is sp.false:
        return libsbml.ASTNode(
            libsbml.AST_CONSTANT_TRUE if expr is sp.true else libsbml.AST_CONSTANT_FALSE)
    if expr.is_Integer:
        n = libsbml.ASTNode(libsbml.AST_INTEGER)
        n.setValue(int(expr))
        return n
    if expr.is_Rational:
        n = libsbml.ASTNode(libsbml.AST_RATIONAL)
        n.setValue(int(expr.p), int(expr.q))
        return n
    if expr.is_Float:
        n = libsbml.ASTNode(libsbml.AST_REAL)
        n.setValue(float(expr))
        return n
    def sub(e):
        return to_ast(e, time)
    if expr.is_Add:
        return _node(libsbml.AST_PLUS, *map(sub, expr.args))
    if expr.is_Mul:
        return _node(libsbml.AST_TIMES, *map(sub, expr.args))
    if expr.is_Pow:
        return _node(libsbml.AST_POWER, sub(expr.base), sub(expr.exp))
    if isinstance(expr, sp.exp):
        return _node(libsbml.AST_FUNCTION_EXP, sub(expr.args[0]))
    if isinstance(expr, sp.log):
        if len(expr.args) != 1:
            raise UnsupportedSBML(f"log with an explicit base: {expr}")
        return _node(libsbml.AST_FUNCTION_LN, sub(expr.args[0]))
    if isinstance(expr, sp.Piecewise):
        n = libsbml.ASTNode(libsbml.AST_FUNCTION_PIECEWISE)
        for value, cond in expr.args:
            n.addChild(sub(value))
            if cond is not sp.true:
                n.addChild(sub(cond))
        return n
    for table in (_TO_UNARY, _TO_RELATIONAL, _TO_NARY):
        if expr.func in table:
            return _node(table[expr.func], *map(sub, expr.args))
    raise UnsupportedSBML(f"no SBML MathML mapping for {expr.func.__name__}: {expr}")


# Names the printed text can use as functions or constants. An SBML identifier
# equal to one of these must be renamed, or the text would mean something else
# (a parameter ``pi`` read as the constant, a parameter ``sin`` used as a call).
RESERVED = frozenset(
    {f.__name__ for f in (*_UNARY.values(), *_NARY.values())}
    | {"Eq", "Ne", "Piecewise", "log", "exp", "sqrt", "pi", "True", "False"})


class _ExactPrinter(StrPrinter):
    """``str`` output that reparses to the same doubles (``repr`` floats)."""

    def _print_Float(self, expr):
        return repr(float(expr))

    def _print_Xor(self, expr):
        # StrPrinter writes ``a ^ b``, which the grammar (convert_xor) reads as a power
        return f"Xor({', '.join(self._print(a) for a in expr.args)})"

    def _print_Exp1(self, expr):
        # the grammar binds a bare ``E`` to a plain symbol, not Euler's number
        return "exp(1)"


def to_text(expr: sp.Basic) -> str:
    """Print ``expr`` in the expression grammar OdeProcess parses."""
    return _ExactPrinter().doprint(expr)
