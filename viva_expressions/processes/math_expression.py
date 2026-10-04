import logging
from graphlib import TopologicalSorter

import sympy as sp
from process_bigraph import Step

from viva_expressions.expressions import parse

logger = logging.getLogger(__name__)


class MathExpressionStep(Step):
    config_schema = {
        "expressions": "list[map[string]]",   # [{"out": "z", "expr": "a*b + sin(c)"}]
        "functions":   {"_type": "string", "_default": "numpy"},  # lambdify modules
        "params":      "map[float]",           # named constants, not ports
        "input_types": "map[string]",          # optional override, default "float"
        "output_types":"map[string]",          # optional override, default "float"
        "debug":       {"_type": "boolean", "_default": False},
    }

    def initialize(self, config):
        cfg = self.config
        self.params = cfg.get("params", {})
        self.exprs = {}                 # out -> sympy expr
        for e in cfg["expressions"]:
            self.exprs[e["out"]] = parse(e["expr"])

        outs = set(self.exprs)
        param_syms = set(self.params)
        self.deps, needs = {}, set()
        for out, ex in self.exprs.items():
            syms = {s.name for s in ex.free_symbols}
            self.deps[out] = syms & outs
            needs |= syms - outs - param_syms
        self.input_names = sorted(needs)

        # topological order (graphlib.TopologicalSorter); raise on cycles
        self.order = list(TopologicalSorter(self.deps).static_order())

        # compile each: args = inputs + upstream outputs, then the parameters
        # it uses, passed as values. Substituting them into the expression
        # instead lets sympy fold constants past float range (e.g.
        # exp(-80*(t - ts)) at ts = 30 becomes 1e1042*exp(-80*t) -> inf*0 = nan).
        self.fns = {}
        for out in self.order:
            names = sorted(s.name for s in self.exprs[out].free_symbols)
            args = [n for n in names if n not in self.params]
            used = [n for n in names if n in self.params]
            fn = sp.lambdify(args + used, self.exprs[out], modules=cfg["functions"])
            self.fns[out] = (args, fn, [float(self.params[n]) for n in used])

        level = logging.INFO if cfg["debug"] else logging.DEBUG
        logger.log(level, "inputs: %s", self.input_names)
        logger.log(level, "params: %s", self.params)
        for out in self.order:
            logger.log(level, "%s = %s  (args: %s)",
                       out, self.exprs[out], self.fns[out][0])

    def inputs(self):
        t = self.config.get("input_types", {})
        return {n: t.get(n, "float") for n in self.input_names}

    def outputs(self):
        t = self.config.get("output_types", {})
        return {o: f"overwrite[{t.get(o, 'float')}]" for o in self.exprs}

    def update(self, state):
        # A declared float output is returned as a Python float: lambdified
        # Piecewise (numpy.select) yields a 0-d ndarray, which serializing
        # emitters (e.g. JSONEmitter) reject.
        types = self.config.get("output_types", {})
        vals, result = dict(state), {}
        for out in self.order:
            args, fn, values = self.fns[out]
            value = fn(*[vals[a] for a in args], *values)
            if types.get(out, "float") == "float":
                value = float(value)
            result[out] = vals[out] = value
        return result
