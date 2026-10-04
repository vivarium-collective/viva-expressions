import logging
from graphlib import TopologicalSorter

import sympy as sp
from process_bigraph import Step

from viva_expressions.expressions import parse, substitute

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

        # compile each: args = inputs + upstream outputs
        self.fns = {}
        for out in self.order:
            args = sorted(s.name for s in self.exprs[out].free_symbols
                          if s.name not in self.params)
            sub = substitute(self.exprs[out], self.params)
            self.fns[out] = (args, sp.lambdify(args, sub, modules=cfg["functions"]))

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
        vals, result = dict(state), {}
        for out in self.order:
            args, fn = self.fns[out]
            result[out] = vals[out] = fn(*[vals[a] for a in args])
        return result
