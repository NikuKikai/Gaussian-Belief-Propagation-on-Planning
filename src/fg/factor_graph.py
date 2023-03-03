import numpy as np
from typing import List, Dict
import itertools

from .graph import Node, Edge, Graph
from .gaussian import Gaussian


class VNode(Node):
    def __init__(self, name: str, dims: list, belief: Gaussian = None) -> None:
        super().__init__(name, dims)
        self._belief: Gaussian = belief
        if belief is not None:
            assert belief._dims == dims
        else:
            self._belief = Gaussian.identity(dims)

    @property
    def belief(self) -> Gaussian:
        return self._belief.copy()

    @property
    def mean(self) -> np.ndarray:
        return self._belief.mean

    def update_belief(self) -> Gaussian:
        edges = self.edges

        # Product over all incoming messages
        belief = None
        for e in edges:
            msg = e.get_message_to(self)
            if msg is None:
                continue
            belief = msg if belief is None else belief * msg
        self._belief = belief
        return belief

    def calc_msg(self, edge: Edge):
        """Return message of the sum-product algorithm."""
        msg = None
        for e in self.edges:
            if e is edge:
                continue
            _msg = e.get_message_to(self)
            if _msg is None:
                continue
            msg = _msg * msg
        if msg is None:
            msg = Gaussian.identity(self.dims)
        return msg

    def propagate(self):
        for e in self.edges:
            msg = self.calc_msg(e)
            if e is not None:
                e.set_message_from(self, msg)


class FNode(Node):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None) -> None:
        dims = list(itertools.chain(*[v.dims for v in vnodes]))
        super().__init__(name, dims)
        self._vnodes = vnodes
        self._factor: Gaussian = Gaussian.identity(dims) if factor is None else factor
        assert dims == self._factor._dims

    def update_factor(self):
        pass

    def calc_msg(self, edge: Edge) -> Gaussian:
        msg = self._factor.copy()

        for e in self.edges:
            if e is edge:
                continue
            _msg = e.get_message_to(self)
            if _msg is None:
                continue
            msg = _msg * msg

        # Marginalize
        dims = []
        for e in self.edges:
            if e is edge:
                continue
            o = e.get_other(self)
            dims.extend(o._dims)
        msg = msg.marginalize(dims)
        return msg

    def propagate(self):
        for e in self.edges:
            msg = self.calc_msg(e)
            if e is not None:
                e.set_message_from(self, msg)


class FactorGraph(Graph):
    def __init__(self) -> None:
        super().__init__()

    def get_vnodes(self) -> List[VNode]:
        return [n for n in self._nodes if isinstance(n, VNode)]

    def get_fnodes(self) -> List[FNode]:
        return [n for n in self._nodes if isinstance(n, FNode)]

    def loopy_propagate(self, steps: int = 1) -> Dict[VNode, Gaussian]:
        vnodes = self.get_vnodes()
        fnodes = self.get_fnodes()

        beliefs = {}

        for istep in range(steps):
            # Var -> Factor
            for v in vnodes:
                v.propagate()

            # Factor Update
            for f in fnodes:
                f.update_factor()

            # Factor -> Var
            for f in fnodes:
                f.propagate()

            # Var Belief Update
            for v in vnodes:
                beliefs[v] = v.update_belief()
        return beliefs


