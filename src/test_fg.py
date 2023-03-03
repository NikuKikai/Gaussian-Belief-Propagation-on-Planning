import numpy as np
from fg.gaussian import Gaussian
from fg.factor_graph import VNode, FNode, FactorGraph


g = FactorGraph()

v0 = VNode('v0', ['v0.0', 'v0.1'])
v1 = VNode('v1', ['v1.0', 'v1.1'])
f0 = FNode('f0', [v0], factor=Gaussian(['v0.0', 'v0.1'], [[1], [1]], np.diag([1, 1])))
f01 = FNode('f01', [v0, v1], factor=Gaussian(['v0.0', 'v0.1', 'v1.0', 'v1.1'], [[2], [1], [3], [3]], np.diag([1, 1, 1, 1])))

g.connect(f0, v0)
g.connect(v0, f01)
g.connect(f01, v1)

beliefs = g.loopy_propagate()
for v, p in beliefs.items():
    print(v, p.mean)
