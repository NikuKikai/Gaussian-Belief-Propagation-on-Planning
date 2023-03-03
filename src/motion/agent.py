from typing import Tuple, List, Dict
import numpy as np
from fg.gaussian import Gaussian
from fg.factor_graph import VNode, FNode, FactorGraph

from .obstacle import ObstacleMap
from .nodes import DynaFNode, ObstacleFNode, DistFNode, RemoteVNode


class Agent:
    def __init__(self, name: str, state, target = None, steps: int = 8, radius: int = 5, omap: ObstacleMap = None, env: 'Env' = None) -> None:
        assert steps > 1
        if np.shape(state) == ():
            state = np.array([[state]])
        elif len(np.shape(state)) == 1:
            state = np.array(state)[:, None]
        if np.shape(target) == ():
            target = np.array([[target]])
        elif len(np.shape(target)) == 1:
            target = np.array(target)[:, None]

        self._steps = steps
        self._name = name
        self._state = np.array(state)
        self._omap = omap
        self._radius = radius
        self._env = env

        self._vnodes = [VNode(f'v{i}', [f'v{i}.x', f'v{i}.y', f'v{i}.vx', f'v{i}.vy']) for i in range(steps)]
        self._fnode_start = FNode('fstart', [self._vnodes[0]])
        self._fnode_end = FNode('fend', [self._vnodes[-1]])
        self.set_state(state)
        self.set_target(target)

        self._fnodes_dyna = [DynaFNode(f'fd{i}{i+1}', [self._vnodes[i], self._vnodes[i+1]]) for i in range(steps-1)]
        self._fnodes_obst = [ObstacleFNode(f'fo{i}', [self._vnodes[i]], omap=omap) for i in range(1, steps)]

        self._graph = FactorGraph()
        self._graph.connect(self._vnodes[0], self._fnode_start)
        for v, f in zip(self._vnodes[:-1], self._fnodes_dyna):
            self._graph.connect(v, f)
        for v, f in zip(self._vnodes[1:], self._fnodes_dyna):
            self._graph.connect(v, f)
        for v, f in zip(self._vnodes[1:], self._fnodes_obst):
            self._graph.connect(v, f)
        self._graph.connect(self._vnodes[-1], self._fnode_end)

        self._beliefs = None
        self._others = {}

    def __str__(self) -> str:
        return f'({self._name} s={self._state})'

    @property
    def x(self) -> float:
        return self._state[0, 0]
    @property
    def y(self) -> float:
        return self._state[1, 0]

    def step_all_async(self):
        # Search near agents
        others = self._env.find_near(self)
        for o in others:
            self.setup_com(o)
        for on in list(self._others.keys()):
            if self._others[on] not in others:
                self.end_com(o)

        # Propagate
        for i in range(self._steps * 3):
            for o in self._others:
                self.send(o)
            beliefs = self._graph.loopy_propagate()

        v1 = beliefs[self._vnodes[1]].mean
        self.set_state(v1)
        self._beliefs = beliefs

    def step_connect(self):
        # Search near agents
        others = self._env.find_near(self)
        for o in others:
            self.setup_com(o)
        for on in list(self._others.keys()):
            if self._others[on] not in others:
                self.end_com(o)

    def step_com(self):
        for o in self._others:
            self.send(o)

    def step_propagate(self):
        self._beliefs = self._graph.loopy_propagate()

    def step_move(self):
        v1 = self._beliefs[self._vnodes[1]].mean
        self.set_state(v1)

    def set_state(self, state):
        self._state = np.array(state)
        v0 = self._vnodes[0]
        self._fnode_start._factor = Gaussian(v0.dims, state, np.diag([0.001]*4))
        v0._belief = Gaussian(v0.dims, state, np.diag([0.001]*4))

        # Init position for each vnode, to avoid 0 distance in DistFNode
        dt = 0.1  # NOTE
        for i in range(1, self._steps):
            s = state + np.random.rand(4, 1) * 0.01
            s[:2] += s[2:] * dt
            v = self._vnodes[i]
            v._belief = Gaussian(v.dims, s, np.diag([0.001]*4))

    def set_target(self, target):
        if target is not None:
            self._fnode_end._factor = Gaussian(self._vnodes[-1].dims, target, np.diag([0.05]*4))
        else:
            self._fnode_end._factor = Gaussian.identity(self._vnodes[-1].dims)

    def push_msg(self, msg):
        _type, aname, vname, p = msg
        if p is None:
            return
        if aname not in self._others:
            print(f'push msg: name {aname} not found')
            return
        vnodes = self._others[aname]['v']
        vnode: RemoteVNode = None
        for v in vnodes:
            if v._name == vname:
                vnode = v
                break
        if vnode is None:
            print('vname not found')
            return

        # print(self._name, _type, vname, p._dims, vnode._dims, vnode)

        p: Gaussian
        p._dims = vnode.dims
        if _type == 'belief':
            vnode._belief = p
        if _type == 'f2v': # TODO FIXME dims
            e = vnode.edges[0]
            e.set_message_from(e.get_other(vnode), p)
        if _type == 'v2f': # TODO FIXME dims
            e = vnode.edges[0]
            vnode._msgs[e] = p
            # e.set_message_from(vnode, p)


    def setup_com(self, other: 'Agent'):
        on = other._name
        if on in self._others:
            return

        vnodes = [RemoteVNode(f'{on}.v{i}', [f'{on}.v{i}.x', f'{on}.v{i}.y', f'{on}.v{i}.vx', f'{on}.v{i}.vy']) for i in range(1, self._steps)]
        fnodes = [DistFNode(f'{on}.f{i}', [vnodes[i-1], self._vnodes[i]],) for i in range(1, self._steps)]
        for i in range(1, self._steps):
            self._graph.connect(self._vnodes[i], fnodes[i-1])
            self._graph.connect(vnodes[i-1], fnodes[i-1])
        self._others[on] = {'a': other, 'v': vnodes, 'f': fnodes}

        other.setup_com(self)

        self.send(on)

    def send(self, name: str):
        other = self._others[name]['a']
        for i in range(1, self._steps):
            vname = f'{self._name}.v{i}'
            v = self._vnodes[i]
            f: FNode = self._others[name]['f'][i-1]

            belief = v.belief.copy()
            other.push_msg(('belief', self._name, vname, belief))

            f2v = f.edges[0].get_message_to(v)
            if f2v is not None:
                f2v = f2v.copy()
            other.push_msg(('f2v', self._name, vname, f2v))

            v2f = f.edges[0].get_message_to(f)
            if v2f is not None:
                v2f = v2f.copy()
            other.push_msg(('v2f', self._name, vname, v2f))

    def end_com(self, name: str):
        if name not in self._others:
            return

        vnodes = self._others[name]['v']
        fnodes = self._others[name]['f']
        for v in vnodes:
            self._graph.remove_node(v)
        for f in fnodes:
            self._graph.remove_node(f)
        other_dict = self._others.pop(name)
        other_dict['a'].end_com(self._name)


class Env:
    def __init__(self) -> None:
        self._agents: List[Agent] = []

    def add_agent(self, a: Agent):
        if a not in self._agents:
            a._env = self
            self._agents.append(a)

    def find_near(self, this: Agent, range: float = 1000, max_num: int = -1) -> List[Agent]:
        agent_ds = []
        for a in self._agents:
            if a is this:
                continue
            d = np.sqrt((a.x-this.x)**2 + (a.y-this.y)**2)
            if d < range:
                agent_ds.append((a, d))
        agent_ds.sort(key=lambda ad: ad[1])
        if max_num > 0:
            agent_ds = agent_ds[:max_num]
        return [a for a, d in agent_ds]

    def step(self, iters = 12):
        for a in self._agents:
            a.step_connect()
        for i in range(iters):
            for a in self._agents:
                a.step_com()
            for a in self._agents:
                a.step_propagate()
        for a in self._agents:
            a.step_move()