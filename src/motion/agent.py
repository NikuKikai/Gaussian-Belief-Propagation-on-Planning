from typing import Tuple, List, Dict
import numpy as np
from fg.gaussian import Gaussian
from fg.factor_graph import VNode, FNode, FactorGraph

from .obstacle import ObstacleMap
from .nodes import DynaFNode, ObstacleFNode, DistFNode, RemoteVNode


class Agent:
    def __init__(
            self, name: str, state, target = None, steps: int = 8, radius: int = 5, omap: ObstacleMap = None, env: 'Env' = None,
            start_position_precision = 1000,
            start_velocity_precision = 100,
            target_position_precision = 1,
            target_velocity_precision = 1,
            dynamic_postion_precision = 10,
            dynamic_velocity_precision = 2,
            obstacle_precision = 100,
            distance_precision = 100,
            dt: float = 0.1
        ) -> None:
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

        self._dt = dt
        self._startFNode_pos_prec = start_position_precision
        self._startFNode_vel_prec = start_velocity_precision
        self._targetFNode_pos_prec = target_position_precision
        self._targetFNode_vel_prec = target_velocity_precision
        self._distFNode_prec = distance_precision

        # Create VNodes
        self._vnodes = [VNode(f'v{i}', [f'v{i}.x', f'v{i}.y', f'v{i}.vx', f'v{i}.vy']) for i in range(steps)]

        # Create FNode for start and target
        self._fnode_start = FNode('fstart', [self._vnodes[0]])
        self._fnode_end = FNode('fend', [self._vnodes[-1]])
        self.set_state(state)
        self.set_target(target)

        # Create DynaFNode
        self._fnodes_dyna = [DynaFNode(
            f'fd{i}{i+1}', [self._vnodes[i], self._vnodes[i+1]], dt=self._dt,
            pos_prec=dynamic_postion_precision, vel_prec=dynamic_velocity_precision
        ) for i in range(steps-1)]

        # Create ObstacleFNode
        self._fnodes_obst = [ObstacleFNode(
            f'fo{i}', [self._vnodes[i]], omap=omap, safe_dist=self.r,
            z_precision=obstacle_precision
        ) for i in range(1, steps)]

        self._graph = FactorGraph()
        self._graph.connect(self._vnodes[0], self._fnode_start)
        for v, f in zip(self._vnodes[:-1], self._fnodes_dyna):
            self._graph.connect(v, f)
        for v, f in zip(self._vnodes[1:], self._fnodes_dyna):
            self._graph.connect(v, f)
        for v, f in zip(self._vnodes[1:], self._fnodes_obst):
            self._graph.connect(v, f)
        self._graph.connect(self._vnodes[-1], self._fnode_end)

        self._others = {}

    def __str__(self) -> str:
        return f'({self._name} s={self._state})'

    @property
    def name(self) -> str:
        return self._name
    @property
    def x(self) -> float:
        '''current x'''
        return self._state[0, 0]
    @property
    def y(self) -> float:
        '''current y'''
        return self._state[1, 0]
    @property
    def r(self) -> float:
        '''radius'''
        return self._radius

    def get_state(self) -> List[np.ndarray]:
        poss = []
        for v in self._vnodes:
            if v.belief is None:
                poss.append(None)
            else:
                poss.append(v.belief.mean[:, 0])
        return poss

    def get_target(self) -> np.ndarray:
        return self._fnode_end._factor.mean

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

        self.set_state(self._vnodes[1].belief.mean)

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
        self._graph.loopy_propagate()

    def step_move(self):
        self.set_state(self._vnodes[1].belief.mean)

    def set_state(self, state):
        self._state = np.array(state)
        v0 = self._vnodes[0]
        cov = np.diag([1/self._startFNode_pos_prec, 1/self._startFNode_pos_prec, 1/self._startFNode_vel_prec, 1/self._startFNode_vel_prec])
        self._fnode_start._factor = Gaussian(v0.dims, state, cov)
        v0._belief = Gaussian(v0.dims, state, cov.copy())

        # Init position for each vnode, to avoid 0 distance in DistFNode
        for i in range(1, self._steps):
            s = state + np.random.rand(4, 1) * 0.01
            s[:2] += s[2:] * self._dt
            v = self._vnodes[i]
            v._belief = Gaussian(v.dims, s, cov.copy())

    def set_target(self, target):
        if target is not None:
            self._fnode_end._factor = Gaussian(self._vnodes[-1].dims, target, np.diag([1]*4))
        else:
            self._fnode_end._factor = Gaussian.identity(self._vnodes[-1].dims)

    def push_msg(self, msg):
        '''Called by other agent to simulate the other sending message to self agent.'''
        _type, aname, vname, p = msg
        if p is None:
            return
        if aname not in self._others:
            print(f'push msg: name {aname} not found')
            return
        vnodes: List[RemoteVNode] = self._others[aname]['v']
        vnode: RemoteVNode = None
        for v in vnodes:
            if v.name == vname:
                vnode = v
                break
        if vnode is None:
            print('vname not found')
            return

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
        fnodes = [DistFNode(
            f'{on}.f{i}', [vnodes[i-1], self._vnodes[i]], safe_dist=self.r+other.r,
            z_precision=self._distFNode_prec
        ) for i in range(1, self._steps)]

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

    def step_plan(self, iters = 12):
        for a in self._agents:
            a.step_connect()
        for i in range(iters):
            for a in self._agents:
                a.step_com()
            for a in self._agents:
                a.step_propagate()

    def step_move(self):
        for a in self._agents:
            a.step_move()
