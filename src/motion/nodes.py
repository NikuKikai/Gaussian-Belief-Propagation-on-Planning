from typing import Tuple, List, Dict
import numpy as np
from fg.gaussian import Gaussian
from fg.factor_graph import VNode, FNode, FactorGraph

from .obstacle import ObstacleMap


class RemoteVNode(VNode):
    def __init__(self, name: str, dims: list, belief: Gaussian = None) -> None:
        super().__init__(name, dims, belief)
        self._msgs = {}

    def update_belief(self) -> Gaussian:
        # return super().update_belief()
        return None
    def calc_msg(self, edge):
        # return super().calc_msg(edge)
        return self._msgs.get(edge, None)


class DynaFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                 dt: float = 0.1, pos_prec: float = 10, vel_prec: float = 2) -> None:
        assert len(vnodes) == 2
        super().__init__(name, vnodes, factor)
        self._dt = dt
        self._pos_prec= pos_prec
        self._vel_prec= vel_prec

    def update_factor(self):
        # NOTE target: ||h(x) - z)||2 -> 0
        dt = self._dt
        v0 = self._vnodes[0].mean
        v1 = self._vnodes[1].mean
        v = np.vstack([v0, v1])  # [8, 1]
        z = np.zeros((4, 1))

        # kinetic
        k = np.identity(4)   # [4, 4]
        k[:2, 2:] = np.identity(2) * dt

        h = k @ v0 - v1     # [4, 1]
        # jacob of h
        jacob = np.array([
            [1, 0, dt, 0, -1, 0, 0, 0],  # h(x)[0] = dx = x(k) + vx(k) * dt - x(k+1)
            [0, 1, 0, dt, 0, -1, 0, 0],  # h(x)[1] = dy = y(k) + vy(k) * dt - y(k+1)
            [0, 0, 1, 0, 0, 0, -1, 0],  # h(x)[2] = dvx = vx(k) - vx(k+1)
            [0, 0, 0, 1, 0, 0, 0, -1],  # h(x)[3] = dvy = vy(k) - vy(k+1)
        ])  # [4, 8]

        # presicion of observation z (i.e. the target of h) (here is zero)
        # precision = np.linalg.inv(np.array([
        #     [dt**3/3, 0, dt**2/2, 0],
        #     [0, dt**3/3, 0, dt**2/2],
        #     [dt**2/2, 0, dt, 0],
        #     [0, dt**2/2, 0, dt]
        # ])) * self._z_precision  # [4, 4]
        precision = np.diag([self._pos_prec, self._pos_prec, self._vel_prec, self._vel_prec])

        # NOTE https://arxiv.org/pdf/1910.14139.pdf
        prec = jacob.T @ precision @ jacob
        info = jacob.T @ precision @ (jacob @ v + z - h)

        self._factor = Gaussian.from_info(self.dims, info, prec)


class ObstacleFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                 omap: ObstacleMap = None, safe_dist: float = 5, z_precision: float = 100) -> None:
        assert len(vnodes) == 1
        super().__init__(name, vnodes, factor)
        self._omap = omap
        self._safe_dist = safe_dist
        self._z_precision = z_precision

    def update_factor(self):
        # target: ||h(x) - z)||2 -> 0
        z = np.zeros((1, 1))
        v = self._vnodes[0].mean  # [4, 1]

        distance, distance_gradx, distance_grady = self._omap.get_d_grad(v[0, 0], v[1, 0])
        # distance -= self._safe_radius

        h = np.array([[max(0, 1 - distance / self._safe_dist)]])
        if distance > self._safe_dist:
            jacob = np.zeros((1, 4))
        else:
            jacob = np.array([[-distance_gradx/self._safe_dist, -distance_grady/self._safe_dist, 0, 0]])  # [1, 4]
        precision = np.identity(1) * self._z_precision * self._safe_dist**2

        prec = jacob.T @ precision @ jacob
        info = jacob.T @ precision @ (jacob @ v + z - h)
        self._factor = Gaussian.from_info(self.dims, info, prec)


class DistFNode(FNode):
    def __init__(self, name: str, vnodes: List[VNode], factor: Gaussian = None,
                 safe_dist: float = 20, z_precision: float = 100) -> None:
        assert len(vnodes) == 2
        super().__init__(name, vnodes, factor)
        self._safe_dist = safe_dist
        self._z_precision = z_precision

    def update_factor(self):
        # target: ||h(x) - z)||2 -> 0
        z = np.zeros((1, 1))
        v0 = self._vnodes[0].mean  # [4, 1]
        v1 = self._vnodes[1].mean  # [4, 1]
        if np.allclose(v0, v1):
            v1 += np.random.rand(4, 1) * 0.01
        v = np.vstack([v0, v1])  # [8, 1]

        distance = np.linalg.norm(v0[:2, 0] - v1[:2, 0])
        distance_gradx0, distance_grady0 = v0[0, 0]-v1[0, 0], v0[1, 0]-v1[1, 0]
        distance_gradx0 /= distance
        distance_grady0 /= distance

        if distance > self._safe_dist:
            prec = np.identity(8) * 0.0001
            info = prec @ v

        else:
            h = np.array([[1 - distance / self._safe_dist]])
            jacob = np.array([[
                -distance_gradx0/self._safe_dist, -distance_grady0/self._safe_dist, 0, 0,
                distance_gradx0/self._safe_dist, distance_grady0/self._safe_dist, 0, 0]])  # [1, 8]
            precision = np.identity(1) * self._z_precision * (self._safe_dist**2)

            prec = jacob.T @ precision @ jacob
            info = jacob.T @ precision @ (jacob @ v + z - h)

            # NOTE
            # prec has a structure like [A, 0, C, 0] , which is not invertable
            #                           [0, 0, 0, 0]
            #                           [D, 0, B, 0]
            #                           [0, 0, 0, 0]
            # modify prec to be [A, 0, C, 0] to avoid this problem and will not affect result.
            #                   [0, I, 0, 0]
            #                   [D, 0, B, 0]
            #                   [0, 0, 0, I]
            prec[2, 2] = 1
            prec[3, 3] = 1
            prec[6, 6] = 1
            prec[7, 7] = 1
        self._factor = Gaussian.from_info(self.dims, info, prec)
