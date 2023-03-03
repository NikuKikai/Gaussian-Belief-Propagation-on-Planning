from typing import Tuple, List, Dict
import numpy as np


class ObstacleMap:
    def __init__(self) -> None:
        self.objects = {}

    def set_circle(self, name: str, centerx, centery, radius):
        o = {'type': 'circle', 'name': name, 'centerx': centerx, 'centery': centery, 'radius': radius}
        self.objects[name] = o

    def get_d_grad(self, x, y) -> Tuple[float, float, float]:
        mindist = np.inf
        mino = None
        for o in self.objects.values():
            if o['type'] == 'circle':
                ox, oy, r = o['centerx'], o['centery'], o['radius']
                d = np.sqrt((x - ox)**2 + (y - oy)**2) - r
                if d < mindist:
                    mindist = d
                    mino = o
        if mino is None:
            return np.inf, 0, 0
        if mino['type'] == 'circle':
            ox, oy = o['centerx'], o['centery']
            dx, dy = x - ox, y - oy
            mag = np.sqrt(dx**2 + dy**2)
            return mindist, dx/mag, dy/mag
