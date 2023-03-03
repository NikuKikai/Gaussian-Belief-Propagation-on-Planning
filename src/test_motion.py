import pygame as pg
import pygame.locals as pgl
import sys

from motion.obstacle import ObstacleMap
from motion.agent import Agent, Env


if __name__ == '__main__':

    omap = ObstacleMap()
    omap.set_circle('', 145, 85, 20)

    agent0 = Agent('a0', [0, 0, 100, 50], [350, 100, 0, 0], steps=5, omap=omap)
    agent1 = Agent('a1', [350, 100, 0, 0], [0, 50, 0, 0], steps=5, omap=omap)

    env = Env()
    env.add_agent(agent0)
    env.add_agent(agent1)


    pg.init()
    surf = pg.display.set_mode((1000, 800))
    while True:
        surf.fill((0, 0, 0))

        # Draw obstacles
        for o in omap.objects.values():
            if o['type'] == 'circle':
                pg.draw.circle(surf, (222, 0, 0), (o['centerx']+10, o['centery']+10), o['radius'], 1)

        env.step()
        for agent in env._agents:
            c = 255
            for vnode, p in agent._beliefs.items():
                if p is None:
                    continue
                x, y, vx, vy = p.mean[:, 0]
                pg.draw.circle(surf, (c, c, c), (x+10, y+10), 5, 1)
                c -= 33

        for event in pg.event.get():
            if event.type == pgl.QUIT:
                pg.quit()
                sys.exit()
        pg.display.update()
        pg.time.wait(500)
