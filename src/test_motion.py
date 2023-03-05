import pygame as pg
import pygame.locals as pgl
import sys
import itertools

from motion.obstacle import ObstacleMap
from motion.agent import Agent, Env


if __name__ == '__main__':

    omap = ObstacleMap()
    omap.set_circle('', 145, 85, 20)

    agent0 = Agent('a0', [0, 0, 10, 20], [350, 100, 0, 0], steps=6, radius=15, omap=omap)
    agent1 = Agent('a1', [350, 100, 0, 0], [0, 50, 0, 0], steps=6, radius=15, omap=omap)

    env = Env()
    env.add_agent(agent0)
    env.add_agent(agent1)

    colors = {
        agent0: (0, 222, 0),
        agent1: (0, 0, 255),
    }

    pg.init()
    surf = pg.display.set_mode((1000, 800))
    xoff, yoff= 500, 400
    while True:
        surf.fill((255, 255, 255))

        # Draw obstacles
        for o in omap.objects.values():
            if o['type'] == 'circle':
                pg.draw.circle(surf, (222, 0, 0), (o['centerx']+xoff, o['centery']+yoff), o['radius'], 1)

        env.step_plan()
        for agent in env._agents:
            color = colors[agent]
            state = agent.get_state()
            if state[0] is None:
                continue
            x, y, vx, vy = state[0]
            pg.draw.circle(surf, color, (x+xoff, y+yoff), agent._radius, 1)

            for s0, s1 in itertools.pairwise(state):
                if s0 is None or s1 is None:
                    break
                x0, y0, _, _ = s0
                x1, y1, _, _ = s1
                pg.draw.line(surf, color, (x0+xoff, y0+yoff), (x1+xoff, y1+yoff))
        env.step_move()

        for event in pg.event.get():
            if event.type == pgl.QUIT:
                pg.quit()
                sys.exit()
        pg.display.update()
        pg.time.wait(10)
