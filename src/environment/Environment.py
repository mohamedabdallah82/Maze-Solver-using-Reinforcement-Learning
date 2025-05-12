# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 21:57:17 2025

@author: Mohamed Abdallah
"""

import gym
from gym import spaces
import numpy as np
import pygame
import random
import os

class myMazeEnv(gym.Env):
    # This is a standard attribute in Gym environments.
    # It gives Gym some info about how your environment supports rendering.
    # 'human' means your environment can open a window and display visual output (like the animated maze).
    metadata = {'render_modes': ['human'], 'render_fps': 20}

    # “dunder” (short for double underscore) automatically called when you create an instance of a class.
    def __init__(self, render_mode=None, grid_size=(6,6), number_of_walls=10, cell_size=80):
        super().__init__()      # Make sure the Gym engine is running before I start customizing my maze.
        self.grid_size = grid_size              # n*n maze
        self.grid = np.zeros(self.grid_size, dtype=int)             # All cells free
        
        # initial pos
        self.start_pos = (-1, -1)
        self.agent_pos = self.start_pos
        self.previous_pos = self.start_pos

        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size[0]-1, shape=(2,), dtype=np.int32)
        
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        self.cell_side = ["top", "bottom", "left", "right"]
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Get the project root directory (two levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # agent, goal & enemy as sprite
        self.agent_img = pygame.image.load(os.path.join(project_root, "assets", "jerry.png"))
        self.goal_img = pygame.image.load(os.path.join(project_root, "assets", "cheese.png"))
        self.enemy_img = pygame.image.load(os.path.join(project_root, "assets", "tom.png"))
        
        # Optional: resize images to fit cell size
        self.agent_img = pygame.transform.scale(self.agent_img, (self.cell_size, self.cell_size))
        self.goal_img = pygame.transform.scale(self.goal_img, (self.cell_size, self.cell_size))
        self.enemy_img = pygame.transform.scale(self.enemy_img, (self.cell_size, self.cell_size))
        
        # Wall structure: walls on each cell side (top, bottom, left, right)
        self.cell_walls = {}  # {(row, col): {"top": True, "right": False, ...}}

        mazeRows, mazeColumns = self.grid_size
        for i in range(mazeRows):
            for j in range(mazeColumns):
                self.cell_walls[(i, j)] = {"top": False, "bottom": False, "left": False, "right": False}

        for w in range(number_of_walls):
            self._random_walls()
            
        if render_mode == "human":
            self.render()                   # show initial grid
            self._setup_mode()

        return
    
    
    def reset(self):
        super().reset()
        self.agent_pos = self.start_pos
        self.previous_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        self.previous_pos = self.agent_pos
        move = self.actions[action]
        new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])

        if self._is_valid(new_pos) and not self._is_wall(self.agent_pos, new_pos, action):
            self.agent_pos = new_pos

        cell_value = self.grid[self.agent_pos]
        if cell_value == 2:
            reward = 1.0
        elif cell_value == -2:
            reward = -1.0
        else:
            reward = -0.01
            
        done = (cell_value == 2) or (cell_value == -2)

        self.render()
        return np.array(self.agent_pos), reward, done, False, {}    # False: not important now, {}: should be prob
     
    
    def render(self):
        if self.render_mode != 'human':
            return
        if self.window is None:
            # initializes all of Pygame’s modules, called before any Pygame drawing or display stuff.
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.grid.shape[1] * self.cell_size, self.grid.shape[0] * self.cell_size)  # (width, hight)
            )
            pygame.display.set_caption("Maze")
            self.clock = pygame.time.Clock()
            
        self._animate_agent(self.previous_pos, self.agent_pos)
        return


    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
        return
    
    
    
    def _random_walls(self):
        n = self.grid_size[0]-1
        # top, bottom, left, right
        randomSide = random.randint(0, 3)
        
        # available cells with this side
        cells_for_side = {
            0 : [random.randint(1, n), random.randint(0, n)], 
            1 : [random.randint(0, n-1), random.randint(0, n)], 
            2 : [random.randint(0, n), random.randint(1, n)], 
            3 : [random.randint(0, n), random.randint(0, n-1)]
        }
        randomRow, randomCol = cells_for_side[randomSide]
        # print("----------", randomRow, randomCol, randomSide)
        
        # set wall for cells
        side = self.cell_side
        if randomSide == 0:
            self.cell_walls[(randomRow, randomCol)][side[0]] = True
            self.cell_walls[(randomRow-1, randomCol)][side[1]] = True
            
        elif randomSide == 1:
            self.cell_walls[(randomRow, randomCol)][side[1]] = True
            self.cell_walls[(randomRow+1, randomCol)][side[0]] = True
            
        elif randomSide == 2:
            self.cell_walls[(randomRow, randomCol)][side[2]] = True
            self.cell_walls[(randomRow, randomCol-1)][side[3]] = True
            
        elif randomSide == 3:
            self.cell_walls[(randomRow, randomCol)][side[3]] = True
            self.cell_walls[(randomRow, randomCol+1)][side[2]] = True
        
    
    def _is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]

    def _is_wall(self, from_pos, to_pos, action):
        fr, fc = from_pos
        tr, tc = to_pos
        side = ["top", "bottom", "left", "right"]
        
        if fr == tr or fc == tc:
            return self.cell_walls[(fr, fc)][side[action]]  # (fr, fc) cell dimension
        return True                                         # Not adjacent = not allowed
            
        """
        another sol.
        if fr == tr and fc == tc + 1:       # fr == tr : same row, fc == tc + 1 : go left
            return self.cell_walls[(fr, fc)]["left"]
        if fr == tr and fc == tc - 1:
            return self.cell_walls[(fr, fc)]["right"]
        if fr == tr + 1 and fc == tc:
            return self.cell_walls[(fr, fc)]["top"]
        if fr == tr - 1 and fc == tc:
            return self.cell_walls[(fr, fc)]["bottom"]
        return True  # Not adjacent = not allowed
        """        

    
    def _animate_agent(self, from_pos, to_pos, steps=10):
        for i in range(steps + 1):
            # prevent freezing when(Clicking, Moving or resizing)
            pygame.event.pump() # I don’t care what the events are, but just... acknowledge them so my app doesn’t freeze!
            
            x = from_pos[1] + (to_pos[1] - from_pos[1]) * i / steps
            y = from_pos[0] + (to_pos[0] - from_pos[0]) * i / steps
            
            self._draw(y, x)
            # updates the screen so you can see what you just drew
            pygame.display.flip()
            # Waits just long enough to make sure the loop doesn’t run faster than fps
            self.clock.tick(self.metadata["render_fps"])   # wait so you get about (20) frames per second

    def _draw(self, agent_row, agent_col):
        self.window.fill((0, 0, 0))  # (R, G, B)
        cs = self.cell_size

        for (y, x), walls in self.cell_walls.items():
            rect = pygame.Rect(x * cs, y * cs, cs, cs)

            # Cells color
            pygame.draw.rect(self.window, (50, 50, 50), rect)
            
            # goal & enemy sprite
            if self.grid[y, x] == 2:
                self.window.blit(self.goal_img, rect.topleft)
            elif self.grid[y, x] == -2:
                self.window.blit(self.enemy_img, rect.topleft)

            # Wall borders
            if walls["top"]:
                pygame.draw.line(self.window, (255, 255, 255), (x * cs, y * cs), ((x + 1) * cs, y * cs), 2)
            if walls["bottom"]:
                pygame.draw.line(self.window, (255, 255, 255), (x * cs, (y + 1) * cs), ((x + 1) * cs, (y + 1) * cs), 2)
            if walls["left"]:
                pygame.draw.line(self.window, (255, 255, 255), (x * cs, y * cs), (x * cs, (y + 1) * cs), 2)
            if walls["right"]:
                pygame.draw.line(self.window, (255, 255, 255), ((x + 1) * cs, y * cs), ((x + 1) * cs, (y + 1) * cs), 2)

        # agent sprite
        agent_rect = pygame.Rect(int(agent_col * self.cell_size), int(agent_row * self.cell_size), self.cell_size, self.cell_size)
        self.window.blit(self.agent_img, agent_rect.topleft)
        

    def _setup_mode(self):
        self._draw(-1, -1)
        print("Click to select START position...")
        self.start_pos = self._wait_for_click()
        
        self.agent_pos = self.start_pos
        self.previous_pos = self.start_pos
        self.render()
    
        print("Click to select GOAL position...")
        self.goal_pos = self._wait_for_click()
        
        self.grid[self.goal_pos] = 2
        self.render()
        
        print("Click to select Tom position...")
        self.enemy_pos = self._wait_for_click()
        
        self.grid[self.enemy_pos] = -2
        self.render()
        
        

    def _wait_for_click(self):
        while True:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    col = mouse_x // self.cell_size
                    row = mouse_y // self.cell_size
                    return (row, col)
                        