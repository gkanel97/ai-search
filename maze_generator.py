import random
import numpy as np
from mazelib import Maze
from mazelib.generate.Prims import Prims

class MazeGenerator:

    def __init__(self, dimension, random_seed=None):
        self.dim = dimension
        self.random_seed = random_seed

    def random_dfs(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # Create a grid filled with walls, represented by 1
        maze = np.ones((self.dim*2+1, self.dim*2+1))

        # Define the starting point
        x, y = (0, 0)
        maze[2*x+1, 2*y+1] = 0

        # Initialize the stack with the starting point
        stack = [(x, y)]
        while len(stack) > 0:
            x, y = stack[-1]

            # Define possible directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # If there is a not visited neighbour within the boundaries and is a wall, make it a cell
                if nx >= 0 and ny >= 0 and nx < self.dim and ny < self.dim and maze[2*nx+1, 2*ny+1] == 1:
                    maze[2*nx+1, 2*ny+1] = 0
                    maze[2*x+1+dx, 2*y+1+dy] = 0
                    stack.append((nx, ny))
                    break
            else:
                # If all neighbours have been visited, pop the cell from the stack
                stack.pop()
                
        # Create an entrance and an exit
        maze[1, 0] = 0
        maze[-2, -1] = 0

        return maze
    
    def prim(self):
        Maze.set_seed(self.random_seed)
        m = Maze()
        m.generator = Prims(self.dim, self.dim)
        m.generate()
        maze = np.array(m.grid)
        maze[1, 0] = 0
        maze[maze.shape[0] - 2, maze.shape[1] - 1] = 0
        return maze