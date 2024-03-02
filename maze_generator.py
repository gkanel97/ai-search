import numpy as np
from mazelib import Maze
from mazelib.generate.Prims import Prims

class MazeGenerator:
    """
    A class for generating mazes using the Prims algorithm.

    Parameters:
    -----------
    dimension : int
        The dimension of the maze (number of rows and columns).
    random_seed : int, optional
        The random seed used for maze generation. If not provided, a random seed will be used.

    Attributes:
    -----------
    dim : int
        The dimension of the maze.
    random_seed : int or None
        The random seed used for maze generation.

    Methods:
    --------
    prim()
        Generates a maze using the Prims algorithm.

    """

    def __init__(self, dimension, random_seed=None):
        self.dim = dimension
        self.random_seed = int(random_seed) if random_seed is not None else None
    
    def prim(self):
        """
        Generates a maze using the Prims algorithm.

        Returns:
        --------
        maze : numpy.ndarray
            The generated maze represented as a 2D numpy array.

        """
        Maze.set_seed(self.random_seed)
        m = Maze()
        m.generator = Prims(self.dim, self.dim)
        m.generate()
        maze = np.array(m.grid)
        maze[1, 0] = 0
        maze[maze.shape[0] - 2, maze.shape[1] - 1] = 0
        return maze