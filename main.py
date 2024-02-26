import time
import numpy as np
import matplotlib.pyplot as plt

from maze_solver import MazeSolver
from maze_generator import MazeGenerator
from maze_visualiser import MazeVisualiser

def calculate_execution_time(solver_fn):
    t_start = time.perf_counter()
    solver_fn()
    t_end = time.perf_counter()
    return t_end - t_start

if __name__ == '__main__':
    
    maze_generator = MazeGenerator(dimension=5, random_seed=17)
    maze = maze_generator.generate_maze()
    visualiser = MazeVisualiser(maze)
    maze_solver = MazeSolver(maze)
    value_function, history = maze_solver.makrov_value_iteration(iterations=50)
    visualiser.draw_value_function(history, filename='value_iteration.gif')
    # path, exploration = maze_solver.bfs()
    # visualiser.draw_maze(path=path, explored=exploration, filename='./figures/bfs.png')
    # path, exploration = maze_solver.dfs()
    # visualiser.draw_maze(path=path, explored=exploration, filename='./figures/dfs.png')
    # path, exploration = maze_solver.a_star()
    # visualiser.draw_maze(path=path, explored=exploration, filename='./figures/a_star.png')

    # import time

    # # Compare the three algorithms for different input sizes
    # input_sizes = [10, 20, 30, 40, 50, 60]
    # bfs_times = []
    # dfs_times = []
    # a_star_times = []

    # for size in input_sizes:
    #     maze_generator = MazeGenerator(dimension=size, random_seed=42)
    #     maze = maze_generator.generate_maze()
    #     maze_solver = MazeSolver(maze)
    #     bfs_times.append(calculate_execution_time(maze_solver.bfs))
    #     dfs_times.append(calculate_execution_time(maze_solver.dfs))
    #     a_star_times.append(calculate_execution_time(maze_solver.a_star))

    # bar_width = 2
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.bar(np.array(input_sizes) - bar_width, bfs_times, width=bar_width, label='BFS')
    # ax.bar(input_sizes, dfs_times, width=bar_width, label='DFS')
    # ax.bar(np.array(input_sizes) + bar_width, a_star_times, width=bar_width, label='A*')
    # ax.set_xlabel('Input size')
    # ax.set_ylabel('Time (s)')
    # ax.set_title('Comparison of search algorithms')
    # ax.legend()
    # plt.show()