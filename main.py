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
    maze = maze_generator.prim()
    visualiser = MazeVisualiser(maze)
    maze_solver = MazeSolver(maze)
    path, exploration = maze_solver.bfs()
    visualiser.draw_search_algorithm(path=path, explored=exploration, filename='./figures/bfs.png')
    path, exploration = maze_solver.dfs()
    visualiser.draw_search_algorithm(path=path, explored=exploration, filename='./figures/dfs.png')
    path, exploration = maze_solver.a_star()
    visualiser.draw_search_algorithm(path=path, explored=exploration, filename='./figures/a_star.png')
    value_function, path, history = maze_solver.makrov_value_iteration()
    visualiser.draw_value_policy(history, filename='./figures/value_iteration.gif')
    policy, value_function, path, history = maze_solver.makrov_policy_iteration()
    visualiser.draw_value_policy(history, path, filename='./figures/policy_iteration.gif')


    # Compare the three algorithms for different input sizes
    input_sizes = np.arange(10, 101, 10)
    random_seeds = np.arange(1, 20)
    exec_times = {
        'bfs': np.zeros((len(input_sizes), len(random_seeds))),
        'dfs': np.zeros((len(input_sizes), len(random_seeds))),
        'a_star': np.zeros((len(input_sizes), len(random_seeds))),
        'value_iteration': np.zeros((len(input_sizes), len(random_seeds))),
        'policy_iteration': np.zeros((len(input_sizes), len(random_seeds)))
    }

    for i, size in enumerate(input_sizes):
        for j, seed in enumerate(random_seeds):
            print(f"Creating maze with dimension = {size} and random state = {seed}")
            maze_generator = MazeGenerator(dimension=size, random_seed=seed)
            maze = maze_generator.prim()
            maze_solver = MazeSolver(maze)
            exec_times['bfs'][i, j] = calculate_execution_time(maze_solver.bfs)
            exec_times['dfs'][i, j] = calculate_execution_time(maze_solver.dfs)
            exec_times['a_star'][i, j] = calculate_execution_time(maze_solver.a_star)
            exec_times['value_iteration'][i, j] = calculate_execution_time(maze_solver.makrov_value_iteration)
            exec_times['policy_iteration'][i, j] = calculate_execution_time(maze_solver.makrov_policy_iteration)

        for alg, arr in exec_times.items():
            np.savetxt(f'./results/{alg}.csv', arr, delimiter=',')

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