import time
import numpy as np
import matplotlib.pyplot as plt

import os
from maze_solver import MazeSolver
from maze_generator import MazeGenerator
from maze_visualiser import MazeVisualiser

def calculate_execution_time(solver_fn):
    t_start = time.perf_counter()
    solver_fn()
    t_end = time.perf_counter()
    return t_end - t_start

def record_memory_usage(solver_fn):
    ret_val = solver_fn()
    return ret_val[-1] # Memory usage is the last element in the return value

def execution_time_experiment():
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
            os.makedirs('./results/exec_time', exist_ok=True)
            np.savetxt(f'./results/exec_time/{alg}.csv', arr, delimiter=',')

def memory_usage_experiment():
    # Compare the three algorithms for different input sizes
    input_sizes = np.arange(10, 101, 10)
    random_seeds = np.arange(1, 20)
    memory_usage = {
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
            memory_usage['bfs'][i, j] = record_memory_usage(maze_solver.bfs)
            memory_usage['dfs'][i, j] = record_memory_usage(maze_solver.dfs)
            memory_usage['a_star'][i, j] = record_memory_usage(maze_solver.a_star)
            memory_usage['value_iteration'][i, j] = record_memory_usage(maze_solver.makrov_value_iteration)
            memory_usage['policy_iteration'][i, j] = record_memory_usage(maze_solver.makrov_policy_iteration)

        for alg, arr in memory_usage.items():
            os.makedirs('./results/memory_usage', exist_ok=True)
            np.savetxt(f'./results/memory_usage/{alg}.csv', arr, delimiter=',')

def visualise_small_maze():
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
            
if __name__ == '__main__':
    
    maze_generator = MazeGenerator(dimension=5, random_seed=5)
    maze = maze_generator.prim()
    maze_solver = MazeSolver(maze)
    value, path, _, _, _ = maze_solver.makrov_policy_iteration()
    print(path[:20])
    maze_visualiser = MazeVisualiser(maze)
    maze_visualiser.draw_search_algorithm(path=path, filename='./figures/problematic.png')
