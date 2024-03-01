import time
import numpy as np
import matplotlib.pyplot as plt

import os
from maze_solver import MazeSolver
from maze_generator import MazeGenerator
from maze_visualiser import MazeVisualiser
from plot_results import ResultsPlotter

def calculate_execution_time(solver_fn):
    t_start = time.perf_counter()
    solver_fn()
    t_end = time.perf_counter()
    return t_end - t_start

def record_memory_usage(solver_fn):
    ret_val = solver_fn()
    return ret_val[-1] # Memory usage is the last element in the return value

def execution_time_experiment():
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
            maze_solver = MazeSolver(maze, record_memory=True)
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
    maze_solver = MazeSolver(maze, keep_history=True)
    path, exploration, _ = maze_solver.bfs()
    visualiser.draw_search_algorithm(path=path, explored=exploration, filename='./figures/bfs.png')
    path, exploration, _ = maze_solver.dfs()
    visualiser.draw_search_algorithm(path=path, explored=exploration, filename='./figures/dfs.png')
    path, exploration, _ = maze_solver.a_star()
    visualiser.draw_search_algorithm(path=path, explored=exploration, filename='./figures/a_star.png')
    _, path, history, _ = maze_solver.makrov_value_iteration()
    visualiser.draw_value_policy(history, path, filename='./figures/value_iteration.gif')
    _, _, path, history, _ = maze_solver.makrov_policy_iteration()
    visualiser.draw_value_policy(history, path, filename='./figures/policy_iteration.gif')
            
if __name__ == '__main__':

    # visualise_small_maze()
    # memory_usage_experiment()
    # execution_time_experiment()
    plotter = ResultsPlotter()
    plotter.plot_execution_time()
    plotter.plot_memory_usage()