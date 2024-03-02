import os
import time
import numpy as np
import matplotlib.pyplot as plt

from maze_solver import MazeSolver
from maze_generator import MazeGenerator

class ExperimentRunner():

    def __init__(self) -> None:
        self.input_sizes = np.arange(10, 51, 5)
        self.random_seeds = np.arange(1, 20)
        self.algorithms = ['bfs', 'dfs', 'a_star', 'value_iteration', 'policy_iteration']
        self.labels = ['BFS', 'DFS', 'A*', 'Value Iteration', 'Policy Iteration']
        self.colors = ['royalblue', 'darkgoldenrod', 'darkgreen', 'purple', 'black']

    def calculate_execution_time(self, solver_fn):
        t_start = time.perf_counter()
        solver_fn()
        t_end = time.perf_counter()
        return t_end - t_start

    def record_memory_usage(self, solver_fn):
        ret_val = solver_fn()
        return ret_val[-1] # Memory usage is the last element in the return value

    def execution_time_experiment(self):
        exec_times = {
            'bfs': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'dfs': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'a_star': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'value_iteration': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'policy_iteration': np.zeros((len(self.input_sizes), len(self.random_seeds)))
        }

        for i, size in enumerate(self, self.input_sizes):
            for j, seed in enumerate(self.random_seeds):
                print(f"Creating maze with dimension = {size} and random state = {seed}")
                maze_generator = MazeGenerator(dimension=size, random_seed=seed)
                maze = maze_generator.prim()
                maze_solver = MazeSolver(maze)
                exec_times['bfs'][i, j] = self.calculate_execution_time(maze_solver.bfs)
                exec_times['dfs'][i, j] = self.calculate_execution_time(maze_solver.dfs)
                exec_times['a_star'][i, j] = self.calculate_execution_time(maze_solver.a_star)
                exec_times['value_iteration'][i, j] = self.calculate_execution_time(maze_solver.makrov_value_iteration)
                exec_times['policy_iteration'][i, j] = self.calculate_execution_time(maze_solver.makrov_policy_iteration)

            os.makedirs('./results/exec_time', exist_ok=True)
            for alg, arr in exec_times.items():
                np.savetxt(f'./results/exec_time/{alg}.csv', arr, delimiter=',')

    def memory_usage_experiment(self):
        memory_usage = {
            'bfs': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'dfs': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'a_star': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'value_iteration': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'policy_iteration': np.zeros((len(self.input_sizes), len(self.random_seeds)))
        }

        for i, size in enumerate(self.input_sizes):
            for j, seed in enumerate(self.random_seeds):
                print(f"Creating maze with dimension = {size} and random state = {seed}")
                maze_generator = MazeGenerator(dimension=size, random_seed=seed)
                maze = maze_generator.prim()
                maze_solver = MazeSolver(maze, record_memory=True)
                memory_usage['bfs'][i, j] = self.record_memory_usage(maze_solver.bfs)
                memory_usage['dfs'][i, j] = self.record_memory_usage(maze_solver.dfs)
                memory_usage['a_star'][i, j] = self.record_memory_usage(maze_solver.a_star)
                memory_usage['value_iteration'][i, j] = self.record_memory_usage(maze_solver.makrov_value_iteration)
                memory_usage['policy_iteration'][i, j] = self.record_memory_usage(maze_solver.makrov_policy_iteration)

            os.makedirs('./results/memory_usage', exist_ok=True)
            for alg, arr in memory_usage.items():
                np.savetxt(f'./results/memory_usage/{alg}.csv', arr, delimiter=',')

    def calculate_y_yerr_from_file(self, filename):
        with open(filename, 'r') as fp:
            result_arr = []
            line = fp.readline()
            while line:
                tmp_arr = []
                for token in line.split(','):
                    tmp_arr.append(float(token))
                result_arr.append(tmp_arr)
                line = fp.readline()
            result_arr = np.array(result_arr)
        
        y = y = np.mean(result_arr, axis=1)
        y_error = np.std(result_arr, axis=1)
        return (y, y_error)
    
    def plot_execution_time(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = self.input_sizes * 2
        for alg, label, color in zip(self.algorithms, self.labels, self.colors):
            y, y_error = self.calculate_y_yerr_from_file(filename=f'results/exec_time/{alg}.csv')
            ax.plot(x, y, marker='o', markersize=5, linewidth=2, label=label, color=color)
            ax.fill_between(x, y-y_error, y+y_error, alpha=0.3)
        ax.set_xlabel('Maze dimension', fontsize=12)
        ax.set_ylabel('Execution time (sec)', fontsize=12)
        ax.legend(ncols=5, loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=11)
        ax.semilogy()
        plt.savefig('figures/exec_time.pdf', bbox_inches='tight')
        plt.show()

    def plot_memory_usage(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = self.input_sizes * 2
        for alg, label, color in zip(self.algorithms, self.labels, self.colors):
            y, y_error = self.calculate_y_yerr_from_file(filename=f'results/mem_usage/{alg}.csv')
            y, y_error = y / 1e6, y_error / 1e6
            ax.plot(x, y, marker='o', markersize=5, linewidth=2, label=label, color=color)
            ax.fill_between(x, y-y_error, y+y_error, alpha=0.3)
        ax.set_xlabel('Maze dimension', fontsize=12)
        ax.set_ylabel('Maximum memory allocation (MB)', fontsize=12)
        ax.legend(ncols=5, loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=11)
        ax.semilogy()
        plt.savefig('figures/mem_usage.pdf', bbox_inches='tight')
        plt.show()

    def percentage_of_explored_cells_experiment(self):

        percentage_explored = {
            'bfs': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'dfs': np.zeros((len(self.input_sizes), len(self.random_seeds))),
            'a_star': np.zeros((len(self.input_sizes), len(self.random_seeds)))
        }

        for i, size in enumerate(self.input_sizes):
            for j, seed in enumerate(self.random_seeds):
                print(f"Creating maze with dimension = {size} and random state = {seed}")
                maze_generator = MazeGenerator(dimension=size, random_seed=seed)
                maze = maze_generator.prim()
                maze_solver = MazeSolver(maze)
                maze_cells = np.count_nonzero(maze == 0)
                _, exploration, _ = maze_solver.bfs()
                percentage_explored['bfs'][i, j] = 100 * len(exploration) / maze_cells
                _, exploration, _ = maze_solver.dfs()
                percentage_explored['dfs'][i, j] = 100 * len(exploration) / maze_cells
                _, exploration, _ = maze_solver.a_star()
                percentage_explored['a_star'][i, j] = 100 * len(exploration) / maze_cells

        os.makedirs('./results/percentage_explored', exist_ok=True)
        for alg, arr in percentage_explored.items():
            np.savetxt(f'./results/percentage_explored/{alg}.csv', arr, delimiter=',')

    def plot_percentage_explored(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        x = self.input_sizes * 2
        for alg, label, color in zip(['bfs', 'dfs', 'a_star'], ['BFS', 'DFS', 'A*'], ['royalblue', 'darkgoldenrod', 'darkgreen']):
            y, y_error = self.calculate_y_yerr_from_file(filename=f'results/percentage_explored/{alg}.csv')
            ax.plot(x, y, marker='o', markersize=5, linewidth=2, label=label, color=color)
            ax.fill_between(x, y-y_error, y+y_error, alpha=0.3)
        ax.set_xlabel('Maze dimension', fontsize=12)
        ax.set_ylabel('Percentage of explored cells', fontsize=12)
        ax.legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=11)
        plt.savefig('figures/percentage_explored.pdf', bbox_inches='tight')
        plt.show()

    def a_star_heuristic_experiment(self):
        tmp_arr = []
        maze_size = 50
        for heuristic in ['manhattan', 'euclidean', 'chebyshev', 'zero']:
            for seed in self.random_seeds:
                print(f"Creating maze with dimension = {maze_size} and random state = {seed}")
                maze_generator = MazeGenerator(dimension=maze_size, random_seed=seed)
                maze = maze_generator.prim()
                maze_solver = MazeSolver(maze)
                _, exploration, _ = maze_solver.a_star(heuristic_fn=heuristic)
                tmp_arr.append(len(exploration))
            print(f"Heuristic: {heuristic}, Mean: {np.mean(tmp_arr)}, Std: {np.std(tmp_arr)}")


if __name__ == '__main__':
    exp_runner = ExperimentRunner()
    exp_runner.execution_time_experiment()
    exp_runner.memory_usage_experiment()
    exp_runner.plot_execution_time()
    exp_runner.plot_memory_usage()
    exp_runner.percentage_of_explored_cells_experiment()
    exp_runner.plot_percentage_explored()
    exp_runner.a_star_heuristic_experiment()
    exp_runner.gamma_experiment()
        
