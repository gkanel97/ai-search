import numpy as np
import matplotlib.pyplot as plt

class ResultsPlotter():
    def __init__(self) -> None:
        self.algorithms = ['bfs', 'dfs', 'a_star', 'value_iteration', 'policy_iteration']
        self.labels = ['BFS', 'DFS', 'A*', 'Value Iteration', 'Policy Iteration']
        self.colors = ['royalblue', 'darkgoldenrod', 'darkgreen', 'purple', 'black']
        self.maze_sizes = np.arange(10, 51, 5)

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
        x = self.maze_sizes * 2
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
        x = self.maze_sizes * 2
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