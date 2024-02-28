import numpy as np
import matplotlib.pyplot as plt

def calculate_y_yerr(filename):
    with open(f'results/{filename}.csv', 'r') as fp:
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

# Every line in the file is a different input size
# Create a line plot with error bars
# x = input_sizes
# y = mean of the execution times
# error = standard deviation of the execution times
files = ['bfs', 'dfs', 'a_star', 'value_iteration', 'policy_iteration']
labels = ['BFS', 'DFS', 'A*', 'Value Iteration', 'Policy Iteration']
colors = ['royalblue', 'darkgoldenrod', 'darkgreen', 'purple', 'black']
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(10, 101, 10)
for f, l, c in zip(files, labels, colors):
    y, y_error = calculate_y_yerr(f)
    ax.plot(x, y, marker='o', markersize=5, linewidth=2, label=l, color=c)
    ax.fill_between(x, y-y_error, y+y_error, alpha=0.3)
ax.set_xlabel('Maze dimension', fontsize=12)
ax.set_ylabel('Execution time (sec)', fontsize=12)
ax.legend(ncols=5, loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=11)
ax.semilogy()
plt.show()