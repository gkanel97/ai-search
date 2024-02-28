import numpy as np
import matplotlib.pyplot as plt

with open('results/bfs.csv', 'r') as file:
    bfs = file.read().split('\n')
    bfs = [x.split(',') for x in bfs]
    bfs = [[float(x) for x in y] for y in bfs]
    bfs = np.array(bfs)

# Every line in the file is a different input size
# Create a line plot with error bars
# x = input_sizes
# y = mean of the execution times
# error = standard deviation of the execution times
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(10, 101, 10)
y = np.mean(bfs, axis=0)
error = np.std(bfs, axis=0)
ax.errorbar(x, y, yerr=error, fmt='o', label='BFS')
plt.show()