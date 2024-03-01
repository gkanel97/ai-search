import matplotlib as mpl
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt

class MazeVisualiser:

    def __init__(self, maze):
        self.maze = maze

    def draw_search_algorithm(self, path=None, explored=None, filename=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), layout='constrained')
    
        # Set the border color to white
        fig.patch.set_edgecolor('white')
        fig.patch.set_linewidth(0)

        ax.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')

        if path is not None:
            x_coords = [x[1] for x in path]
            y_coords = [y[0] for y in path]
            ax.plot(x_coords, y_coords, color='red', linewidth=2)

        if explored is not None:
            x_coords = [x[1] for x in explored]
            y_coords = [y[0] for y in explored]
            num_dots = len(x_coords)

            cmap = mpl.cm.viridis
            bounds = range(num_dots+1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='max')

            colors = [cmap(i/num_dots) for i in range(num_dots)]
            ax.scatter(x_coords, y_coords, color=colors, s=500)

            # Add colorbar
            cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', shrink=0.7)
            cbar.set_label('Exploration Steps', fontsize=14)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.show()

    def draw_value_policy(self, value_function, policy, title='', path=None, ax=None, max_value=None, filename=None):
        
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])

        value_function = np.array(value_function)
        if max_value is None:
            max_value = np.max(value_function)
        x_coords = np.arange(value_function.shape[1])
        y_coords = np.arange(value_function.shape[0])
        X, Y = np.meshgrid(x_coords, y_coords)
        ax.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')
        ax.scatter(X, Y, c=value_function, cmap=plt.cm.viridis, s=700, alpha=self.maze != 1)

        for x, row in enumerate(value_function):
            for y, col in enumerate(row):
                if self.maze[x, y] == 0:
                    xx = X[x, y]
                    yy = Y[x, y]
                    ax.text(
                        xx, yy, f'{col:.2f}', fontsize=8, ha='center', va='center', 
                        color='black' if col > 0.4 * max_value else 'silver'
                    )

        if policy is not None:
            for x, row in enumerate(policy):
                for y, col in enumerate(row):
                    if self.maze[x, y] == 0:
                        xx = X[x, y]
                        yy = Y[x, y]
                        color='black' if value_function[x, y] > 0.4 * max_value else 'silver'
                        if col == (1, 0): # Down
                            ax.arrow(xx, yy+0.2, 0, 0.001, head_length=0.1, head_width=0.1, fc=color, ec=color)
                        elif col == (-1, 0): # Up
                            ax.arrow(xx, yy-0.2, 0, -0.001, head_length=0.1, head_width=0.1, fc=color, ec=color)
                        elif col == (0, 1): # Right
                            ax.arrow(xx+0.2, yy, 0.001, 0, head_length=0.1, head_width=0.1, fc=color, ec=color)
                        elif col == (0, -1): # Left
                            ax.arrow(xx-0.2, yy, -0.001, 0, head_length=0.1, head_width=0.1, fc=color, ec=color)
                        else:
                            print(f"{x, y}: Invalid policy!")
        if path is not None:
            x_coords = [x[1] for x in path]
            y_coords = [y[0] for y in path]
            ax.plot(x_coords, y_coords, '--', color='red', linewidth=2)

        ax.set_title(title)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        if fig is not None:
            plt.show()

    '''
    Create an animation with the evolution of the value function and policy
    A heatmap is used to represent the value function at each iteration over the maze
    Arguments:
    history -- dictionary with list of values and policies
    filename -- name of the file to save the animation
    '''
    def animate_value_policy(self, history, path=None, filename=None):

        max_value = np.max(history['value'])

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])

        def animate(i):
            ax.clear()
            value_function = np.array(history['value'])[i]
            try:
                policy = history['policy'][i]
            except:
                policy = None
            p = path if i == len(history['value'])-1 else None
            self.draw_value_policy(value_function, policy, f"Iteration {i+1}", path=p, ax=ax, max_value=max_value)

        ani = animation.FuncAnimation(fig, animate, frames=len(history['value']), interval=1000, repeat_delay=5000)
        if filename is not None:
            ani.save(filename, writer='pillow')
        plt.show()
