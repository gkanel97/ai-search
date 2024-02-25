import matplotlib as mpl
import matplotlib.pyplot as plt

class MazeVisualiser:

    def __init__(self, maze):
        self.maze = maze

    def draw_maze(self, path=None, explored=None, filename=None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10), layout='constrained')
    
        # Set the border color to white
        fig.patch.set_edgecolor('white')
        fig.patch.set_linewidth(0)

        ax.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')

        if path is not None:
            x_coords = [x[1] for x in path]
            y_coords = [y[0] for y in path]
            ax.plot(x_coords, y_coords, '--', color='red', linewidth=2)

        if explored is not None:
            x_coords = [x[1] for x in explored]
            y_coords = [y[0] for y in explored]
            num_dots = len(x_coords)

            cmap = mpl.cm.viridis
            bounds = range(num_dots+1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='max')

            colors = [cmap(i/num_dots) for i in range(num_dots)]
            ax.scatter(x_coords, y_coords, color=colors, s=200)

            # Add colorbar
            # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', extend='max')
            # cbar.set_label('Exploration Steps')

        # Draw entry and exit arrows
        ax.arrow(
            self.maze.shape[1]-1.3, self.maze.shape[0]-2, 0.4, 0, 
            fc='black', ec='black', head_width=0.3, head_length=0.3
        )
    
        ax.set_xticks([])
        ax.set_yticks([])

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()