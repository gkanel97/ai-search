import argparse
from maze_solver import MazeSolver
from maze_generator import MazeGenerator
from maze_visualiser import MazeVisualiser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=20, help='Dimension of the maze')
    parser.add_argument('--alg', type=str, default='bfs', choices=['bfs', 'dfs', 'a_star', 'value_iteration', 'policy_iteration'], help='Solver algorithm')
    parser.add_argument('--seed', type=int, default=17, help='Random seed')
    parser.add_argument('--hist', action='store_true', help='Show the history of the search algorithm')
    parser.add_argument('--anim', action='store_true', help='Enable animation')
    parser.add_argument('--filename', type=str, default=None, help='Filename to save the figure')
    
    args = parser.parse_args()

    maze_dimension = args.dim // 2
    solver_algorithm = args.alg
    enable_animation = args.anim
    filename = args.filename
    search_history = args.hist
    random_seed = args.seed

    if solver_algorithm not in ['bfs', 'dfs', 'a_star', 'value_iteration', 'policy_iteration']:
        raise ValueError("Invalid solver algorithm. Please choose from 'bfs', 'dfs', 'a_star', 'value_iteration', 'policy_iteration'.")
    
    if maze_dimension > 11 and search_history:
        raise ValueError("Search history can only be visualised for mazes with dimension <= 23.")

    maze_generator = MazeGenerator(dimension=maze_dimension, random_seed=17)
    maze = maze_generator.prim()
    visualiser = MazeVisualiser(maze)
    maze_solver = MazeSolver(maze, keep_history=True)

    if solver_algorithm == 'bfs':
        path, exploration, _ = maze_solver.bfs()
        if enable_animation:
            visualiser.animate_search_algorithm(explored=exploration, path=path, filename=filename)
        else: 
            visualiser.draw_search_algorithm(path=path, explored=exploration if search_history else None, filename=filename)
    elif solver_algorithm == 'dfs':
        path, exploration, _ = maze_solver.dfs()
        if enable_animation:
            visualiser.animate_search_algorithm(explored=exploration, path=path, filename=filename)
        else:
            visualiser.draw_search_algorithm(path=path, explored=exploration if search_history else None, filename=filename)
    elif solver_algorithm == 'a_star':
        path, exploration, _ = maze_solver.a_star()
        if enable_animation:
            visualiser.animate_search_algorithm(explored=exploration, path=path, filename=filename)
        else:
            visualiser.draw_search_algorithm(path=path, explored=exploration if search_history else None, filename=filename)
    elif solver_algorithm == 'value_iteration':
        value_function, path, history, _ = maze_solver.makrov_value_iteration()
        if enable_animation:
            visualiser.animate_value_policy(history=history, path=path, filename=filename)
        else:
            visualiser.draw_value_policy(value_function=value_function, policy=None, path=path, filename=filename)
    elif solver_algorithm == 'policy_iteration':
        policy, value_function, path, history, _ = maze_solver.makrov_policy_iteration()
        if enable_animation:
            visualiser.animate_value_policy(history=history, path=path, filename=filename)
        else:
            visualiser.draw_value_policy(value_function=value_function, policy=policy, path=path, filename=filename)