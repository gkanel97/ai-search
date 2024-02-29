import random
import numpy as np
from queue import Queue
from utils import get_deep_size

class MazeSolver:

    def __init__(self, maze, keep_history=False, record_memory=False):
        self.maze = maze
        self.start = (1, 0)
        self.end = (maze.shape[0] - 2, maze.shape[1] - 1)
        self.moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.goal_reward = 1
        self.wall_penalty = -1
        self.keep_history = keep_history
        self.record_memory = record_memory

    def node_in_maze(self, node):
        x, y = node
        return x >= 0 and y >= 0 and x < self.maze.shape[0] and y < self.maze.shape[1]
    
    def bfs(self):
        
        queue = Queue() # A queue to add the nodes to visit
        queue.put((self.start, [self.start]))
        explored = set() # A set to store explored nodes
        explored.add(self.start)
        visited = [] # A list to store the visited nodes (for visualisation purposes)
        used_memory = 0 # Memory usage

        while not queue.empty():
            # Visit the next node in the queue
            (node, path) = queue.get()
            visited.append(node)
            if node == self.end:
                return path, visited, used_memory
            # Explore all the neighbours of the node
            for dx, dy in self.moves:
                next_node = (node[0]+dx, node[1]+dy)
                if (self.node_in_maze(next_node) and self.maze[next_node] == 0 and next_node not in explored):
                    explored.add(next_node)
                    queue.put((next_node, path + [next_node]))

            if self.record_memory:
                used_memory = max(
                    used_memory, 
                    get_deep_size(queue) + get_deep_size(explored) + get_deep_size(visited)
            )

        return None, visited, used_memory
    
    def dfs(self):
        
        stack = [(self.start, [self.start])] # A stack to add the nodes to visit
        explored = set()
        explored.add(self.start) 
        visited = [] # A list to store the visited nodes (for visualisation purposes)
        used_memory = 0 # Memory usage
        
        while len(stack) > 0:
            # Visit the next node in the stack
            (node, path) = stack.pop()
            visited.append(node)
            if node == self.end:
                return path, visited, used_memory
            # Explore all the neighbours of the node
            for dx, dy in self.moves:
                next_node = (node[0]+dx, node[1]+dy)
                if (self.node_in_maze(next_node) and self.maze[next_node] == 0 and next_node not in explored):
                    explored.add(next_node)
                    stack.append((next_node, path + [next_node]))
            
            if self.record_memory:
                used_memory = max(
                    used_memory, 
                    get_deep_size(stack) + get_deep_size(explored) + get_deep_size(visited)
                )

        return None, visited, used_memory
    
    def a_star(self):

        queue = [] # A priority queue to add the nodes to visit
        queue.append((0, self.start, [self.start]))
        explored = set() # A set to store the explored nodes
        explored.add(self.start)
        visited = [] # A list to store the visited nodes (for visualisation purposes)
        used_memory = 0 # Memory usage
        
        while len(queue) > 0:
         
            # Visit the next node in the queue
            (cost, node, path) = queue.pop(0)
            visited.append(node)
            if node == self.end:
                return path, visited, used_memory
            
            # Visit all the neighbours of the node
            for dx, dy in self.moves:
                next_node = (node[0]+dx, node[1]+dy)
                if (self.node_in_maze(next_node) and self.maze[next_node] == 0 and next_node not in explored):
                    explored.add(next_node)
                    path_cost = len(path) # Number of steps taken so far
                    heuristic = abs(self.end[0] - next_node[0]) + abs(self.end[1] - next_node[1]) # Manhattan distance to the goal
                    next_cost = path_cost + heuristic
                    queue.append((next_cost, next_node, path + [next_node]))

            # Sort the queue by cost        
            queue.sort(key=lambda x: x[0])

            if self.record_memory:
                used_memory = max(
                    used_memory, 
                    get_deep_size(queue) + get_deep_size(explored) + get_deep_size(visited)
                )
            
        return None, visited, used_memory
    
    def value_has_converged(self, curr_array, next_array):
        return np.all(np.isclose(curr_array, next_array))
    
    def policy_has_converged(self, curr_array, next_array):
        return np.all(curr_array == next_array)
    
    def find_optimal_path(self, value_function):
        path = [self.start]
        while path[-1] != self.end:
            x, y = path[-1]
            possible_moves = []
            for dx, dy in self.moves:
                next_move = (x + dx, y + dy)
                if next_move == self.end:
                    path.append(next_move)
                    return path
                elif self.node_in_maze(next_move) and self.maze[next_move] == 0:
                    possible_moves.append(value_function[next_move])
                else:
                    possible_moves.append(self.wall_penalty)
            best_dx, best_dy = self.moves[np.argmax(possible_moves)]
            path.append((x + best_dx, y + best_dy))
        return path
    
    def define_transition_probabilities(self):
        transition_probabilities = np.zeros((self.maze.shape[0], self.maze.shape[1], len(self.moves)))
        for i, (dx, dy) in enumerate(self.moves):
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    next_x, next_y = x + dx, y + dy
                    if self.node_in_maze((next_x, next_y)) and self.maze[next_x, next_y] == 0:
                        transition_probabilities[x, y, i] = 1
        return transition_probabilities
    
    def makrov_value_iteration(self, max_iterations=1e6, gamma=0.9):

        # Initializations
        value_function = np.zeros(self.maze.shape)
        history = dict(value=[value_function]) if self.keep_history else None
        reward_function = np.zeros(self.maze.shape)
        reward_function[self.end] = self.goal_reward
        transition_probabilities = self.define_transition_probabilities()
        used_memory = 0 # Memory usage

        # Perform the value iteration
        iter = 0
        while iter < max_iterations:
            new_value_function = np.zeros(self.maze.shape)
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    if self.maze[x, y] == 1:
                        continue
                    if (x, y) == self.end:
                        new_value_function[x, y] = self.goal_reward
                    else:
                        possible_moves = []
                        for i, (dx, dy) in enumerate(self.moves):
                            next_move = (x + dx, y + dy)
                            if self.node_in_maze(next_move) and self.maze[next_move] == 0:
                                possible_moves.append(
                                    transition_probabilities[x, y, i] * (
                                        reward_function[next_move] + gamma * value_function[next_move]
                                    )
                                )
                            else:
                                possible_moves.append(self.wall_penalty)
                        new_value_function[x, y] = max(possible_moves)
            iter += 1
            if self.keep_history:
                history['value'].append(new_value_function)
            if self.value_has_converged(new_value_function, value_function):
                break
            else:
                value_function = new_value_function

        if iter > max_iterations:
            print(f"value iteration did not converge after {max_iterations} iterations")
            optimal_path = None
        else:
            optimal_path = self.find_optimal_path(value_function)

        if self.record_memory:
            used_memory = (
                get_deep_size(value_function) +
                get_deep_size(new_value_function) +
                get_deep_size(reward_function) +
                get_deep_size(transition_probabilities) +
                get_deep_size(optimal_path)
            )
        
        return value_function, optimal_path, history, used_memory
    
    def makrov_policy_iteration(self, max_iterations=1e6, gamma=0.9):

        # Initializations
        value_function = np.zeros(self.maze.shape)
        policy = [random.choices(self.moves, k=self.maze.shape[1]) for _ in range(self.maze.shape[0])]
        history = dict(value=[value_function], policy=[policy]) if self.keep_history else None
        reward_function = np.zeros(self.maze.shape)
        reward_function[self.end] = self.goal_reward
        transition_probabilities = self.define_transition_probabilities()
        used_memory = 0 # Memory usage

        # Perform the policy iteration
        iter = 0
        while iter < max_iterations:
            new_policy = [[self.moves[0] for _ in range(self.maze.shape[1])] for _ in range(self.maze.shape[0])]

            # Policy Evaluation
            while True:
                iter += 1
                new_value_function = np.zeros(self.maze.shape)
                for x in range(self.maze.shape[0]):
                    for y in range(self.maze.shape[1]):
                        if (x, y) == self.end:
                            new_value_function[x, y] = self.goal_reward
                        else:
                            dx, dy = policy[x][y]
                            next_move = (x + dx, y + dy)
                            if self.node_in_maze(next_move) and self.maze[next_move] == 0:
                                i = self.moves.index(policy[x][y])
                                new_value_function[x, y] = transition_probabilities[x, y, i] * (
                                    reward_function[next_move] + gamma * value_function[next_move]
                                )
                value_converged = self.value_has_converged(new_value_function, value_function)
                if self.keep_history:
                    history['value'].append(new_value_function)
                    if not value_converged:
                        history['policy'].append(policy)
                value_function = new_value_function
                if value_converged or iter > max_iterations:
                    break

            # Policy Improvement
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    possible_moves = []
                    for i, (dx, dy) in enumerate(self.moves):
                        next_move = (x + dx, y + dy)
                        if self.node_in_maze(next_move) and self.maze[next_move] == 0:
                            possible_moves.append(
                                transition_probabilities[x, y, i] * (
                                    reward_function[next_move] + gamma * new_value_function[next_move]
                                )
                            )
                        else:
                            possible_moves.append(self.wall_penalty)
                    
                    new_policy[x][y] = self.moves[np.argmax(possible_moves)]

            policy_converged = self.policy_has_converged(new_policy, policy)
            # Update the value function and the policy
            if self.keep_history:
                history['policy'].append(new_policy)
            value_function = new_value_function
            policy = new_policy
            # Check for convergence
            if policy_converged:
                break

        if iter > max_iterations:
            print(f"Policy iteration did not converge after {max_iterations} iterations")
            optimal_path = None
        else:
            optimal_path = self.find_optimal_path(value_function)

        if self.record_memory:
            used_memory = (
                get_deep_size(value_function) +
                get_deep_size(new_value_function) +
                get_deep_size(policy) +
                get_deep_size(new_policy) +
                get_deep_size(reward_function) +
                get_deep_size(transition_probabilities) +
                get_deep_size(optimal_path)
            )
            
        return policy, value_function, optimal_path, history, used_memory
        
