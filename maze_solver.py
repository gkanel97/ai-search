import numpy as np
from queue import Queue

class MazeSolver:

    def __init__(self, maze):
        self.maze = maze
        self.start = (1, 0)
        self.end = (maze.shape[0] - 2, maze.shape[1] - 1)
        self.moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def node_in_maze(self, node):
        x, y = node
        return x >= 0 and y >= 0 and x < self.maze.shape[0] and y < self.maze.shape[1]
    
    def bfs(self):
        
        queue = Queue() # A queue to add the nodes to visit
        queue.put((self.start, [self.start]))
        explored = set() # A set to store explored nodes
        explored.add(self.start)
        visited = [] # A list to store the visited nodes (for visualisation purposes)

        while not queue.empty():
            # Visit the next node in the queue
            (node, path) = queue.get()
            visited.append(node)
            if node == self.end:
                return path, visited
            # Explore all the neighbours of the node
            for dx, dy in self.moves:
                next_node = (node[0]+dx, node[1]+dy)
                if (self.node_in_maze(next_node) and self.maze[next_node] == 0 and next_node not in explored):
                    explored.add(next_node)
                    queue.put((next_node, path + [next_node]))

        return None, visited
    
    def dfs(self):
        
        stack = [(self.start, [self.start])] # A stack to add the nodes to visit
        explored = set()
        explored.add(self.start) 
        visited = [] # A list to store the visited nodes (for visualisation purposes)
        
        while len(stack) > 0:
            # Visit the next node in the stack
            (node, path) = stack.pop()
            visited.append(node)
            if node == self.end:
                return path, visited
            # Explore all the neighbours of the node
            for dx, dy in self.moves:
                next_node = (node[0]+dx, node[1]+dy)
                if (self.node_in_maze(next_node) and self.maze[next_node] == 0 and next_node not in explored):
                    explored.add(next_node)
                    stack.append((next_node, path + [next_node]))

        return None, visited
    
    def a_star(self):

        queue = [] # A priority queue to add the nodes to visit
        queue.append((0, self.start, [self.start]))
        explored = set() # A set to store the explored nodes
        explored.add(self.start)
        visited = [] # A list to store the visited nodes (for visualisation purposes)
        
        while len(queue) > 0:
         
            # Visit the next node in the queue
            (cost, node, path) = queue.pop(0)
            visited.append(node)
            if node == self.end:
                return path, visited
            
            # Visit all the neighbours of the node
            for dx, dy in self.moves:
                next_node = (node[0]+dx, node[1]+dy)
                if (self.node_in_maze(next_node) and self.maze[next_node] == 0 and next_node not in explored):
                    explored.add(next_node)
                    path_cost = len(path) # Number of steps taken so far
                    goal_proximity = abs(self.end[0] - next_node[0]) + abs(self.end[1] - next_node[1]) # Manhattan distance to the goal
                    next_cost = path_cost + goal_proximity
                    queue.append((next_cost, next_node, path + [next_node]))

            # Sort the queue by cost        
            queue.sort(key=lambda x: x[0])
            
        return None, visited
    
    def makrov_value_iteration(self, iterations=100, gamma=0.9):

        # Initialize the value function
        value_function = np.zeros(self.maze.shape)
        policy = [[self.moves[0] for _ in range(self.maze.shape[1])] for _ in range(self.maze.shape[0])]
        history = dict(value=[value_function], policy=[policy])

        # Define the reward function
        reward_function = np.zeros(self.maze.shape)
        reward_function[self.end] = 1
        # Define the transition probabilities
        transition_probabilities = np.zeros((self.maze.shape[0], self.maze.shape[1], len(self.moves)))
        for i, (dx, dy) in enumerate(self.moves):
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    next_x, next_y = x + dx, y + dy
                    if self.node_in_maze((next_x, next_y)) and self.maze[next_x, next_y] == 0:
                        transition_probabilities[x, y, i] = 1

        # Perform the value iteration
        for _ in range(iterations):
            new_value_function = np.zeros(self.maze.shape)
            new_policy = [[self.moves[0] for _ in range(self.maze.shape[1])] for _ in range(self.maze.shape[0])]
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    if self.maze[x, y] == 1:
                        continue
                    if (x, y) == self.end:
                        new_value_function[x, y] = 1
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
                                possible_moves.append(0)
                        new_value_function[x, y] = max(possible_moves)
                        new_policy[x][y] = self.moves[np.argmax(possible_moves)]
            history['value'].append(new_value_function)
            history['policy'].append(new_policy)
            value_function = new_value_function
            policy = new_policy
        return value_function, history
    
    def makrov_policy_iteration(self, iterations=100, gamma=0.9):

        def has_converged(policy_history, n):
            # If there are less than n policies, return False
            if len(policy_history) < n:
                return False
            # Check if the last n policies are the same
            return all(policy == policy_history[-1] for policy in policy_history[-n:])

        # Initialize the value function
        value_function = np.zeros(self.maze.shape)
        # Initialize the policy function
        policy = [[self.moves[0] for _ in range(self.maze.shape[1])] for _ in range(self.maze.shape[0])]
        history = dict(value=[value_function], policy=[policy])
        # Define the reward function
        reward_function = np.zeros(self.maze.shape)
        reward_function[self.end] = 1
        # Define the transition probabilities
        transition_probabilities = np.zeros((self.maze.shape[0], self.maze.shape[1], len(self.moves)))
        for i, (dx, dy) in enumerate(self.moves):
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    next_x, next_y = x + dx, y + dy
                    if self.node_in_maze((next_x, next_y)) and self.maze[next_x, next_y] == 0:
                        transition_probabilities[x, y, i] = 1

        # Perform the policy iteration
        for _ in range(iterations):
            new_value_function = np.zeros(self.maze.shape)
            new_policy = [[self.moves[0] for _ in range(self.maze.shape[1])] for _ in range(self.maze.shape[0])]

            # Policy Evaluation
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    if (x, y) == self.end:
                        new_value_function[x, y] = 1
                    else:
                        dx, dy = policy[x][y]
                        next_move = (x + dx, y + dy)
                        if self.node_in_maze(next_move) and self.maze[next_move] == 0:
                            i = self.moves.index(policy[x][y])
                            new_value_function[x, y] = transition_probabilities[x, y, i] * (
                                reward_function[next_move] + gamma * value_function[next_move]
                            )

            # Policy Improvement
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    old_action = policy[x][y]
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
                            possible_moves.append(0)
                    new_policy[x][y] = self.moves[np.argmax(possible_moves)]

            # Update the value function and the policy
            history['value'].append(new_value_function)
            history['policy'].append(new_policy)
            value_function = new_value_function
            policy = new_policy

            # Check for convergence
            if has_converged(history['policy'], 5):
                break
        return policy, value_function, history
        
