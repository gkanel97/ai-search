import numpy as np
from queue import Queue

class MazeSolver:

    def __init__(self, maze):
        self.maze = maze
        self.start = (1, 0)
        self.end = (maze.shape[0] - 2, maze.shape[1] - 2)
        self.moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def node_in_maze(self, node):
        x, y = node
        return x >= 0 and y >= 0 and x < self.maze.shape[0] and y < self.maze.shape[1]
    
    def bfs(self):
        
        queue = Queue() # A queue to add the nodes to visit
        queue.put((self.start, [self.start]))
        explored = [self.start] # A list to store the visited nodes

        while not queue.empty():
            # Visit the next node in the queue
            (node, path) = queue.get()
            if node == self.end:
                return path, explored
            # Explore all the neighbours of the node
            for dx, dy in self.moves:
                next_node = (node[0]+dx, node[1]+dy)
                if (self.node_in_maze(next_node) and self.maze[next_node] == 0 and next_node not in explored):
                    explored.append(next_node)
                    queue.put((next_node, path + [next_node]))

        return None, explored
    
    def dfs(self):
        
        stack = [(self.start, [self.start])] # A stack to add the nodes to visit
        explored = [self.start] # A list to store the visited nodes
        
        while len(stack) > 0:
            # Visit the next node in the stack
            (node, path) = stack.pop()
            if node == self.end:
                return path, explored
            # Explore all the neighbours of the node
            for dx, dy in self.moves:
                next_node = (node[0]+dx, node[1]+dy)
                if (self.node_in_maze(next_node) and self.maze[next_node] == 0 and next_node not in explored):
                    explored.append(next_node)
                    stack.append((next_node, path + [next_node]))

        return None, explored
    
    def a_star(self):

        queue = [] # A priority queue to add the nodes to visit
        queue.append((0, self.start, [self.start]))
        explored = [self.start] # A list to store the visited nodes
        
        while len(queue) > 0:
         
            # Visit the next node in the queue
            (cost, node, path) = queue.pop(0)
            
            # Base case: if the current node is the end point, return
            if node == self.end:
                return path, explored
            
            # Visit all the neighbours of the node
            for dx, dy in self.moves:
                next_node = (node[0]+dx, node[1]+dy)
                if (self.node_in_maze(next_node) and self.maze[next_node] == 0 and next_node not in explored):
                    explored.append(next_node)
                    path_cost = len(path) # Number of steps taken so far
                    goal_proximity = abs(self.end[0] - next_node[0]) + abs(self.end[1] - next_node[1]) # Manhattan distance to the goal
                    next_cost = path_cost + goal_proximity
                    queue.append((next_cost, next_node, path + [next_node]))

            # Sort the queue by cost        
            queue.sort(key=lambda x: x[0])
            
        return None, explored
    
    def makrov_value_iteration(self, iterations=100, gamma=0.9):
        # Initialize the value function
        value_function = np.zeros(self.maze.shape)
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
        new_value_function = np.zeros(self.maze.shape)
        for _ in range(iterations):
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    if (x, y) == self.end:
                        new_value_function[x, y] = 1
                    else:
                        valid_moves = []
                        for i, (dx, dy) in enumerate(self.moves):
                            next_move = (x + dx, y + dy)
                            if self.node_in_maze(next_move) and self.maze[next_move] == 0:
                                valid_moves.append(
                                    transition_probabilities[x, y, i] * (
                                        reward_function[next_move] + gamma * value_function[next_move]
                                    )
                                )
                        if valid_moves:
                            new_value_function[x, y] = max(valid_moves)
                value_function = new_value_function
        return value_function
    
    # Update the value function based on the current policy
    def update_value_function(self, policy, value_function, reward_function, transition_probabilities, gamma):
        new_value_function = np.zeros(self.maze.shape)
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                if (x, y) == self.end:
                    new_value_function[x, y] = 1
                else:
                    next_move = (x + self.moves[int(policy[x, y])])
                    new_value_function[x, y] = (
                        reward_function[x, y] + gamma * value_function[next_move]
                    )
        return new_value_function
    
    def makrov_policy_iteration(self, iterations=100, gamma=0.9):
        # Initialize the value function
        value_function = np.zeros(self.maze.shape)
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
        policy = np.zeros(self.maze.shape)
        for _ in range(iterations):
            # Update the value function based on the current policy
            value_function = self.update_value_function(policy, value_function, reward_function, transition_probabilities, gamma)
            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    if (x, y) == self.end:
                        policy[x, y] = -1
                    else:
                        valid_moves = []
                        for i, (dx, dy) in enumerate(self.moves):
                            next_move = (x + dx, y + dy)
                            if self.node_in_maze(next_move) and self.maze[next_move] == 0:
                                valid_moves.append(
                                    transition_probabilities[x, y, i] * (
                                        reward_function[next_move] + gamma * value_function[next_move]
                                    )
                                )
                        if valid_moves:
                            policy[x, y] = np.argmax(valid_moves)
        return policy
