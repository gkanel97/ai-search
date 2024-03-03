# README

## Maze Solver

This program is a maze solver that generates a maze and solves it using various algorithms. It provides visualization options to display the search algorithm or value/policy iteration.

### Usage

1. Unzip the source code.

2. Open a terminal or command prompt and navigate to the project directory.

3. Run the following command to install the required packages:

     ```shell
     pip install -r requirements.txt
     ```

4. Run the main.py file to execute the maze solver:

     ```shell
     python main.py [--dim DIM] [--alg ALG] [--seed SEED] [--hist] [--anim] [--filename FILENAME]
     ```

     Optional arguments:
     - `--dim DIM`: Dimension of the maze (default: 20)
     - `--alg ALG`: Solver algorithm (default: bfs)
     - `--seed SEED`: Random seed (default: 17)
     - `--hist`: Show the history of the search algorithm
     - `--anim`: Enable animation
     - `--filename FILENAME`: Filename to save the figure

     Available solver algorithms: bfs, dfs, a_star, value_iteration, policy_iteration

5. The program will generate a maze, solve it using the specified algorithm, and display the visualization based on the chosen options.

6. Enjoy exploring the maze-solving algorithms!

### Examples

- Create a 50x50 maze:
    ```shell
    python main.py --dim 50
    ```

- Run all the maze solvers for a 10x10 maze with animation:

    ```shell
    python main.py --dim 10 --alg bfs --anim
    python main.py --dim 10 --alg dfs --anim
    python main.py --dim 10 --alg a_star --anim
    python main.py --dim 10 --alg value_iteration --anim
    python main.py --dim 10 --alg policy_iteration --anim
    ```

- Run the DFS maze solver for a 10x10 maze generated with a specific seed and show history of search algorithm:

    ```shell
    python main.py --dim 10 --alg dfs --seed 42 --hist
    ```

- Run the maze solver with value iteration algorithm and save the figure:

    ```shell
    python main.py --alg value_iteration --filename value_iteration.png
    ```

### Notes

- The search history can only be visualized for mazes with a dimension less than or equal to 23.

- Make sure to have the required packages installed before running the program.

- The generated maze and visualization will be displayed in a separate window.

- Press the 'X' button or close the window to exit the program.

- For more information, refer to the source code documentation.