# Multi Agent Reinforcement Learning Project:
#### By Adam Lagssaibi, Edgard Dabier and Nolan Sisouphanthong

In this repository, we will explore various RL methods (**single agent** algorithms to learn the best policy, as well as more complex **multi agents** algorithms)

The file `single_agent_env.py` contains the basic code to load an environment, find the best policy using a **Value Iteration** algorithm and then simulate the best policy to visualize it.

The `Q_learning_Sarsa_single_agent.py` file implements a single agent environment as well as both **Q Learning** and **Sarsa** algorithms and compare them on the frozen lake base env with learning curves.

The `multi-agent-FrozenLake.ipynb` file implements a custom 2-agents version of open AI's FrozenLake game. It includes a **Central Q Learning** algortihm as well as a visual render of the game. In the `main` method, we can select the number of agents, the size of the grid, the seed for the random map generation, as well as predefined maps and other parameters.
