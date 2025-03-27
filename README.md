# Multi Agent Reinforcement Learning Project:
#### By Adam Lagssaibi, Edgard Dabier and Nolan Sisouphanthong

In this repository, we explore various RL methods (**single agent** algorithms to learn the best policy, as well as more complex **multi agents** algorithms)

The most important files are located in the `main-code` folder and are split into 4 files:

- `environments.py`: defines the two types of MultiAgent FrozenLake environments
- `algorithms.py`: defines the actual multi-agent learning algorithms (`IndependentQLearning`, `CentralizedQLearning`, `AlternatingIQL`, `CentralQLearningCommonGoal`)
- `utils.py`: defines some visualization functions
- `main.ipynb`: this notebook runs the whole process of single and multi-agent learning algorithms and visualize the learned policies.

Ton run the code properly, we recommand installing the following libraries:

```
numpy
matplotlib
pygame
gymnasium
tqdm
```

You can also run `pip3 install -r requirements.txt`
