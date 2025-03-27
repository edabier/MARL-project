# Multi Agent Reinforcement Learning Project:
#### By Adam Lagssaibi, Edgard Dabier and Nolan Sisouphanthong

In this repository, we explore various RL methods (**single agent** algorithms to learn the best policy, as well as more complex **multi agents** algorithms), using custom versions of OpenAi's [ **FrozenLake** environment](https://github.com/openai/gym/tree/master/gym/envs).

## MARL Class:

This project's goal was to give a 30mn introduction class on multi-agent reinforcement learning. The code in this repository is used for illustration purposes, and the course material can be found in the `MARL_Class_latex` folder.

## Project implementation details:

The most important files are located in the `main-code` folder and are split into 4 files:

- `environments.py`: defines the two types of MultiAgent FrozenLake environments
- `algorithms.py`: defines the actual multi-agent learning algorithms (`IndependentQLearning`, `CentralizedQLearning`, `AlternatingIQL`, `CentralQLearningCommonGoal`)
- `utils.py`: defines some visualization functions
- `main.ipynb`: this notebook runs the whole process of single and multi-agent learning algorithms and visualize the learned policies.

## Requirements:

Ton run the code properly, we recommand installing the following libraries:

```
numpy
matplotlib
pygame
gymnasium
tqdm
```

You can also run `pip3 install -r requirements.txt`
