Marl presentation

- What is Reinforcement Learning?
RL is part of ML but is not supervised or unsupervised, and it deals with « games »
We have an agent, inside an environment that it can interact with, and the goal is to have it learn the best interactions to reach a specific goal.

For example, in frozen lake, the agent can move around (we call that an **action**) in an environment that is made of holes, ice paths, and a destination spot, and the goal is to have the agent reach the end.

With this simple example, we can observe a central property in RL which is the **markovianity** of the action chain. The probability of the agent being in a new state and receiving some reward only depends on its previous state and what action it takes, but not all the previous actions.

- how do we teach the agent what is a good or a bad decision?
We use **reward functions**: depending on what the agent does on a given state (if it is at a pivot spot, next to a hole, a blank ice path, and at the end of the game for example) his action will have a strong impact on the outcome and we can design a function rewarding it or punishing it if its action has a positive impact on the objective.

Now that an agent knows what is a good or a bad action, we can let it interact with its environment. But we have to set limits, otherwise it would last forever. So we consider **final states**, that if the agent reaches them, the game is over, as well as **maximum time steps** (that time limit) that if the agent doesn’t reach the objective within this amount of time steps, the game is over. 
Then at the end of the game, we can sum all the rewards cumulated by the agent to see what its global performance was. 

*Optional: Evaluating the agent on its final cumulated reward can be tricky, because since we set a time limit to its exploration, the agent could cumulate small rewards, but we might want it to reach as quickly as possible the objective, and thus give a higher weight to the first actions, than the last ones. To do this, we introduce a **discount factor gamma** by which to multiply actions to favor the first ones.*

In the example of FrozenLake, the principle is that the agent is moving on ice blocks, which means that they can be slippery and thus lead to uncertainty in the new state of the agent. The agent can decide to go down, but it slips and finds itself on the block to its right. Using this example we can see that the reward obtained by the agent when taking a certain action is **not certain**, if we knew exactly where it would land, then we would know the exact value of its reward, but since it’s uncertain, we can also approximate the **expected reward**.

With this reward function, we can define a function name **value function** that will inform the agent on its expected return (the expectation of its reward, because rewards can be stochastic) for its state s and when following a **policy** $$\pi$$, and this function can later be used to make the agent learn what actions to take.

So to recap, we want the agent to learn what to do in a given situation (we call this the **policy** of the agent), and it learns it based on a reward function.

(These are policy-based methods, but we could also be wanting our agent to learn the value function which informs it on its expected return by taking any action on any given state, and this is a value-based method (the policy derived from the choice of highest return given the value function).)

- What happens if the agent can’t observe the entire environment?
- How do we know in advance what the reward associated with an action is?

In practice, here are 1/2 common methods to find the best value function or the best policy:
- Value Iteration
	
- Q-Learning

So there are several approaches to make the agent learn how to reach its goal most efficiently, but how can we compare the learning methods?
A learning algorithm has to be able to **learn the best policy** (reach the maximum expected return - that is to converge to a single optimal policy), and we might want it to learn it as fast as possible.
To compare these criteria between several algorithms, we use what is called **Learning Curves**

We can now add another agent in our environment, and introduce the MARL environment.
There are many types of games, but we will mainly focus on a cooperative common reward game. In the FrozenLake example, if there are 2 agents, we can choose that their objective is to both reach the goal to win (if only one reaches it, it’s a loss).

Since we know how to handle single agents setups, we might want to simplify this 2 agent setup to an equivalent single-agent version, and for that, we have 2 options:
CQL -> curse of dimensionality 
IQL -> how to distribute it? What every agent knows (every other agent's state or just his state)? Leads to a choice of reward (common or general)
The problem of non-stationarity -> Try alternating learning rate

Finish by presenting the one we managed to implement.
