import gym
import time
import mdptoolbox
import matplotlib.pyplot as plt
import numpy as np

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')

num_runs = 10
max_iterations = 100

# Store the value function and policy changes for each run
value_function_history = []
policy_history = []

# Extract the number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize the transition probabilities and rewards matrices
P = np.zeros((n_actions, n_states, n_states))
R = np.zeros((n_states, n_actions))

# Populate the transition probabilities and rewards matrices
for state in range(n_states):
    for action in range(n_actions):
        transitions = env.P[state][action]
        for prob, next_state, reward, done in transitions:
            P[action][state][next_state] = prob
            R[state][action] += prob * reward

# Value Iteration
for run in range(num_runs):
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9, max_iter=max_iterations)
    vi.run()
    value_function_history.append(vi.V)
    policy_history.append(vi.policy)

# Compute the average value function and policy
average_value_function = np.mean(value_function_history, axis=0)
average_policy = np.mean(policy_history, axis=0)

# Print the results
print("Optimal Policy:", vi.policy)
print("Value Function:", vi.V)

# Simulate the best policy
def simulate_policy(env, policy, max_steps=100):
    state = env.reset()
    state = state[0]  # Unwrap the state from the tuple returned by reset()
    for _ in range(max_steps):
        env.render()
        action = policy[state]
        state, reward, done, _, _ = env.step(action)
        if done:
            break
        time.sleep(0.5)  # Add a delay to make the simulation easier to follow
    env.close()

# Run the simulation
simulate_policy(env, vi.policy)

def display_learning_curves(average_value_function, average_policy):
    # Plot the learning curves
    plt.figure(figsize=(12, 6))

    # Plot the value function changes
    plt.subplot(1, 2, 1)
    for i, V in enumerate(average_value_function):
        plt.plot(V, label=f'Iteration {i}')
    plt.title('Value Function Changes')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

    # Plot the policy changes
    plt.subplot(1, 2, 2)
    for i, policy in enumerate(average_policy):
        plt.plot(policy, label=f'Iteration {i}')
    plt.title('Policy Changes')
    plt.xlabel('State')
    plt.ylabel('Action')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return

# display_learning_curves(average_value_function, average_policy)