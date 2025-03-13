import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import os
import json
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm





def evaluate_agent(Q_table,env, num_episodes=200):
    
    successes = 0  # Count successful episodes
    episodes_length_list=[]
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episodes_length=0
        while not done:
            action = np.argmax(Q_table[state])  # Always pick the best action (Greedy)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            done = terminated or truncated
            episodes_length+=1
            if terminated and reward == 1:  # If agent reaches the goal
                successes += 1
        episodes_length_list.append(episodes_length)

    env.close()
    
    # Compute success rate
    success_rate = successes / num_episodes
    # print(f"Success Rate {success_rate * 100:.2f}%")
    return success_rate,np.mean(episodes_length_list)

# Your existing algorithm implementations
def Q_learning(env,alpha=0.1,gamma=0.99,max_len_episode=200,number_of_episodes=5000):
    
    
    epsilon_start = 1.0  # Start with full exploration
    epsilon_min = 0.1  # Minimum exploration rate
    epsilon_decay = 0.0001  # Step-size of decay could be linear or exponential

    
    total_rewards=0
    total_time_steps=0

    # env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
    n_observations = env.observation_space.n
    n_actions = env.action_space.n

    mean_episodes_length_progression=[]
    rewards_progression=[]
    time_steps_progression = []
    #init Q_table in a 4.4 16 observation/position and 4 actions UP/DOWN/LEFT/RIGHT
    Q_table = np.zeros((n_observations,n_actions))
    for episode in range(1,number_of_episodes+1):
        current_state, info = env.reset() #init the env force to do
        epsilon = max(epsilon_min, epsilon_start * np.exp(-epsilon_decay * episode)) # exponential decay 
        # epsilon = max(epsilon_min, epsilon_start - epsilon_decay * episode) # linear decay 
        for t in range(max_len_episode):
            total_time_steps += 1 
            if random.uniform(0, 1) < epsilon:  # Explore
                action = np.random.choice([0, 1, 2, 3]) 
            else:
                Q_values=Q_table[current_state]
                action = np.argmax(Q_values)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_rewards+=reward #check the sucesss rate of all runs
            
            Q_table[current_state,action]=Q_table[current_state,action]+ alpha*(reward + gamma* np.max(Q_table[next_state])-Q_table[current_state,action])
            if total_time_steps % 10000 == 0:
                success_rate,mean_episodes_length=evaluate_agent(Q_table,env)
                rewards_progression.append(success_rate)
                mean_episodes_length_progression.append(mean_episodes_length)
                time_steps_progression.append(total_time_steps) 

            # should terminate if terminated adn implement episode length 
            if terminated==True:
                break # go to anew of the episode 

            current_state=next_state

            #evaluation every 500 episode of the sucess rate 
            # if episode%500==0:
            
        

    # print(total_rewards,total_time_steps)  
    return Q_table , rewards_progression,mean_episodes_length_progression,time_steps_progression



def Sarsa(env,alpha=0.1,gamma=0.99,max_len_episode=200,number_of_episodes=5000):
    
    
    epsilon_start = 1.0  # Start with full exploration
    epsilon_min = 0.1  # Minimum exploration rate
    epsilon_decay = 0.0001  # Step-size of decay could be linear or exponential

    
    total_rewards=0
    total_time_steps=0

    # env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)
    n_observations = env.observation_space.n
    n_actions = env.action_space.n

    mean_episodes_length_progression=[]
    rewards_progression=[]
    time_steps_progression=[]
    #init Q_table in a 4.4 16 observation/position and 4 actions UP/DOWN/LEFT/RIGHT
    Q_table = np.zeros((n_observations,n_actions))
    for episode in range(1,number_of_episodes+1):
        current_state, info = env.reset() #init the env force to do
        epsilon = max(epsilon_min, epsilon_start * np.exp(-epsilon_decay * episode))  # exponential decay 
        # epsilon = max(epsilon_min, epsilon_start - epsilon_decay * episode) # linear decay 
        if random.uniform(0, 1) < epsilon:  # Explore
                action = np.random.choice([0, 1, 2, 3]) 
        else:
                Q_values=Q_table[current_state]
                action = np.argmax(Q_values)
            
        for t in range(max_len_episode):
            total_time_steps+=1
            next_state, reward, terminated, truncated, info = env.step(action)
            if random.uniform(0, 1) < epsilon:  # Explore
                next_action = np.random.choice([0, 1, 2, 3]) 
            else:
                Q_values=Q_table[next_state]
                next_action = np.argmax(Q_values)
            
            total_rewards+=reward #check the sucesss rate of all runs
            
            Q_table[current_state,action]=Q_table[current_state,action]+ alpha*(reward + gamma* Q_table[next_state,next_action]-Q_table[current_state,action])
            # should terminate if terminated adn implement episode length 
            
        #evaluation every 500 episode of the sucess rate 
        
        # if episode%500==0:
        
            
            if total_time_steps%10000==0:
                success_rate,mean_episodes_length=evaluate_agent(Q_table,env)
                rewards_progression.append(success_rate)
                mean_episodes_length_progression.append(mean_episodes_length)
                time_steps_progression.append(total_time_steps)

            if terminated==True:
                
                break # go to anew of the episode 
            current_state=next_state
            action=next_action 


    # print(total_rewards,total_time_steps)  
    return Q_table , rewards_progression,mean_episodes_length_progression,time_steps_progression

# Experiment running functionality
# experiment_runner.py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from itertools import product

# def run_experiment(algorithm, params, env, num_runs=5):
#     """Run algorithm multiple times with given parameters"""
#     all_rewards = []
#     all_lengths = []
#     all_time_steps = []

    
#     for _ in range(num_runs):
#         if algorithm.__name__ == "Q_learning":
#             Q_table, rewards, lengths ,time_steps= algorithm(env, **params)
#         else:  # SARSA
#             Q_table, rewards, lengths ,time_steps= algorithm(env, **params)
#         all_rewards.append(rewards)
#         all_lengths.append(lengths)
#         all_time_steps.append(time_steps)

#     all_rewards = np.array(all_rewards)
#     all_lengths = np.array(all_lengths)
#     all_time_steps = np.array(all_time_steps)
    
#     return {
#         'mean_rewards': np.mean(all_rewards, axis=0),
#         'std_rewards': np.std(all_rewards, axis=0),
#         'mean_lengths': np.mean(all_lengths, axis=0),
#         'std_lengths': np.std(all_lengths, axis=0),
#         'times_steps' : np.max(all_time_steps)
#     }

# def plot_comparison(results, params, save_path):
#     """Plot and save comparison results"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
#     episodes = [i*500 for i in range(len(results['Q_learning']['mean_rewards']))]
#     time_steps= results['Q_learning']['time_steps']
#     for algo in ['Q_learning', 'SARSA']:
#         ax1.plot(time_steps, results[algo]['mean_rewards'], 
#                 label=f"{algo} (α={params[algo]['alpha']}, γ={params[algo]['gamma']})")
#         ax1.fill_between(time_steps, 
#                         results[algo]['mean_rewards'] - results[algo]['std_rewards'],
#                         results[algo]['mean_rewards'] + results[algo]['std_rewards'],
#                         alpha=0.2)
    
#     ax1.set_xlabel("titme steps")
#     ax1.set_ylabel("Success Rate (%)")
#     ax1.set_title("Success Rate Progression")
#     ax1.legend()
#     ax1.grid(True)
    
#     for algo in ['Q_learning', 'SARSA']:
#             ax2.plot(time_steps, results[algo]['mean_lengths'],
#                     label=f"{algo} (α={params[algo]['alpha']}, γ={params[algo]['gamma']})")
#             ax2.fill_between(time_steps,
#                             results[algo]['mean_lengths'] - results[algo]['std_lengths'],
#                             results[algo]['mean_lengths'] + results[algo]['std_lengths'],
#                             alpha=0.2)
        
#     ax2.set_xlabel("time steps")
#     ax2.set_ylabel("Mean Episode Length")
#     ax2.set_title("Mean Episode Length Progression")
#     ax2.legend()
#     ax2.grid(True)    
#     plt.suptitle(f"Q-Learning vs SARSA Performance - {datetime.now().strftime('%Y-%m-%d_%H-%M')}")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f"comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.png"))
#     plt.close()

def run_experiment(algorithm, params, env, num_runs=5):
    """Run algorithm multiple times with given parameters"""
    all_rewards = []
    all_lengths = []
    all_time_steps = []
    
    pbar = tqdm(range(num_runs), desc=f"{algorithm.__name__} runs", leave=False)
    for run in pbar:
        if algorithm.__name__ == "Q_learning":
            Q_table, rewards, lengths, time_steps = algorithm(env, **params)
        else:  # SARSA
            Q_table, rewards, lengths, time_steps = algorithm(env, **params)
        
        all_rewards.append(rewards)
        all_lengths.append(lengths)
        all_time_steps.append(time_steps)
        
        pbar.set_postfix({'evaluations': len(rewards)})
    
    ## Find the shortest length among all runs
    min_length = min(len(r) for r in all_rewards)
    
    # Trim all arrays to the same length
    all_rewards = [r[:min_length] for r in all_rewards]
    all_lengths = [l[:min_length] for l in all_lengths]
    time_steps = all_time_steps[0][:min_length]  # Use first run's time steps
    
    # Convert to numpy arrays after trimming
    all_rewards = np.array(all_rewards)
    all_lengths = np.array(all_lengths)
    return {
        'mean_rewards': np.mean(all_rewards, axis=0),
        'std_rewards': np.std(all_rewards, axis=0),
        'mean_lengths': np.mean(all_lengths, axis=0),
        'std_lengths': np.std(all_lengths, axis=0),
        'time_steps': time_steps  # Use actual time steps from evaluations
    }

def plot_comparison(results, params, save_path):
    """Plot and save comparison results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for algo in ['Q_learning', 'SARSA']:
        # Plot using the actual evaluation time steps
        ax1.plot(results[algo]['time_steps'], 
                results[algo]['mean_rewards'], 
                label=f"{algo} (α={params[algo]['alpha']}, γ={params[algo]['gamma']})")
        ax1.fill_between(results[algo]['time_steps'],
                        results[algo]['mean_rewards'] - results[algo]['std_rewards'],
                        results[algo]['mean_rewards'] + results[algo]['std_rewards'],
                        alpha=0.2)
    
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Success Rate Progression")
    ax1.legend()
    ax1.grid(True)
    
    for algo in ['Q_learning', 'SARSA']:
        ax2.plot(results[algo]['time_steps'],
                results[algo]['mean_lengths'],
                label=f"{algo} (α={params[algo]['alpha']}, γ={params[algo]['gamma']})")
        ax2.fill_between(results[algo]['time_steps'],
                        results[algo]['mean_lengths'] - results[algo]['std_lengths'],
                        results[algo]['mean_lengths'] + results[algo]['std_lengths'],
                        alpha=0.2)
    
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Mean Episode Length")
    ax2.set_title("Mean Episode Length Progression")
    ax2.legend()
    ax2.grid(True)    
    plt.suptitle(f"Q-Learning vs SARSA Performance - {datetime.now().strftime('%Y-%m-%d_%H-%M')}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.png"))
    plt.close()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    # Initialize environment
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=6,seed=42), is_slippery=True)
    
    # Create results directory
    results_dir = "results/plots"
    os.makedirs(results_dir, exist_ok=True)
    # Define parameters to test
    param_grid = {
        'alpha': [0.1, 0.05,0.2,0.15],
        'gamma': [0.99],
        'max_len_episode': [100],
        'number_of_episodes': [20000]
    }
    
    # Run experiments and save results
    param_combinations = list(product(*param_grid.values()))
    total_experiments = len(param_combinations)

    main_pbar = tqdm(param_combinations, desc="Parameter combinations", total=total_experiments)
    for params in main_pbar:
        param_dict = dict(zip(param_grid.keys(), params))
        
        # Instead of print, use tqdm.write or set_postfix
        main_pbar.set_postfix({'alpha': param_dict['alpha']})
        tqdm.write(f"\nRunning experiments with parameters: {param_dict}")
        
        results = {
            'Q_learning': run_experiment(Q_learning, param_dict, env),
            'SARSA': run_experiment(Sarsa, param_dict, env)
        }
        # Save results
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        experiment_name = f"experiment_{timestamp}"
        
        # Save plots
        plot_comparison(results, {'Q_learning': param_dict, 'SARSA': param_dict}, results_dir)
        
        # Save numerical results
        with open(os.path.join(results_dir, f"{experiment_name}_results.json"), 'w') as f:
            json.dump({
                'parameters': param_dict,
                'results': results
            }, f, cls=NumpyEncoder)
    
    env.close()