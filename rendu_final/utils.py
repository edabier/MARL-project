import os
import pickle
import datetime
import numpy as np
from environments import MAPS, FrozenLakeOneGoal, createMap
import matplotlib.pyplot as plt

def save_agent(agent, agent_type, env_info=None, save_dir="saved_agents"):
    """
    Sauvegarde un agent d'apprentissage par renforcement.
    
    Args:
        agent: L'agent à sauvegarder (IQL, AlternatingIQL, etc.)
        agent_type: String identifiant le type d'agent ("iql", "alt_iql", etc.)
        env_info: Informations sur l'environnement (optionnel)
        save_dir: Répertoire où sauvegarder les agents
    
    Returns:
        String: Le chemin où l'agent a été sauvegardé
    """
    # Créer le répertoire s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Générer un nom de fichier avec horodatage
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Information sur le nombre d'agents
    n_agents = getattr(agent, 'n_agents', None)
    agent_suffix = f"_{n_agents}agents" if n_agents else ""
    
    filename = f"{agent_type}{agent_suffix}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)
    
    # Créer un dictionnaire avec l'agent et les informations d'environnement
    save_data = {
        'agent': agent,
        'env_info': env_info
    }
    
    # Sauvegarder l'agent
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Agent sauvegardé dans: {filepath}")
    return filepath

def load_agent(filepath):
    """
    Charge un agent précédemment sauvegardé.
    
    Args:
        filepath: Chemin vers le fichier pickle de l'agent
    
    Returns:
        tuple: (agent, env_info) - L'agent chargé et les infos d'environnement
    """
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"Agent chargé depuis: {filepath}")
    return save_data['agent'], save_data['env_info']

def list_saved_agents(save_dir="saved_agents"):
    """
    Liste tous les agents sauvegardés.
    
    Args:
        save_dir: Répertoire où chercher les agents
    
    Returns:
        list: Liste des chemins des fichiers d'agents
    """
    if not os.path.exists(save_dir):
        print(f"Le répertoire {save_dir} n'existe pas.")
        return []
    
    agent_files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
    
    if not agent_files:
        print(f"Aucun agent sauvegardé trouvé dans {save_dir}.")
        return []
    
    print("Agents sauvegardés:")
    for i, file in enumerate(agent_files):
        print(f"{i+1}. {file}")
    
    return [os.path.join(save_dir, f) for f in agent_files]

def run_simulation(agent, map_, num_agent, num_episodes=10000, silent=True):
    # Create environment
    env = FrozenLakeOneGoal(map_=map_, max_steps=100, num_agents=num_agent)
    
    # Tracking metrics
    episode_rewards = []
    success_rate = []
    success_window = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        step = 0
        
        # Run episode
        while not done and not truncated:
            # Select action
            action = agent.select_action(state)
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            # Update state and total reward
            state = next_state
            total_reward += reward
            step += 1
        
        # Record episode success/failure
        success = total_reward > 0.5
        success_window.append(success)
        if len(success_window) > 200:
            success_window.pop(0)
        
        # Calculate success rate over last 100 episodes
        current_success_rate = sum(success_window) / len(success_window)
        success_rate.append(current_success_rate)
        
        # Record total reward
        # mean_reward = total_reward / step if step > 0 else 0
        mean_reward = total_reward
        episode_rewards.append(mean_reward)
        
        # Print progress
        if not silent and episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {mean_reward}, Success Rate: {current_success_rate:.2f}, Epsilon: {agent.epsilon:.4f}")
    
    window_size = 500
    mean_rewards_smooth = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    
    # Plot learning curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mean_rewards_smooth)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=2)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(success_rate)
    plt.title('Success Rate (500-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    
    plt.tight_layout()
    plt.show()
    
    return agent