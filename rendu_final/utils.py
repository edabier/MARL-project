import os
import pickle
import datetime
import numpy as np
from environments import MAPS, FrozenLakeOneGoal, createMap
import matplotlib.pyplot as plt
import pygame
import time
from algorithms import IndependentQLearning,AlternatingIQL

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

def visualize_policy_pygame(env, agent, max_steps=100, delay=0.5, screen_size=600, save_images=False):
    """
    Visualise la politique d'un agent avec pygame
    
    Parameters:
    ----------
    env : l'environnement FrozenLake4goals
    agent : l'algorithme d'apprentissage (IQL, CQL, etc.)
    agent_idx : l'indice de l'agent à visualiser (pour IQL)
    max_steps : nombre maximum d'étapes par épisode
    delay : délai entre chaque étape (en secondes)
    screen_size : taille de la fenêtre pygame
    save_images : sauvegarder les images de chaque étape
    """
    # Réinitialiser l'environnement
    state_tuple, _ = env.reset()
    
    # Créer un dossier pour sauvegarder les images si nécessaire
    if save_images:
        img_dir = f"visu/policy_visualization_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(img_dir, exist_ok=True)
    
    # Initialiser pygame si nécessaire
    pygame.init()
    
    # Statistiques pour l'épisode
    total_reward = 0
    step_count = 0
    goals_reached = set()
    
    print(f"\n--- Début de la visualisation de la politique ---")
    
    # Rendre l'état initial
    env.render_pygame(screen_size=screen_size)
    if save_images:
        pygame.image.save(env.screen, f"{img_dir}/step_{step_count:03d}.png")
    
    print(f"État initial")
    
    # Boucle principale
    done = False
    # Désactiver complètement l'exploration
    if hasattr(agent, 'agents'):
        for a in agent.agents:
            a.epsilon = 0
    else:
        agent.epsilon = 0

    for step in range(max_steps):
        if all(env.agent_done):  # Si tous les agents ont terminé
            break
            
        actions = []
        for i in range(env.num_agents):
            if env.agent_done[i]:
                actions.append(0)  # Action factice pour les agents terminés
            elif isinstance(agent, (IndependentQLearning, AlternatingIQL)):
                # Obtenir l'action de l'agent pour l'état actuel
                if isinstance(agent, (IndependentQLearning, AlternatingIQL)):
                    # Pour IQL, chaque agent a sa propre table Q
                    state = state_tuple[i]
                    action = np.argmax(agent.agents[i].q_table[state])  # Strictement déterministe
                    actions.append(action)
                else:
                    # Pour CQL  utiliser la table Q centralisée
                    joint_state = state_tuple
                    action = agent.get_action(joint_state)[i]
                    actions.append(action)
            else:
                # Pour les autres agents  choisir l'action aléatoirement
                actions.append(np.random.randint(0, 4))
                print("random action")
        
        # Exécuter l'action
        next_state_tuple, rewards, dones, truncated, info = env.step(actions)
        
        # Mettre à jour les statistiques
        step_count += 1
        reward = sum(rewards)
        total_reward += reward
        
        # Vérifier les objectifs atteints
        if 'goals_reached' in info:
            current_goals = set(agent_id for agent_id, count in enumerate(info['goals_reached']) if count > 0)
            new_goals = current_goals - goals_reached
            if new_goals:
                print(f"Étape {step_count}: Agents {new_goals} ont atteint un objectif!")
            goals_reached = current_goals
        
        # Afficher les informations sur l'étape
        action_names = ['GAUCHE', 'BAS', 'DROITE', 'HAUT']
        print(f"Étape {step_count}: Actions={[action_names[a] for a in actions]}, "
              f"Récompense={reward}, Total={total_reward}")
        
        if info.get('collisions', False):
            print(f"⚠️ Collision entre les agents {info.get('collision_agents', [])}!")
        
        # Rendre l'état suivant
        env.render_pygame(screen_size=screen_size)
        if save_images:
            pygame.image.save(env.screen, f"{img_dir}/step_{step_count:03d}.png")
        
        # Mettre à jour l'état
        state_tuple = next_state_tuple
        
        # Attendre un peu pour que l'utilisateur puisse voir l'animation
        time.sleep(delay)
        
        # Vérifier si l'épisode est terminé
        if all(dones):
            print("Tous les agents ont terminé!")
            break
    
    print(f"\n--- Fin de la visualisation ---")
    print(f"Total de {step_count} étapes")
    print(f"Récompense totale: {total_reward}")
    print(f"Objectifs atteints par les agents: {goals_reached}")
    
    # Attendre que l'utilisateur ferme la fenêtre pygame
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    # Si save_images est activé, informer l'utilisateur
    if save_images:
        print(f"\nImages sauvegardées dans le dossier: {img_dir}")
    
    pygame.quit()


def launch_visualization(agent, algo_type, num_agents=2, steps=100, delay=0.5, save_images=False):
    """
    Lance la visualisation dans un terminal séparé
    
    Parameters:
    ----------
    agent : l'agent entraîné
    algo_type : type d'algorithme ('iql', 'cql', 'alt_iql')
    num_agents : nombre d'agents
    steps : nombre maximum d'étapes
    delay : délai entre les étapes
    save_images : sauvegarder les images de chaque étape
    """
    import os
    import subprocess
    import sys
    import pickle
    import tempfile
    
    # Créer un fichier temporaire pour sauvegarder l'agent
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, f"temp_agent_{algo_type}.pkl")
    
    # Sauvegarder l'agent
    with open(model_path, 'wb') as f:
        pickle.dump(agent, f)
    
    # Construire la commande
    cmd = [
        sys.executable,
        "run_visualization.py",
        "--algo", algo_type,
        "--model", model_path,
        "--agents", str(num_agents),
        "--steps", str(steps),
        "--delay", str(delay)
    ]
    
    if save_images:
        cmd.append("--save")
    
    # Lancer le processus selon le système d'exploitation
    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(["start", "cmd", "/k"] + [" ".join(cmd)], shell=True)
        elif sys.platform == 'darwin':  # MacOS
            cmd_str = " ".join(cmd)
            subprocess.Popen(["open", "-a", "Terminal", cmd_str], shell=True)
        else:  # Linux
            cmd_str = " ".join(cmd)
            subprocess.Popen(["gnome-terminal", "--", "bash", "-c", f"{cmd_str}; exec bash"], shell=False)
        
        print(f"Visualisation de l'agent {algo_type} lancée dans un terminal séparé.")
        print(f"L'agent a été temporairement sauvegardé à: {model_path}")
    except Exception as e:
        print(f"Erreur lors du lancement de la visualisation: {e}")

        
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