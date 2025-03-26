import os
import pickle
import datetime
import numpy as np
from environments import MAPS, FrozenLakeOneGoal, createMap
import matplotlib.pyplot as plt
import pygame
import time
from algorithms import IndependentQLearning,AlternatingIQL,QAgent

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
        "run_policy.py",
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

def plot_results(agent,results,windows=200):
    if isinstance(agent, QAgent):
        plt.figure(figsize=(15, 5))
        
        # Utiliser la clé 'rewards' comme indiqué dans votre fonction train
        rewards = results['rewards']
        rewards_smoothed = []
        
        # Calculer les moyennes glissantes
        for j in range(0, len(rewards), windows):
            if j + windows <= len(rewards):
                rewards_smoothed.append(np.mean(rewards[j:j+windows]))
        
        # Tracer la courbe lissée
        plt.plot(range(0, len(rewards_smoothed) * windows, windows), rewards_smoothed)
        plt.title(f"Récompenses moyennes (moyenne sur {windows} épisodes)")
        plt.xlabel("Épisodes")
        plt.ylabel("Récompenses moyennes")
        plt.show()
        return
    elif isinstance(agent, IndependentQLearning):
        plt.figure(figsize=(15, 5*agent.n_agents))

        # Tracer les récompenses pour chaque agent
    else:
        plt.figure(figsize=(15, 5*agent.n_agents+1))
        
    for i in range(agent.n_agents):
        plt.subplot(agent.n_agents +isinstance(agent,AlternatingIQL), 1, i+1)

        # Calculer les récompenses moyennes par fenêtre
        rewards_rates = results['rewards_rates'][i]
        rewards_rates_smoothed = []

        for j in range(0, len(rewards_rates), windows):
            if j + windows <= len(rewards_rates):
                rewards_rates_smoothed.append(np.mean(rewards_rates[j:j+windows]))

        plt.plot(range(0, len(rewards_rates_smoothed) * windows, windows), rewards_rates_smoothed)
        plt.title(f"Agent {i+1} - Récompenses moyennes (moyenne sur {windows} épisodes)")
        plt.xlabel("Épisodes")
        plt.ylabel("Récompenses moyennes")

    if not isinstance(agent, IndependentQLearning):
        plt.subplot(agent.n_agents + 1, 1, agent.n_agents + 1)
        
        for i in range(agent.n_agents):
            lr_rates = results['learning_rates'][i]
            
            # Sous-échantillonner pour éviter de tracer trop de points
            sample_size = min(1000, len(lr_rates))
            indices = np.linspace(0, len(lr_rates)-1, sample_size, dtype=int)
            
            plt.plot(indices, [lr_rates[j] for j in indices], label=f"Agent {i+1}")
        
        plt.title("Taux d'apprentissage des agents (alternance)")
        plt.xlabel("Épisodes")
        plt.ylabel("Taux d'apprentissage")
        plt.legend()
    plt.tight_layout()


def visualize_policy_pygame_reusable(env, agent, max_steps=100, delay=0.5, screen_size=600, save_images=False):
    """
    Visualise la politique d'un agent avec pygame - version réutilisable
    """
    # Force reset Pygame
    if pygame.get_init():
        pygame.quit()
    
    pygame.init()
    
    # Réinitialiser l'environnement
    state_tuple, _ = env.reset()
    
    # Créer un dossier pour sauvegarder les images si nécessaire
    if save_images:
        img_dir = f"visu/policy_visualization_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(img_dir, exist_ok=True)
    
    # Remplacer complètement l'initialisation pygame de l'environnement
    env.pygame_initialized = True
    env.screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Multi-Agent Frozen Lake")
    env.cell_size = screen_size // max(env.grid_size[0], env.grid_size[1])
    
    # Charger les images si nécessaire
    if not hasattr(env, 'images') or env.images is None:
        # Load images
        img_dir = "../img/"
        env.images = {
            'F': pygame.image.load(img_dir + "ice.png"),
            'H': pygame.image.load(img_dir + "hole.png"),
            'G': pygame.image.load(img_dir + "ice.png"),  # Use ice as background for goal
            'S': pygame.image.load(img_dir + "stool.png")
        }
        
        # Scale images to cell size
        for key in env.images:
            env.images[key] = pygame.transform.scale(env.images[key], (env.cell_size, env.cell_size))
        
        # Load goal sprite separately to overlay on ice
        env.goal_sprite = pygame.image.load(img_dir + "goal.png")
        env.goal_sprite = pygame.transform.scale(env.goal_sprite, (env.cell_size, env.cell_size))
        
        # Load agent images for different directions
        env.agent_images = {
            'up': pygame.image.load(img_dir + "elf_up.png"),
            'down': pygame.image.load(img_dir + "elf_down.png"),
            'left': pygame.image.load(img_dir + "elf_left.png"),
            'right': pygame.image.load(img_dir + "elf_right.png")
        }
        
        # Scale agent images
        for key in env.agent_images:
            env.agent_images[key] = pygame.transform.scale(env.agent_images[key], (env.cell_size, env.cell_size))
    
    # Initialize other attributes that might be needed
    if not hasattr(env, 'font') or env.font is None:
        env.font = pygame.font.Font(None, env.cell_size // 2)
    
    env.last_actions = [1] * env.num_agents  # Default: facing down
    
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
                    # Pour CQL utiliser la table Q centralisée
                    joint_state = state_tuple
                    action = agent.get_action(joint_state)[i]
                    actions.append(action)
            else:
                # Pour les autres agents choisir l'action aléatoirement
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
                break
    
    # Si save_images est activé, informer l'utilisateur
    if save_images:
        print(f"\nImages sauvegardées dans le dossier: {img_dir}")
    
    # Nettoyer proprement Pygame
    pygame.quit()
    
    # Assurer que l'attribut pygame_initialized est correctement mis à jour
    env.pygame_initialized = False


def evaluate_policy(env, agent, max_steps=50, verbose=True):
    state_tuple, _ = env.reset()
    ##single agent part
    done = False
    step_count = 0
    if isinstance(agent,QAgent):
        agent_reward=0
        if verbose:
            print(f"\n--- Début de l'évaluation de la politique ---")
        agent.epsilon=0
        
        for step in range(max_steps):
            state = state_tuple
            action = np.argmax(agent.q_table[state])  # Strictement déterministe
            next_state_tuple, reward, done, truncated, info = env.step(action)
            step_count += 1
            agent_reward+=reward
            if verbose:
                action_names = ['GAUCHE', 'BAS', 'DROITE', 'HAUT']
                print(f"Étape {step_count}: Action={[action_names[action]]}")
                print(f"Récompenses à cette étape: {reward}")
            state_tuple = next_state_tuple
            if done:
                break
        return {
        'agent_reward': agent_reward,
        'steps': step_count
    }
    # Suivre les récompenses individuelles de chaque agent
    agent_rewards = [0] * env.num_agents
    # Suivre quel objectif chaque agent a atteint (-1 = aucun)
    agent_goals = [-1] * env.num_agents
    
    collision_count = 0
    collision_steps = []
    
    if verbose:
        print(f"\n--- Début de l'évaluation de la politique ---")
    
    
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
                    # Pour CQL utiliser la table Q centralisée
                    joint_state = state_tuple
                    action = agent.get_action(joint_state)[i]
                    actions.append(action)
            else:
                # Pour les autres agents choisir l'action aléatoirement
                actions.append(np.random.randint(0, 4))
                if verbose:
                    print("random action")
        
        # Exécuter l'action
        next_state_tuple, rewards, dones, truncated, info = env.step(actions)
        
        # Mettre à jour les statistiques
        step_count += 1
        
        # Stocker les récompenses individuelles
        for i in range(env.num_agents):
            agent_rewards[i] += rewards[i]
        
        # Vérifier quel agent a atteint quel objectif
        if hasattr(env, 'goal_positions') and hasattr(env, 'agent_positions'):
            # Vérifier chaque agent
            for i in range(env.num_agents):
                if env.agent_done[i] and agent_goals[i] == -1:
                    # L'agent a terminé mais nous n'avons pas encore enregistré son objectif
                    position = tuple(env.agent_positions[i])
                    
                    # Recherche de l'objectif atteint
                    for goal_idx, goal_pos in enumerate(env.goal_positions):
                        if position == goal_pos:
                            agent_goals[i] = goal_idx
                            if verbose:
                                print(f"Étape {step_count}: Agent {i} a atteint l'objectif {goal_idx}!")
                            break
        
        # Suivi des collisions
        if info.get('collisions', False):
            collision_count += 1
            collision_steps.append(step_count)
            if verbose:
                print(f"⚠️ Collision entre les agents {info.get('collision_agents', [])}!")
        
        # Afficher les informations sur l'étape
        if verbose:
            action_names = ['GAUCHE', 'BAS', 'DROITE', 'HAUT']
            print(f"Étape {step_count}: Actions={[action_names[a] for a in actions]}")
            print(f"Récompenses à cette étape: {rewards}")
        
        # Mettre à jour l'état
        state_tuple = next_state_tuple
        
        # Vérifier si l'épisode est terminé
        if all(dones):
            if verbose:
                print("Tous les agents ont terminé!")
            break
    
    # Affichage du résumé
    if verbose:
        print(f"\n--- Fin de l'évaluation ---")
        print(f"Total de {step_count} étapes")
        
        # Afficher les récompenses individuelles finales
        for i in range(env.num_agents):
            goal_str = f"Objectif {agent_goals[i]}" if agent_goals[i] >= 0 else "Aucun objectif"
            print(f"Agent {i}: Récompense = {agent_rewards[i]:.2f}, {goal_str}")
        
        print(f"Récompense totale: {sum(agent_rewards):.2f}")
        print(f"Nombre de collisions: {collision_count}")
        if collision_count > 0:
            print(f"Collisions aux étapes: {collision_steps}")
    
    # Retourner les statistiques pour une utilisation ultérieure
    return {
        'agent_rewards': agent_rewards,
        'agent_goals': agent_goals,
        'total_reward': sum(agent_rewards),
        'steps': step_count,
        'success': sum(1 for g in agent_goals if g >= 0),  # Nombre d'agents ayant atteint un objectif
        'collision_count': collision_count,
        'collision_steps': collision_steps
    }

def runSimulationCommonGoal(agent, map_, num_agent, num_episodes=10000, silent=True):
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

def visualizePolicyCommonGoal(env, agent, num_episodes=2, max_steps=20, use_pygame=True, num_agents=2):
    """Visualize the learned policy"""
    
    # Action names for better visualization
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    
    try:
        for i in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            print(f"\n=== Test Episode {i+1} ===")
            if use_pygame:
                env.render_pygame()
            else:
                print("Initial state:")
                env.render()
            
            while not done and not truncated and steps < max_steps:
                # Use trained policy (no exploration)
                state_tuple = tuple(state)
                
                # Get actions based on agent's Q-table
                # This assumes agent.q_table is structured to handle num_agents
                joint_actions = np.unravel_index(
                    np.argmax(agent.q_table[state_tuple]),
                    tuple([agent.action_size] * num_agents)
                )
                action = joint_actions
                
                # Take action
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Check for overlaps - this needs to be generalized for multiple agents
                overlaps = []
                for i in range(num_agents):
                    for j in range(i+1, num_agents):
                        # Compare positions of each pair of agents
                        agent_i_pos = (next_state[i*2], next_state[i*2 + 1])
                        agent_j_pos = (next_state[j*2], next_state[j*2 + 1])
                        if agent_i_pos == agent_j_pos:
                            overlaps.append((i, j))
                
                # Update state and reward
                state = next_state
                total_reward += reward
                steps += 1
                
                # Render with action information
                print(f"Step {steps}:")
                for agent_idx in range(num_agents):
                    print(f"Agent {agent_idx+1}: {action_names[action[agent_idx]]}")
                print(f"Reward: {reward}")
                
                if overlaps:
                    print("Overlaps detected between agents:", overlaps)
                
                if use_pygame:
                    env.render_pygame()
                    time.sleep(0.5)
                else:
                    env.render()
                    time.sleep(0.5)
            
            print(f"Episode finished after {steps} steps with total reward: {total_reward}")
            if done and total_reward > 0:
                print("Success! At least one agent reached the goal.")
            elif done and total_reward <= 0:
                print("Failed. Agents fell into holes or couldn't reach the goal.")
            else:
                print("Truncated. Maximum steps reached.")
            
            # Short pause between episodes
            time.sleep(1)
        
    finally:
        env.close()
        if pygame.get_init():
            pygame.quit()

