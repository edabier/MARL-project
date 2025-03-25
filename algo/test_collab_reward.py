from multi_agent_frozen_env import FrozenLake4goals,FrozenLakeFlexibleAgentsEnvCol
from IQL_CQL import CentralizedQLearning,IndependentQLearning,IndependentQLearningSave,AlternatingIQL,test_independent_learning_save,test_independent_q_learning,test_centralized_learning,test_alternating_learning


import numpy as np
import time
import pygame
import os
import pickle
import datetime
import matplotlib.pyplot as plt
from scipy import stats
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





if __name__ == "__main__":
    n_agents=10
    Iql_status=False
    iql_save=True
    alt_iql_status=True
    alt_iql_save=True

    ### seed 25 
    """ G H . . . . . G
        . . . H . . H H
        . H H H . H . .
        . H . A . . . A
        . . . A . . . H
        . H H A H . . .
        H H H H . . H .
        G . . . . H . G"""
    env_params={"num_agents":n_agents, 
                "grid_size":(8, 8), 
                "slip_prob":0., 
                "hole_prob":0.3, 
                "seed":25, 
                "collaboration_bonus":0,
                "collision_penalty":30}
    env=FrozenLake4goals(**env_params)
    
    # env=FrozenLake4goals( num_agents=n_agents, grid_size=(8, 8), slip_prob=0., hole_prob=0.3, seed=42, 
    #              collaboration_bonus=0, collision_penalty=0)
    # env.render()
    # env.reset()
    if Iql_status:
        learning_rate_iql = 0.5
        max_episodes_iql = 100000
        discount_factor_iql = 0.99
        exploration_rate_iql = 1.0
        min_exploration_rate_iql = 0.05
        exploration_decay_iql = 0.99997
        window_size_iql = int(max_episodes_iql/200)

        # Run independent Q-learning
        iql, iql_results = test_independent_q_learning(
            env=env,
            n_agents=n_agents,
            learning_rate=learning_rate_iql,
            discount_factor=discount_factor_iql,
            exploration_rate=exploration_rate_iql,
            min_exploration_rate=min_exploration_rate_iql,
            exploration_decay=exploration_decay_iql,
            max_episodes=max_episodes_iql,
            window_size=window_size_iql,
            
        )
        if iql_save:
            save_agent(iql, "iql", env_info=env_params, save_dir="saved_agents")
        
        
        print("\nVisualisation de la politique IQL...")
        visualize_policy_pygame(
            env=env, 
            agent=iql, 
            max_steps=50,
            delay=0.5,  # Délai entre les étapes (en secondes)
            screen_size=600,
            save_images=True  # Sauvegarder des images pour chaque étape
        )
    if alt_iql_status:
        env.reset()
        alt_agent, alt_results = test_alternating_learning(
            env, n_agents=n_agents,
            base_learning_rate=0.4,
            max_episodes=150000,
            discount_factor=0.99,
            exploration_rate=1.0,
            min_exploration_rate=0.05,
            exploration_decay=0.99997,
            alternating_period=500,
            learning_rate_ratio=0.1,
            window_size=200
        )
        if alt_iql_save:
            save_agent(alt_agent, "alt_iql", env_info=None, save_dir="saved_agents")
        visualize_policy_pygame(
            env=env, 
            agent=alt_agent, 
            max_steps=50,
            delay=0.6,  # Délai entre les étapes (en secondes)
            screen_size=600,
            save_images=True  # Sauvegarder des images pour chaque étape
        )