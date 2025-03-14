#@title IQL

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt 
from Q_agent import QAgent
from multi_agent_frozen  import FrozenLakeFlexibleAgentsEnv


class IndependentQLearning:
    """
    Implémentation de Q-Learning indépendant (IQL) pour les environnements multi-agents.
    Chaque agent apprend de manière indépendante sans prendre en compte les autres agents.
    """
    def __init__(self, env, n_agents=2, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.999):
        self.env = env

        # Obtenir les tailles des espaces d'état et d'action pour chaque agent
        # Pour FrozenLakeMultiAgentEnv, state_size est la taille totale de la grille pour chaque agent
        grid_size = env.grid_size
        state_size = grid_size[0] * grid_size[1]
        action_size = 4  # 0: gauche, 1: bas, 2: droite, 3: haut

        # Créer un agent Q pour chaque agent dans l'environnement
        self.agents = [
            QAgent(state_size, action_size, learning_rate, discount_factor,
                   exploration_rate, min_exploration_rate, exploration_decay)
            for _ in range(n_agents)
        ]

        self.n_agents = n_agents

    def train(self, episodes=10000, max_steps=100, verbose=True):
      """
      Entraîne les agents de manière indépendante dans l'environnement multi-agent.

      Args:
          episodes: Nombre total d'épisodes d'entraînement
          max_steps: Nombre maximum d'étapes par épisode
          verbose: Si True, affiche la progression de l'entraînement

      Returns:
          Historique des récompenses et des taux de réussite par épisode
      """
      rewards_history = []
      rewards_rates = [[] for _ in range(self.n_agents)]  # Remplace success_rates
      episode_steps = []

      if verbose:
          episodes_iter = tqdm(range(episodes))
      else:
          episodes_iter = range(episodes)

      for episode in episodes_iter:
          # Réinitialiser l'environnement
          states, _ = self.env.reset()
          episode_rewards = [0] * self.n_agents
          dones = [False] * self.n_agents

          for step in range(max_steps):
              # Chaque agent choisit une action de manière indépendante
              actions = [
                  self.agents[i].get_action(states[i])
                  for i in range(self.n_agents)
              ]

              # Exécuter les actions dans l'environnement
              next_states, rewards, new_dones, truncated, _ = self.env.step(actions)

              # Mettre à jour les tables Q pour chaque agent de manière indépendante
              for i in range(self.n_agents):
                  # Ne mettre à jour que si l'agent n'est pas déjà terminé
                  if not dones[i]:
                      self.agents[i].update(
                          states[i], actions[i], rewards[i], next_states[i], new_dones[i]
                      )
                      episode_rewards[i] += rewards[i]

              # Mettre à jour les états et les drapeaux terminés
              states = next_states
              dones = new_dones

              # Sortir de la boucle si tous les agents ont terminé
              if all(dones) or all(truncated):
                  break

          # Enregistrer les résultats de l'épisode
          rewards_history.append(episode_rewards)
          episode_steps.append(step + 1)

          # Enregistrer les récompenses pour chaque agent au lieu des succès
          for i in range(self.n_agents):
              rewards_rates[i].append(episode_rewards[i])

          # Réduire epsilon pour chaque agent
          for agent in self.agents:
              agent.decay_epsilon()

          # Afficher la progression
          if verbose and (episode + 1) % (episodes // 10) == 0:
              reward_window = 100
              recent_rewards = [
                  np.mean(rewards_rates[i][-min(reward_window, len(rewards_rates[i])):])
                  for i in range(self.n_agents)
              ]

              print(f"Épisode {episode + 1}/{episodes}, "
                    f"Récompenses moyennes agents: {[f'{reward:.2f}' for reward in recent_rewards]}, "
                    f"Epsilon: {self.agents[0].epsilon:.4f}")

      return {
          'rewards': rewards_history,
          'rewards_rates': rewards_rates,  # Remplacé 'success_rates' par 'rewards_rates'
          'steps': episode_steps
      }

    def render_policy(self):
        """
        Affiche la politique apprise pour chaque agent
        """
        for agent_idx, agent in enumerate(self.agents):
            print(f"\nPolitique de l'agent {agent_idx + 1}:")
            grid_size = self.env.grid_size
            policy = np.argmax(agent.q_table, axis=1).reshape(grid_size)

            # Utiliser des symboles pour représenter les actions
            action_symbols = ['←', '↓', '→', '↑']

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    state_idx = i * grid_size[1] + j
                    print(f" {action_symbols[policy[i, j]]} ", end='')
                print()



class CentralizedQLearning:
    """
    Implémentation de l'apprentissage Q centralisé pour un environnement multi-agent.
    Supporte un nombre flexible d'agents.
    Utilise defaultdict pour une allocation efficace de la mémoire.
    """

    def __init__(self, env, n_agents=2, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        """
        Initialise l'apprentissage Q centralisé.

        Args:
            env: Environnement multi-agent
            n_agents: Nombre d'agents
            learning_rate: Taux d'apprentissage (alpha)
            discount_factor: Facteur d'actualisation (gamma)
            exploration_rate: Taux d'exploration initial (epsilon)
            min_exploration_rate: Taux d'exploration minimal
            epsilon_decay: Taux de décroissance de l'exploration
        """
        self.env = env
        self.n_agents = n_agents
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay

        # Obtenir les dimensions de l'espace d'action
        self.action_dims = []
        for i in range(n_agents):
            # Gère les deux façons possibles de représenter l'espace d'action
            if hasattr(env.action_space, '__getitem__'):
                self.action_dims.append(env.action_space[i].n)
            else:
                # Pour les espaces MultiDiscrete
                self.action_dims.append(4)  # Assume 4 actions (LEFT, DOWN, RIGHT, UP)

        # Table Q avec defaultdict pour allocation dynamique
        # Initialisation de la valeur par défaut
        self._setup_q_table()

    def _setup_q_table(self):
        """
        Configure la table Q en fonction du nombre d'agents.
        """
        def create_empty_q_values():
            # Crée dynamiquement un array numpy multidimensionnel pour stocker les valeurs Q
            return np.zeros(self.action_dims)

        self.q_table = defaultdict(create_empty_q_values)

    def get_action(self, states):
        """
        Sélectionne des actions pour tous les agents selon la politique epsilon-greedy.

        Args:
            states: Liste des états individuels des agents

        Returns:
            Liste des actions individuelles pour chaque agent
        """
        state_tuple = tuple(states)  # Convertir en tuple pour utiliser comme clé

        # Exploration-exploitation
        if np.random.random() < self.epsilon:
            # Actions aléatoires
            actions = [np.random.randint(0, self.action_dims[i]) for i in range(self.n_agents)]
        else:
            # Actions optimales selon la table Q
            joint_actions = np.unravel_index(
                np.argmax(self.q_table[state_tuple]),
                self.action_dims
            )
            actions = list(joint_actions)

        return actions

    def update(self, states, actions, rewards, next_states, dones):
        """
        Met à jour la table Q centralisée.

        Args:
            states: Liste des états individuels actuels
            actions: Liste des actions individuelles prises
            rewards: Liste des récompenses individuelles reçues
            next_states: Liste des états individuels suivants
            dones: Liste des drapeaux indiquant si les épisodes sont terminés
        """
        state_tuple = tuple(states)
        next_state_tuple = tuple(next_states)

        # Calculer la récompense totale
        total_reward = sum(rewards)

        # Vérifier si tous les agents ont terminé
        all_done = all(dones)

        # Créer un tuple d'index pour accéder à la valeur Q
        action_tuple = tuple(actions)

        # Valeur Q actuelle
        current_q = self.q_table[state_tuple][action_tuple]

        # Valeur future Q maximum (0 si terminé)
        if all_done:
            next_q = 0
        else:
            next_q = np.max(self.q_table[next_state_tuple])

        # Mise à jour Q-learning
        new_q = current_q + self.lr * (total_reward + self.gamma * next_q - current_q)
        self.q_table[state_tuple][action_tuple] = new_q

    def index_to_actions(self, joint_action):
        """
        Convertit un indice d'action jointe en actions individuelles.

        Args:
            joint_action: Indice d'action jointe

        Returns:
            list: Liste des actions individuelles pour chaque agent
        """
        # Pour un joint_action unique (index), retourne les actions individuelles
        return list(np.unravel_index(joint_action, self.action_dims))

    def state_to_index(self, states):
        """
        Convertit les états individuels des agents en une clé pour la table Q centralisée.

        Args:
            states: Liste des états individuels des agents

        Returns:
            tuple: Tuple d'états utilisable comme clé pour la table Q
        """
        return tuple(states)

    def decay_epsilon(self):
        """
        Réduit le taux d'exploration (epsilon) selon le taux de décroissance.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def render_policy(self):
        """
        Affiche des informations sur la politique apprise.
        """
        print("Politique centralisée:")

        # Calculer des statistiques sur la table Q
        n_states = len(self.q_table)

        if n_states == 0:
            print("Aucun état exploré.")
            return

        q_values = []
        for state, actions_q in self.q_table.items():
            q_values.extend(actions_q.flatten())

        q_values = np.array(q_values)
        nonzero_entries = np.count_nonzero(q_values)

        print(f"Nombre d'états visités: {n_states}")
        print(f"Nombre d'entrées Q non nulles: {nonzero_entries}")
        print(f"Valeur Q moyenne: {np.mean(q_values):.4f}")
        print(f"Valeur Q maximale: {np.max(q_values):.4f}")
        if nonzero_entries > 0:
            nonzero_q = q_values[q_values != 0]
            print(f"Valeur Q minimale (non-nulle): {np.min(nonzero_q):.4f}")
        else:
            print("Valeur Q minimale: N/A")

    def train(self, episodes=10000, max_steps=100, verbose=True):
        """
        Entraîne l'agent centralisé dans l'environnement multi-agent.

        Args:
            episodes: Nombre total d'épisodes d'entraînement
            max_steps: Nombre maximum d'étapes par épisode
            verbose: Si True, affiche la progression de l'entraînement

        Returns:
            Historique des récompenses par épisode
        """
        rewards_history = []
        rewards_rates = [[] for _ in range(self.n_agents)]
        episode_steps = []

        if verbose:
            episodes_iter = tqdm(range(episodes))
        else:
            episodes_iter = range(episodes)

        for episode in episodes_iter:
            # Réinitialiser l'environnement
            states, _ = self.env.reset()
            episode_rewards = [0] * self.n_agents
            dones = [False] * self.n_agents

            for step in range(max_steps):
                # Obtenir les actions
                actions = self.get_action(states)

                # Exécuter les actions dans l'environnement
                next_states, rewards, new_dones, truncated, _ = self.env.step(actions)

                # Mettre à jour la table Q
                self.update(states, actions, rewards, next_states, new_dones)

                # Accumuler les récompenses
                for i in range(self.n_agents):
                    if not dones[i]:
                        episode_rewards[i] += rewards[i]

                # Mettre à jour les états et les drapeaux terminés
                states = next_states
                dones = new_dones

                # Sortir de la boucle si tous les agents ont terminé
                if all(dones) or all(truncated):
                    break

            # Enregistrer les résultats de l'épisode
            rewards_history.append(episode_rewards)
            episode_steps.append(step + 1)

            # Enregistrer les récompenses pour chaque agent
            for i in range(self.n_agents):
                rewards_rates[i].append(episode_rewards[i])

            # Réduire epsilon
            self.decay_epsilon()

            # Afficher la progression
            if verbose and (episode + 1) % (episodes // 10) == 0:
                reward_window = 100
                recent_rewards = [
                    np.mean(rewards_rates[i][-min(reward_window, len(rewards_rates[i])):])
                    for i in range(self.n_agents)
                ]

                print(f"Épisode {episode + 1}/{episodes}, "
                      f"Récompenses moyennes agents: {[f'{reward:.2f}' for reward in recent_rewards]}, "
                      f"Epsilon: {self.epsilon:.4f}")

        return {
            'rewards': rewards_history,
            'rewards_rates': rewards_rates,
            'steps': episode_steps
        }
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

def evaluate_policy(agent, env, n_episodes=100, max_steps=100, results_file=None):
    """
    Évalue la performance de la politique apprise, tenant compte du bonus de collaboration.
    Compatible avec IndependentQLearning et CentralizedQLearning.

    Args:
        agent: Instance IndependentQLearning ou CentralizedQLearning contenant les agents entraînés
        env: Environnement FrozenLakeMultiAgentEnv
        n_episodes: Nombre d'épisodes d'évaluation
        max_steps: Nombre maximum d'étapes par épisode
        results_file: Fichier pour sauvegarder les résultats

    Returns:
        dict: Dictionnaire contenant les statistiques d'évaluation
    """
    success_rates = [0] * agent.n_agents
    steps_when_reached = [[] for _ in range(agent.n_agents)]  # Pour enregistrer l'étape exacte où l'objectif est atteint
    rewards_total = [[] for _ in range(agent.n_agents)]
    collaborative_successes = 0  # Compteur pour les succès collaboratifs

    print(f"\nÉvaluation de la politique sur {n_episodes} épisodes...")
    if results_file:
        results_file.write(f"\nÉvaluation de la politique sur {n_episodes} épisodes...\n")

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = [False] * agent.n_agents
        episode_rewards = [0] * agent.n_agents
        reached_goal = [False] * agent.n_agents  # Pour suivre si l'agent a atteint l'objectif
        collaboration_this_episode = False

        for step in range(max_steps):
            # Utiliser la politique apprise (sans exploration)
            if hasattr(agent, 'agents'):  # IndependentQLearning
                actions = [np.argmax(agent.agents[i].q_table[state[i]]) for i in range(agent.n_agents)]
            else:  # CentralizedQLearning
                # Nous n'utilisons pas la méthode get_action pour éviter l'exploration
                joint_state = agent.state_to_index(state)
                # Modification ici: utiliser directement agent.q_table au lieu de agent.central_agent.q_table
                joint_action = np.argmax(agent.q_table[joint_state])
                actions = agent.index_to_actions(joint_action)

            # Exécuter les actions
            next_state, rewards, done, _, info = env.step(actions)

            # Vérifier s'il y a eu une arrivée simultanée
            if 'simultaneous_arrival' in info and info['simultaneous_arrival']:
                collaboration_this_episode = True

            # Accumuler les récompenses
            for i in range(agent.n_agents):
                episode_rewards[i] += rewards[i]
                # Considérer qu'un agent a atteint l'objectif s'il reçoit une récompense positive
                if rewards[i] > 0 and not reached_goal[i]:  # S'il atteint l'objectif pour la première fois
                    reached_goal[i] = True
                    steps_when_reached[i].append(step + 1)  # Enregistrer l'étape exacte

            # Si tous les agents ont terminé, sortir de la boucle
            if all(done):
                break

            state = next_state

        # Incrémenter le compteur de succès collaboratifs
        if collaboration_this_episode:
            collaborative_successes += 1

        # Enregistrer les statistiques
        for i in range(agent.n_agents):
            # Un agent réussit s'il a atteint l'objectif
            success_rates[i] += 1 if reached_goal[i] else 0

            # Enregistrer la récompense totale
            rewards_total[i].append(episode_rewards[i])

    # Calculer les statistiques finales
    for i in range(agent.n_agents):
        success_rates[i] /= n_episodes

    # Calculer le taux de collaboration
    collaboration_rate = collaborative_successes / n_episodes

    # Préparer les résultats
    eval_results = {
        'success_rates': success_rates,
        'avg_steps_to_goal': [np.mean(steps) if steps else float('nan') for steps in steps_when_reached],
        'avg_rewards': [np.mean(rewards) for rewards in rewards_total],
        'collaboration_rate': collaboration_rate
    }

    # Afficher les résultats
    print("\nRésultats de l'évaluation:")
    if results_file:
        results_file.write("\nRésultats de l'évaluation:\n")
        
    for i in range(agent.n_agents):
        agent_results = [
            f"Agent {i+1}:",
            f"  - Taux de réussite: {success_rates[i]*100:.2f}%"
        ]
        
        if steps_when_reached[i]:
            agent_results.append(f"  - Nombre moyen d'étapes (succès): {eval_results['avg_steps_to_goal'][i]:.2f}")
        else:
            agent_results.append(f"  - Nombre moyen d'étapes (succès): N/A (aucun succès)")
            
        agent_results.append(f"  - Récompense moyenne: {eval_results['avg_rewards'][i]:.4f}")
        
        # Afficher et écrire dans le fichier
        for line in agent_results:
            print(line)
            if results_file:
                results_file.write(line + "\n")

    collab_result = f"\nTaux de collaboration: {collaboration_rate*100:.2f}%"
    print(collab_result)
    if results_file:
        results_file.write(collab_result + "\n")

    return eval_results

def test_centralized_learning(env, n_agents,
                              learning_rate=0.3,
                              max_episodes=50000,
                              discount_factor=0.99,
                              exploration_rate=1.0,
                              min_exploration_rate=0.05,
                              exploration_decay=0.99995,
                              window_size=200):
    
    # Créer le dossier results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    
    # Nom du fichier de résultats avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/centralized_learning_{timestamp}.txt"
    
    # Ouvrir le fichier de résultats
    with open(results_filename, "w", encoding="utf-8") as results_file:
        # Enregistrer les paramètres
        results_file.write("=== PARAMÈTRES D'APPRENTISSAGE CENTRALISÉ ===\n")
        params = {
            "n_agents": n_agents,
            "learning_rate": learning_rate,
            "max_episodes": max_episodes,
            "discount_factor": discount_factor,
            "exploration_rate": exploration_rate,
            "min_exploration_rate": min_exploration_rate,
            "exploration_decay": exploration_decay,
            "window_size": window_size
        }
        
        for key, value in params.items():
            results_file.write(f"{key}: {value}\n")
        
        # Visualiser l'environnement initial
        print("Environnement initial:")
        results_file.write("\nEnvironnement initial:\n")
        env_str = str(env.render())
        results_file.write(env_str + "\n" if env_str else "[Rendu de l'environnement non capturé]\n")
        
        # Créer l'instance d'apprentissage centralisé
        cql = CentralizedQLearning(
            env,
            n_agents=n_agents,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            min_exploration_rate=min_exploration_rate,
            exploration_decay=exploration_decay
        )

        # Entraîner l'agent centralisé
        print("\nDébut de l'entraînement centralisé...")
        results_file.write("\nDébut de l'entraînement centralisé...\n")
        results = cql.train(episodes=max_episodes, max_steps=100)

        # Sauvegarder les métriques d'entraînement
        metrics_file = f"results/centralized_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            # Créez une copie pour ne pas modifier l'original
            json_results = {}
            
            # Traitez chaque clé disponible
            if 'rewards_rates' in results:
                json_results["rewards_rates"] = [rates.tolist() if isinstance(rates, np.ndarray) else rates for rates in results['rewards_rates']]
            
            # Ajoutez d'autres clés présentes dans le dictionnaire
            for key in results:
                if key != 'rewards_rates':  # Déjà traité
                    if isinstance(results[key], np.ndarray):
                        json_results[key] = results[key].tolist()
                    elif isinstance(results[key], list):
                        json_results[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in results[key]]
                    else:
                        json_results[key] = results[key]
                        
            json.dump(json_results, f)

        # Afficher la politique apprise
        print("\nPolitique centralisée apprise:")
        results_file.write("\nPolitique centralisée apprise:\n")
        policy_str = str(cql.render_policy())
        results_file.write(policy_str + "\n" if policy_str else "[Politique non capturée sous forme de texte]\n")

        # Visualiser les résultats
        plt.figure(figsize=(15, 10))

        # Tracer les taux de récompense pour chaque agent
        window_size_plot = window_size
        for i in range(cql.n_agents):
            plt.subplot(cql.n_agents, 1, i+1)

            # Calculer les récompenses moyennes par fenêtre
            rewards_rates = results['rewards_rates'][i]
            rewards_rates_smoothed = []

            for j in range(0, len(rewards_rates), window_size_plot):
                if j + window_size_plot <= len(rewards_rates):
                    rewards_rates_smoothed.append(np.mean(rewards_rates[j:j+window_size_plot]))

            plt.plot(range(0, len(rewards_rates_smoothed) * window_size_plot, window_size_plot), rewards_rates_smoothed)
            plt.title(f"Agent {i+1} - Récompenses moyennes (moyenne sur {window_size_plot} épisodes)")
            plt.xlabel("Épisodes")
            plt.ylabel("Récompenses moyennes")

        plt.tight_layout()
        plot_filename = f"results/centralized_learning_plot_{timestamp}.png"
        plt.savefig(plot_filename)
        results_file.write(f"\nGraphique sauvegardé dans {plot_filename}\n")
        plt.show()

        print('Test de la politique obtenue')
        results_file.write('\nTest de la politique obtenue\n')
        eval_results = evaluate_policy(cql, env, n_episodes=1000, max_steps=100, results_file=results_file)
        
        # Sauvegarder les résultats d'évaluation
        eval_file = f"results/centralized_eval_{timestamp}.json"
        with open(eval_file, "w") as f:
            # Convertir les valeurs numpy en types Python standard pour JSON
            json_eval = {}
            for key, value in eval_results.items():
                if isinstance(value, np.ndarray):
                    json_eval[key] = value.tolist()
                elif isinstance(value, list) and any(isinstance(x, np.number) for x in value):
                    json_eval[key] = [float(x) if isinstance(x, np.number) else x for x in value]
                else:
                    json_eval[key] = value
            json.dump(json_eval, f)
            
        results_file.write(f"\nRésultats d'évaluation sauvegardés dans {eval_file}\n")
        print(f"\nTous les résultats ont été sauvegardés dans le dossier 'results', avec le fichier principal: {results_filename}")

    return cql, results

def test_independent_learning(env, n_agents=2,
                              learning_rate=0.3,
                              max_episodes=50000,
                              discount_factor=0.99,
                              exploration_rate=1.0,
                              min_exploration_rate=0.05,
                              exploration_decay=0.99995,
                              window_size=200):
    
    # Créer le dossier results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    
    # Nom du fichier de résultats avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/independent_learning_{timestamp}.txt"
    
    # Ouvrir le fichier de résultats
    with open(results_filename, "w", encoding="utf-8") as results_file:
        # Enregistrer les paramètres
        results_file.write("=== PARAMÈTRES D'APPRENTISSAGE INDÉPENDANT ===\n")
        params = {
            "n_agents": n_agents,
            "learning_rate": learning_rate,
            "max_episodes": max_episodes,
            "discount_factor": discount_factor,
            "exploration_rate": exploration_rate,
            "min_exploration_rate": min_exploration_rate,
            "exploration_decay": exploration_decay,
            "window_size": window_size
        }
        
        for key, value in params.items():
            results_file.write(f"{key}: {value}\n")

        # Visualiser l'environnement initial
        print("Environnement initial:")
        results_file.write("\nEnvironnement initial:\n")
        env_str = str(env.render())
        results_file.write(env_str + "\n" if env_str else "[Rendu de l'environnement non capturé]\n")

        # Créer l'instance d'apprentissage indépendant
        iql = IndependentQLearning(
            env,
            n_agents=n_agents,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            min_exploration_rate=min_exploration_rate,
            exploration_decay=exploration_decay
        )

        # Entraîner les agents
        print("\nDébut de l'entraînement...")
        results_file.write("\nDébut de l'entraînement indépendant...\n")
        results = iql.train(episodes=max_episodes, max_steps=100)

        
        # # Sauvegarder les métriques d'entraînement
        # metrics_file = f"results/independant_metrics_{timestamp}.json"
        # with open(metrics_file, "w") as f:
        #     # Créez une copie pour ne pas modifier l'original
        #     json_results = {}
            
        #     # Traitez chaque clé disponible
        #     if 'rewards_rates' in results:
        #         json_results["rewards_rates"] = [rates.tolist() if isinstance(rates, np.ndarray) else rates for rates in results['rewards_rates']]
            
        #     # Ajoutez d'autres clés présentes dans le dictionnaire
        #     for key in results:
        #         if key != 'rewards_rates':  # Déjà traité
        #             if isinstance(results[key], np.ndarray):
        #                 json_results[key] = results[key].tolist()
        #             elif isinstance(results[key], list):
        #                 json_results[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in results[key]]
        #             else:
        #                 json_results[key] = results[key]
                    
        # json.dump(json_results, f)

        # Visualiser les résultats
        plt.figure(figsize=(15, 10))

        # Tracer les taux de récompense pour chaque agent
        window_size_plot = window_size
        for i in range(iql.n_agents):
            plt.subplot(iql.n_agents, 1, i+1)

            # Calculer les récompenses moyennes par fenêtre
            rewards_rates = results['rewards_rates'][i]
            rewards_rates_smoothed = []

            for j in range(0, len(rewards_rates), window_size_plot):
                if j + window_size_plot <= len(rewards_rates):
                    rewards_rates_smoothed.append(np.mean(rewards_rates[j:j+window_size_plot]))

            plt.plot(range(0, len(rewards_rates_smoothed) * window_size_plot, window_size_plot), rewards_rates_smoothed)
            plt.title(f"Agent {i+1} - Récompenses moyennes (moyenne sur {window_size_plot} épisodes)")
            plt.xlabel("Épisodes")
            plt.ylabel("Récompenses moyennes")

        plt.tight_layout()
        plot_filename = f"results/independent_learning_plot_{timestamp}.png"
        plt.savefig(plot_filename)
        results_file.write(f"\nGraphique sauvegardé dans {plot_filename}\n")
        plt.show()
        
        print('Test de la politique obtenue')
        results_file.write('\nTest de la politique obtenue\n')
        eval_results = evaluate_policy(iql, env, n_episodes=1000, max_steps=100, results_file=results_file)
        
        # Sauvegarder les résultats d'évaluation
        eval_file = f"results/independent_eval_{timestamp}.json"
        with open(eval_file, "w") as f:
            # Convertir les valeurs numpy en types Python standard pour JSON
            json_eval = {}
            for key, value in eval_results.items():
                if isinstance(value, np.ndarray):
                    json_eval[key] = value.tolist()
                elif isinstance(value, list) and any(isinstance(x, np.number) for x in value):
                    json_eval[key] = [float(x) if isinstance(x, np.number) else x for x in value]
                else:
                    json_eval[key] = value
            json.dump(json_eval, f)

        results_file.write(f"\nRésultats d'évaluation sauvegardés dans {eval_file}\n")
        print(f"\nTous les résultats ont été sauvegardés dans le dossier 'results', avec le fichier principal: {results_filename}")

    return iql, results
if __name__ == "__main__":
    # Number of agents
    n_agents = 2

    # Environment setup
    env = FrozenLakeFlexibleAgentsEnv(
        num_agents=n_agents, 
        grid_size=(5, 5), 
        slip_prob=0.0, 
        hole_prob=0.2, 
        seed=7, 
        collaboration_bonus=1.0
    )

    # Display initial environment
    env.render()
    env.reset()

    # Centralized Q-learning parameters
    learning_rate_cql = 0.5
    max_episodes_cql = 30000
    discount_factor_cql = 0.99
    exploration_rate_cql = 1.0
    min_exploration_rate_cql = 0.05
    exploration_decay_cql = 0.9995
    window_size_cql = int(max_episodes_cql/200)

    # Run centralized Q-learning
    cql, cql_results = test_centralized_learning(
        env=env,
        n_agents=n_agents,
        learning_rate=learning_rate_cql,
        discount_factor=discount_factor_cql,
        exploration_rate=exploration_rate_cql,
        min_exploration_rate=min_exploration_rate_cql,
        exploration_decay=exploration_decay_cql,
        max_episodes=max_episodes_cql,
        window_size=window_size_cql
    )

    # Reset environment before independent learning
    env.reset()

    # Independent Q-learning parameters
    learning_rate_iql = 0.5
    max_episodes_iql = 30000
    discount_factor_iql = 0.99
    exploration_rate_iql = 1.0
    min_exploration_rate_iql = 0.05
    exploration_decay_iql = 0.9997
    window_size_iql = int(max_episodes_iql/200)

    # Run independent Q-learning
    iql, iql_results = test_independent_learning(
        env=env,
        n_agents=n_agents,
        learning_rate=learning_rate_iql,
        discount_factor=discount_factor_iql,
        exploration_rate=exploration_rate_iql,
        min_exploration_rate=min_exploration_rate_iql,
        exploration_decay=exploration_decay_iql,
        max_episodes=max_episodes_iql,
        window_size=window_size_iql
    )
