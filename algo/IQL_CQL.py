#@title IQL

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt 
from Q_agent import QAgent
from multi_agent_frozen_env  import FrozenLakeFlexibleAgentsEnvCol,FrozenLake4goals
import os 
from datetime import datetime
import json
class IndependentQLearningSave:
    """
    Implémentation de Q-Learning indépendant (IQL) pour les environnements multi-agents.
    Chaque agent apprend de manière indépendante sans prendre en compte les autres agents.
    """
    def __init__(self, env, n_agents=2, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.999):
        self.env = env

        # Obtenir les tailles des espaces d'état et d'action pour chaque agent
        grid_size = env.grid_size
        state_size = grid_size[0] * grid_size[1]
        action_size = 4  # 0: gauche, 1: bas, 2: droite, 3: haut

        # Créer un agent Q pour chaque agent dans l'environnement
        self.agents = [
            QAgent(state_size, action_size, learning_rate, discount_factor,
                   exploration_rate, min_exploration_rate, exploration_decay)
            for _ in range(n_agents)
        ]

        # Sauvegarder les meilleures politiques
        self.best_agents = [None] * n_agents
        self.best_performance = float('-inf')  # Performance globale
        self.best_agent_performances = [float('-inf')] * n_agents  # Performance individuelle

        self.n_agents = n_agents

    def train(self, episodes=10000, max_steps=100, eval_frequency=1000, eval_episodes=100, verbose=True):
        """
        Entraîne les agents de manière indépendante dans l'environnement multi-agent.

        Args:
            episodes: Nombre total d'épisodes d'entraînement
            max_steps: Nombre maximum d'étapes par épisode
            eval_frequency: Fréquence d'évaluation de la politique
            eval_episodes: Nombre d'épisodes pour l'évaluation
            verbose: Si True, affiche la progression de l'entraînement

        Returns:
            Historique des récompenses et des taux de réussite par épisode
        """
        rewards_history = []
        rewards_rates = [[] for _ in range(self.n_agents)]
        episode_steps = []
        eval_results_history = []  # Pour suivre les résultats d'évaluation

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

            # Enregistrer les récompenses pour chaque agent
            for i in range(self.n_agents):
                rewards_rates[i].append(episode_rewards[i])

            # Réduire epsilon pour chaque agent
            for agent in self.agents:
                agent.decay_epsilon()

            # Évaluer périodiquement la politique actuelle et sauvegarder si meilleure
            if (episode + 1) % eval_frequency == 0:
                # Désactiver l'exploration temporairement
                original_epsilons = [agent.epsilon for agent in self.agents]
                for agent in self.agents:
                    agent.epsilon = 0.0
                
                # Évaluer la politique actuelle
                eval_results = evaluate_policy(self, self.env, n_episodes=eval_episodes, max_steps=max_steps)
                eval_results_history.append((episode + 1, eval_results))
                
                # Calculer un score global (moyenne des taux de réussite et collaboration)
                avg_success = sum(eval_results['success_rates']) / self.n_agents
                collab_bonus = eval_results['collaboration_rate']
                current_performance = avg_success + collab_bonus  # Ajuster selon vos priorités
                
                if verbose:
                    print(f"\nÉvaluation à l'épisode {episode + 1}:")
                    print(f"Performance actuelle: {current_performance:.4f}")
                    print(f"Meilleure performance précédente: {self.best_performance:.4f}")
                
                # Sauvegarder la meilleure politique globale
                if current_performance > self.best_performance:
                    self.best_performance = current_performance
                    # Faire une copie profonde des agents
                    self.best_agents = [
                        self._clone_agent(agent) for agent in self.agents
                    ]
                    if verbose:
                        print(f"Nouvelle meilleure politique sauvegardée! Performance: {current_performance:.4f}")
                
                # Sauvegarder également les meilleures politiques individuelles pour chaque agent
                for i in range(self.n_agents):
                    if eval_results['success_rates'][i] > self.best_agent_performances[i]:
                        self.best_agent_performances[i] = eval_results['success_rates'][i]
                        # Cette partie est optionnelle et dépend de vos besoins
                
                # Restaurer les valeurs epsilon d'origine
                for i, agent in enumerate(self.agents):
                    agent.epsilon = original_epsilons[i]

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
            'rewards_rates': rewards_rates,
            'steps': episode_steps,
            'eval_results': eval_results_history
        }

    def _clone_agent(self, agent):
            """Crée une copie profonde d'un agent Q"""
            clone = QAgent(
                state_size=len(agent.q_table),
                action_size=agent.q_table.shape[1],
                learning_rate=agent.lr,            # Utiliser lr au lieu de alpha
                discount_factor=agent.gamma,
                exploration_rate=agent.epsilon,
                min_exploration_rate=agent.min_epsilon,
                exploration_decay=agent.epsilon_decay
            )
            # Copier la table Q
            clone.q_table = np.copy(agent.q_table)
            return clone
    def use_best_policy(self):
        """Remplace les agents actuels par les meilleurs agents sauvegardés"""
        if self.best_agents[0] is None:
            print("Aucune meilleure politique n'a été sauvegardée!")
            return
        
        self.agents = [self._clone_agent(agent) for agent in self.best_agents]
        # Désactiver l'exploration pour les agents
        for agent in self.agents:
            agent.epsilon = 0.0
        
        # print(f"Utilisation de la meilleure politique (performance: {self.best_performance:.4f})")

    def render_policy(self, use_best=True):
        """
        Affiche la politique apprise pour chaque agent
        
        Args:
            use_best: Si True, affiche la meilleure politique sauvegardée
        """
        agents_to_render = self.best_agents if use_best and self.best_agents[0] is not None else self.agents
        
        for agent_idx, agent in enumerate(agents_to_render):
            print(f"\nPolitique de l'agent {agent_idx + 1} {'(meilleure)' if use_best else ''}:")
            grid_size = self.env.grid_size
            policy = np.argmax(agent.q_table, axis=1).reshape(grid_size)

            # Utiliser des symboles pour représenter les actions
            action_symbols = ['←', '↓', '→', '↑']

            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    state_idx = i * grid_size[1] + j
                    print(f" {action_symbols[policy[i, j]]} ", end='')
                print()

import numpy as np
from tqdm import tqdm
import random

class QAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0,
                 min_exploration_rate=0.01, exploration_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size

        # Initialisation de la table Q avec des zéros
        self.q_table = np.zeros((state_size, action_size))

        # Paramètres d'apprentissage
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay

    def get_action(self, state):
        if random.random() < self.epsilon:
            # Action aléatoire (exploration)
            return random.randint(0, self.action_size - 1)
        else:
            # Action optimale selon la table Q (exploitation)
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        # Mise à jour de la table Q selon l'algorithme Q-learning
        # Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]

        # Si l'épisode est terminé, la valeur future est 0
        if done:
            future_q_value = 0
        else:
            future_q_value = np.max(self.q_table[next_state])

        # Calcul du target Q-value
        target = reward + self.gamma * future_q_value

        # Mise à jour de la valeur Q pour l'état et l'action actuels
        current = self.q_table[state, action]
        self.q_table[state, action] = current + self.lr * (target - current)

    def decay_epsilon(self):
        # Réduire epsilon progressivement pour favoriser l'exploitation au fil du temps
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class IndependentQLearning:
    """
    Implémentation simplifiée de Q-Learning indépendant (IQL) pour les environnements multi-agents.
    Chaque agent apprend de manière indépendante sans prendre en compte les autres agents.
    """
    def __init__(self, env, n_agents=2, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.999):
        self.env = env
        self.n_agents = n_agents

        # Obtenir les tailles des espaces d'état et d'action pour chaque agent
        grid_size = env.grid_size
        state_size = grid_size[0] * grid_size[1]
        action_size = 4  # 0: gauche, 1: bas, 2: droite, 3: haut

        # Créer un agent Q pour chaque agent dans l'environnement
        self.agents = [
            QAgent(state_size, action_size, learning_rate, discount_factor,
                   exploration_rate, min_exploration_rate, exploration_decay)
            for _ in range(n_agents)
        ]

    def train(self, episodes=10000, max_steps=100, verbose=True):
        """
        Entraîne les agents de manière indépendante dans l'environnement multi-agent,
        en se concentrant uniquement sur les récompenses.

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
            
            # Enregistrer les récompenses pour chaque agent
            for i in range(self.n_agents):
                rewards_rates[i].append(episode_rewards[i])

            # Réduire epsilon pour chaque agent
            for agent in self.agents:
                agent.decay_epsilon()

            # Afficher la progression
            if verbose and (episode + 1) % (episodes // 10) == 0:
                # Calculer les récompenses moyennes sur les 100 derniers épisodes
                window = 100
                recent_rewards = [
                    np.mean(rewards_rates[i][-min(window, len(rewards_rates[i])):])
                    for i in range(self.n_agents)
                ]
                
                print(f"Épisode {episode + 1}/{episodes}, "
                    f"Récompenses moyennes agents: {[f'{reward:.2f}' for reward in recent_rewards]}, "
                    f"Epsilon: {self.agents[0].epsilon:.4f}")

        return {
            'rewards': rewards_history,
            'rewards_rates': rewards_rates,
            'steps': episode_steps
        }


    def get_action(self, states):
        """Obtient les actions des agents en utilisant leur politique actuelle (sans exploration)"""
        actions = []
        for i, agent in enumerate(self.agents):
            state = states[i]
            # Utiliser directement la politique sans exploration
            action = np.argmax(agent.q_table[state])
            actions.append(action)
        return actions

    def render_policy(self):
        """
        Affiche la politique apprise pour chaque agent
        """
        for agent_idx, agent in enumerate(self.agents):
            print(f"\nPolitique de l'agent {agent_idx + 1}:")
            action_symbols = ['←', '↓', '→', '↑']
            grid_size = self.env.grid_size
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    state_idx = i * grid_size[1] + j
                    
                    # Vérifier si toutes les valeurs Q sont nulles pour cet état
                    if np.all(agent.q_table[state_idx] == 0):
                        print(" o ", end='')  # État jamais visité ou trou
                    else:
                        # Récupérer la meilleure action pour cet état
                        best_action = np.argmax(agent.q_table[state_idx])
                        print(f" {action_symbols[best_action]} ", end='')
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


class AlternatingIQL:
    """
    Implémentation de Q-Learning indépendant (IQL) avec alternance des taux d'apprentissage.
    Cette approche permet de réduire le problème de non-stationnarité en faisant alterner
    les périodes d'apprentissage actif entre les agents.
    """
    def __init__(self, env, n_agents=2, 
                 base_learning_rate=0.1, 
                 discount_factor=0.99,
                 exploration_rate=1.0, 
                 min_exploration_rate=0.01, 
                 exploration_decay=0.999,
                 alternating_period=100,  # Nombre d'épisodes avant l'alternance
                 learning_rate_ratio=0.1  # Ratio pour le taux d'apprentissage réduit
                ):
        self.env = env
        self.n_agents = n_agents
        self.base_lr = base_learning_rate
        self.lr_ratio = learning_rate_ratio
        self.alternating_period = alternating_period
        
        # Obtenir les tailles des espaces d'état et d'action pour chaque agent
        grid_size = env.grid_size
        state_size = grid_size[0] * grid_size[1]
        action_size = 4  # 0: gauche, 1: bas, 2: droite, 3: haut

        # Créer un agent Q pour chaque agent dans l'environnement
        self.agents = [
            QAgent(state_size, action_size, learning_rate=base_learning_rate, 
                   discount_factor=discount_factor, exploration_rate=exploration_rate, 
                   min_exploration_rate=min_exploration_rate, exploration_decay=exploration_decay)
            for _ in range(n_agents)
        ]
        
        # Tracker pour savoir quel agent est en mode apprentissage actif
        self.active_learner_idx = 0
        self.episode_counter = 0

    def _update_learning_rates(self):
        """
        Met à jour les taux d'apprentissage des agents selon le schéma d'alternance.
        """
        # Vérifier si c'est le moment d'alterner
        if self.episode_counter % self.alternating_period == 0:
            # Changement de l'agent actif
            self.active_learner_idx = (self.active_learner_idx + 1) % self.n_agents
            
        # Mettre à jour les taux d'apprentissage pour tous les agents
        for i in range(self.n_agents):
            if i == self.active_learner_idx:
                # Agent actif utilise le taux d'apprentissage de base
                self.agents[i].lr = self.base_lr
            else:
                # Autres agents utilisent un taux d'apprentissage réduit
                self.agents[i].lr = self.base_lr * self.lr_ratio

    def render_policy(self):
        """
        Affiche la politique apprise pour chaque agent
        """
        for agent_idx, agent in enumerate(self.agents):
            print(f"\nPolitique de l'agent {agent_idx + 1}:")
            action_symbols = ['←', '↓', '→', '↑']
            grid_size = self.env.grid_size
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    state_idx = i * grid_size[1] + j
                    
                    # Vérifier si toutes les valeurs Q sont nulles pour cet état
                    if np.all(agent.q_table[state_idx] == 0):
                        print(" o ", end='')  # État jamais visité ou trou
                    else:
                        # Récupérer la meilleure action pour cet état
                        best_action = np.argmax(agent.q_table[state_idx])
                        print(f" {action_symbols[best_action]} ", end='')
                print()

    def train(self, episodes=10000, max_steps=100, verbose=True):
        """
        Entraîne les agents avec alternance des taux d'apprentissage
        """
        import numpy as np
        from tqdm import tqdm
        
        rewards_history = []
        rewards_rates = [[] for _ in range(self.n_agents)]
        episode_steps = []
        learning_rates_history = [[] for _ in range(self.n_agents)]

        if verbose:
            episodes_iter = tqdm(range(episodes))
        else:
            episodes_iter = range(episodes)

        for episode in episodes_iter:
            # Mise à jour des taux d'apprentissage selon le schéma d'alternance
            self._update_learning_rates()
            self.episode_counter += 1
            
            # Enregistrer les taux d'apprentissage actuels
            for i in range(self.n_agents):
                learning_rates_history[i].append(self.agents[i].lr)
            
            # Réinitialiser l'environnement
            states, _ = self.env.reset()
            episode_rewards = [0] * self.n_agents
            dones = [False] * self.n_agents

            for step in range(max_steps):
                # Chaque agent choisit une action
                actions = [
                    self.agents[i].get_action(states[i])
                    for i in range(self.n_agents)
                ]

                # Exécuter les actions dans l'environnement
                next_states, rewards, new_dones, truncated, _ = self.env.step(actions)

                # Mettre à jour les tables Q pour chaque agent
                for i in range(self.n_agents):
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

            # Enregistrer les récompenses pour chaque agent
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
                      f"Agent actif: {self.active_learner_idx}, "
                      f"Taux d'apprentissage: {[f'{self.agents[i].lr:.4f}' for i in range(self.n_agents)]}"
                      f"Epsilon: {self.agents[0].epsilon:.3f}"
                      f"total _reward: {np.sum(recent_rewards):.4f}"
                      )

        return {
            'rewards': rewards_history,
            'rewards_rates': rewards_rates,
            'steps': episode_steps,
            'learning_rates': learning_rates_history
        }




def evaluate_policy(agent, env, n_episodes=100, max_steps=100, results_file=None, display=False, output=False, render=False):
    """
    Évalue la performance d'une politique dans un environnement multi-agent avec plusieurs objectifs.
    Compatible avec IndependentQLearning, CentralizedQLearning et autres agents similaires.

    Args:
        agent: Instance d'agent contenant la politique à évaluer
        env: Environnement (FrozenLakeFlexibleAgents ou FrozenLake4goals)
        n_episodes: Nombre d'épisodes d'évaluation
        max_steps: Nombre maximum d'étapes par épisode
        results_file: Fichier pour sauvegarder les résultats
        display: Si True, affiche les résultats
        output: Si True, écrit les résultats dans results_file
        render: Si True, affiche l'environnement à chaque étape

    Returns:
        dict: Dictionnaire contenant les statistiques d'évaluation
    """
    import numpy as np
    
    # Statistiques générales
    success_rates = [0] * agent.n_agents
    steps_when_reached = [[] for _ in range(agent.n_agents)]
    rewards_total = [[] for _ in range(agent.n_agents)]
    episodes_with_collisions = 0
    total_collisions = 0
    
    # Statistiques pour FrozenLake4goals
    is_multi_goal_env = hasattr(env, 'goal_positions') and isinstance(env.goal_positions, list) and len(env.goal_positions) > 1
    goals_reached_per_agent = [0] * agent.n_agents
    max_possible_goals = len(env.goal_positions) if is_multi_goal_env else 1
    
    # Statistiques de collaboration
    collaborative_successes = 0
    
    if display:
        print(f"\nÉvaluation de la politique sur {n_episodes} épisodes...")
    if output and results_file:
        results_file.write(f"\nÉvaluation de la politique sur {n_episodes} épisodes...\n")

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = [False] * agent.n_agents
        episode_rewards = [0] * agent.n_agents
        episode_goals_reached = [0] * agent.n_agents
        reached_any_goal = [False] * agent.n_agents
        episode_had_collisions = False
        collaboration_this_episode = False
        
        if render:
            env.render()
        
        for step in range(max_steps):
            # Utiliser la politique apprise (sans exploration)
            if hasattr(agent, 'agents'):  # IndependentQLearning ou similaire
                actions = [np.argmax(agent.agents[i].q_table[state[i]]) for i in range(agent.n_agents)]
            else:  # CentralizedQLearning
                joint_state = tuple(state)
                try:
                    joint_action = np.argmax(agent.q_table[joint_state])
                    actions = agent.index_to_actions(joint_action)
                except (KeyError, TypeError):
                    # Cas où l'état n'a pas été visité pendant l'entraînement
                    actions = [0] * agent.n_agents  # Action par défaut

            # Exécuter les actions
            next_state, rewards, new_done, _, info = env.step(actions)

            # Vérifier s'il y a eu des collisions
            if 'collisions' in info and info['collisions']:
                episode_had_collisions = True
                total_collisions += 1

            # Vérifier s'il y a eu une arrivée simultanée
            if 'simultaneous_arrival' in info and info['simultaneous_arrival']:
                collaboration_this_episode = True

            # Accumuler les récompenses et vérifier les objectifs atteints
            for i in range(agent.n_agents):
                episode_rewards[i] += rewards[i]
                
                # Pour environnement multi-objectifs
                if is_multi_goal_env:
                    # Si l'agent a reçu une récompense significative (>50), c'est qu'il a atteint un nouvel objectif
                    if rewards[i] >= 50 and not done[i]:
                        episode_goals_reached[i] += 1
                        reached_any_goal[i] = True
                        steps_when_reached[i].append(step + 1)
                # Pour environnement à objectif unique
                elif rewards[i] > 0 and not reached_any_goal[i]:
                    reached_any_goal[i] = True
                    steps_when_reached[i].append(step + 1)

            if render:
                env.render()
                
            # Mettre à jour les états et les drapeaux terminés
            state = next_state
            done = new_done

            # Sortir de la boucle si tous les agents ont terminé
            if all(done):
                break

        # Enregistrer les statistiques de l'épisode
        if episode_had_collisions:
            episodes_with_collisions += 1
            
        if collaboration_this_episode:
            collaborative_successes += 1

        for i in range(agent.n_agents):
            # Un agent réussit s'il a atteint au moins un objectif
            success_rates[i] += 1 if reached_any_goal[i] else 0
            
            # Comptabiliser le nombre d'objectifs atteints (pour environnements multi-objectifs)
            if is_multi_goal_env:
                goals_reached_per_agent[i] += episode_goals_reached[i]
            
            # Enregistrer la récompense totale
            rewards_total[i].append(episode_rewards[i])

    # Calculer les statistiques finales
    for i in range(agent.n_agents):
        success_rates[i] /= n_episodes
    
    avg_goals_per_agent = [goals / n_episodes for goals in goals_reached_per_agent] if is_multi_goal_env else None
    collision_rate = episodes_with_collisions / n_episodes
    collaboration_rate = collaborative_successes / n_episodes

    # Préparer les résultats
    eval_results = {
        'success_rates': success_rates,
        'avg_steps_to_goal': [np.mean(steps) if steps else float('nan') for steps in steps_when_reached],
        'avg_rewards': [np.mean(rewards) for rewards in rewards_total],
        'collaboration_rate': collaboration_rate,
        'collision_rate': collision_rate,
        'avg_collisions_per_episode': total_collisions / n_episodes
    }
    
    if is_multi_goal_env:
        eval_results['avg_goals_per_agent'] = avg_goals_per_agent
        eval_results['goal_completion_rate'] = [avg / max_possible_goals for avg in avg_goals_per_agent]

    # Afficher les résultats
    if display:
        print("\nRésultats de l'évaluation:")
        if output and results_file:
            results_file.write("\nRésultats de l'évaluation:\n")
            
        for i in range(agent.n_agents):
            agent_results = [
                f"Agent {i+1}:",
                f"  - Taux de réussite: {success_rates[i]*100:.2f}%"
            ]
            
            if is_multi_goal_env:
                agent_results.append(f"  - Objectifs atteints par épisode: {avg_goals_per_agent[i]:.2f}/{max_possible_goals} ({eval_results['goal_completion_rate'][i]*100:.2f}%)")
            
            if steps_when_reached[i]:
                agent_results.append(f"  - Nombre moyen d'étapes (succès): {eval_results['avg_steps_to_goal'][i]:.2f}")
            else:
                agent_results.append(f"  - Nombre moyen d'étapes (succès): N/A (aucun succès)")
                
            agent_results.append(f"  - Récompense moyenne: {eval_results['avg_rewards'][i]:.4f}")
            
            # Afficher et écrire dans le fichier
            for line in agent_results:
                print(line)
                if output and results_file:
                    results_file.write(line + "\n")

        # Résultats généraux
        general_results = [
            f"\nTaux de collaboration: {collaboration_rate*100:.2f}%",
            f"Taux de collision: {collision_rate*100:.2f}%",
            f"Nombre moyen de collisions par épisode: {eval_results['avg_collisions_per_episode']:.4f}"
        ]
        
        for line in general_results:
            print(line)
            if output and results_file:
                results_file.write(line + "\n")

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
        results = cql.train(episodes=max_episodes, max_steps=200)

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
        eval_results = evaluate_policy(cql, env, n_episodes=1000, max_steps=100, results_file=results_file,display=True)
        
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

def test_independent_q_learning(env, n_agents=2,
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
    results_filename = f"results/independent_q_learning_{timestamp}.txt"
    
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

        # Entraîner les agents indépendants
        print("\nDébut de l'entraînement indépendant...")
        results_file.write("\nDébut de l'entraînement indépendant...\n")
        results = iql.train(episodes=max_episodes, max_steps=200)

        # Sauvegarder les métriques d'entraînement
        metrics_file = f"results/independent_q_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            # Créer une copie pour ne pas modifier l'original
            json_results = {}
            
            # Traiter chaque clé disponible
            if 'rewards_rates' in results:
                json_results["rewards_rates"] = [rates.tolist() if isinstance(rates, np.ndarray) else rates for rates in results['rewards_rates']]
            
            # Ajouter d'autres clés présentes dans le dictionnaire
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
        print("\nPolitiques indépendantes apprises:")
        results_file.write("\nPolitiques indépendantes apprises:\n")
        iql.render_policy()

        # Visualiser les résultats
        plt.figure(figsize=(15, 10))

        # Tracer les récompenses pour chaque agent
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
        plot_filename = f"results/independent_q_learning_plot_{timestamp}.png"
        plt.savefig(plot_filename)
        results_file.write(f"\nGraphique sauvegardé dans {plot_filename}\n")
        plt.show()

        print('Test de la politique obtenue - Récompenses uniquement')
        results_file.write('\nTest de la politique obtenue - Récompenses uniquement\n')
        
        # Évaluer la politique avec focus sur les récompenses
        n_test_episodes = 1000
        test_max_steps = 100  # Défini explicitement ici
        rewards_per_agent = [[] for _ in range(iql.n_agents)]
        
        for episode in range(n_test_episodes):
            state, _ = env.reset()
            done = [False] * iql.n_agents
            episode_rewards = [0] * iql.n_agents
            
            for step in range(test_max_steps):  # Utiliser test_max_steps ici
                # Utiliser la politique apprise (sans exploration)
                actions = [np.argmax(iql.agents[i].q_table[state[i]]) for i in range(iql.n_agents)]
                
                # Exécuter les actions dans l'environnement
                next_state, rewards, done, _, _ = env.step(actions)
                
                # Accumuler les récompenses
                for i in range(iql.n_agents):
                    if not done[i]:
                        episode_rewards[i] += rewards[i]
                
                # Mettre à jour l'état
                state = next_state
                
                # Sortir de la boucle si tous les agents ont terminé
                if all(done):
                    break
            
            # Enregistrer les récompenses de cet épisode
            for i in range(iql.n_agents):
                rewards_per_agent[i].append(episode_rewards[i])
        
        # Afficher et enregistrer les résultats des récompenses
        avg_rewards = [np.mean(rewards) for rewards in rewards_per_agent]
        std_rewards = [np.std(rewards) for rewards in rewards_per_agent]
        
        for i in range(iql.n_agents):
            reward_info = f"Agent {i+1}: Récompense moyenne = {avg_rewards[i]:.4f} ± {std_rewards[i]:.4f}"
            print(reward_info)
            results_file.write(reward_info + "\n")
        
        # Sauvegarder les résultats d'évaluation
        eval_results = {
            'avg_rewards': avg_rewards,
            'std_rewards': std_rewards,
            'all_rewards': rewards_per_agent
        }
        
        eval_file = f"results/independent_q_rewards_{timestamp}.json"
        with open(eval_file, "w") as f:
            # Convertir les valeurs numpy en types Python standard pour JSON
            json_eval = {}
            for key, value in eval_results.items():
                if key == 'all_rewards':
                    json_eval[key] = [[float(r) for r in agent_rewards] for agent_rewards in value]
                elif isinstance(value, np.ndarray):
                    json_eval[key] = value.tolist()
                elif isinstance(value, list) and any(isinstance(x, np.number) for x in value):
                    json_eval[key] = [float(x) if isinstance(x, np.number) else x for x in value]
                else:
                    json_eval[key] = value
            json.dump(json_eval, f)
            
        results_file.write(f"\nRésultats d'évaluation sauvegardés dans {eval_file}\n")
        print(f"\nTous les résultats ont été sauvegardés dans le dossier 'results', avec le fichier principal: {results_filename}")

    return iql, results

def test_independent_learning_save(
    env, n_agents=2, learning_rate=0.1, discount_factor=0.99,
    exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.999,
    max_episodes=10000, window_size=1000, eval_frequency=1000, eval_episodes=100
):
    """
    Test l'algorithme de Q-learning indépendant avec sauvegarde de la meilleure politique.
    """
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
            "window_size": window_size,
            "eval_frequency": eval_frequency,
            "eval_episodes": eval_episodes
        }
        
        for key, value in params.items():
            results_file.write(f"{key}: {value}\n")
        
        # Visualiser l'environnement initial
        print("Environnement initial:")
        results_file.write("\nEnvironnement initial:\n")
        env_str = str(env.render())
        results_file.write(env_str + "\n" if env_str else "[Rendu de l'environnement non capturé]\n")
        
        # Créer l'agent IQL
        iql = IndependentQLearning(
            env=env,
            n_agents=n_agents,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            min_exploration_rate=min_exploration_rate,
            exploration_decay=exploration_decay
        )
        
        # Entraîner l'agent
        print("\nDébut de l'entraînement indépendant...")
        results_file.write("\nDébut de l'entraînement indépendant...\n")
        results = iql.train(
            episodes=max_episodes,
            max_steps=100,
            eval_frequency=eval_frequency,
            eval_episodes=eval_episodes,
            verbose=True
        )

        # Sauvegarder les métriques d'entraînement
        metrics_file = f"results/independent_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            # Créez une copie pour ne pas modifier l'original
            json_results = {}
            
            # Traitez chaque clé disponible
            if 'rewards_rates' in results:
                json_results["rewards_rates"] = [rates if not isinstance(rates, np.ndarray) else rates.tolist() 
                                               for rates in results['rewards_rates']]
            
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
        
        # Utiliser la meilleure politique
        iql.use_best_policy()
        
        # Afficher la politique apprise
        print("\nPolitique indépendante apprise (meilleure):")
        results_file.write("\nPolitique indépendante apprise (meilleure):\n")
        iql.render_policy(use_best=True)
        
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
            
            # Ajouter un marqueur pour les évaluations de politique qui ont conduit à une sauvegarde
            if 'eval_results' in results:
                save_episodes = []
                save_performances = []
                
                for ep, eval_result in results['eval_results']:
                    if len(save_episodes) == 0 or eval_result['success_rates'][i] > max([er[1]['success_rates'][i] for er in results['eval_results'][:len(save_episodes)]]):
                        save_episodes.append(ep)
                        # Utiliser la récompense moyenne correspondante comme point sur le graphique
                        idx = min(ep // window_size_plot, len(rewards_rates_smoothed) - 1)
                        if idx >= 0:
                            save_performances.append(rewards_rates_smoothed[idx])
                
                if save_episodes:
                    plt.scatter(save_episodes, save_performances, color='red', marker='*', s=100, 
                                label='Meilleure politique sauvegardée')
                    plt.legend()

        plt.tight_layout()
        plot_filename = f"results/independent_learning_plot_{timestamp}.png"
        plt.savefig(plot_filename)
        results_file.write(f"\nGraphique sauvegardé dans {plot_filename}\n")
        plt.show()
        
        # Évaluer la meilleure politique
        print("\nÉvaluation de la meilleure politique:")
        results_file.write("\nÉvaluation de la meilleure politique:\n")
        final_eval = evaluate_policy(iql, env, n_episodes=100, max_steps=100, 
                                    results_file=results_file, display=True, output=True)
        
        # Sauvegarder les résultats d'évaluation
        eval_file = f"results/independent_eval_{timestamp}.json"
        with open(eval_file, "w") as f:
            # Convertir les valeurs numpy en types Python standard pour JSON
            json_eval = {}
            for key, value in final_eval.items():
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

def test_alternating_learning(env, n_agents=2,
                            base_learning_rate=0.3,
                            max_episodes=50000,
                            discount_factor=0.99,
                            exploration_rate=1.0,
                            min_exploration_rate=0.05,
                            exploration_decay=0.99995,
                            alternating_period=500,
                            learning_rate_ratio=0.1,
                            window_size=200):
    
    # Créer le dossier results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    
    # Nom du fichier de résultats avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/alternating_learning_{timestamp}.txt"
    
    # Ouvrir le fichier de résultats
    with open(results_filename, "w", encoding="utf-8") as results_file:
        # Enregistrer les paramètres
        results_file.write("=== PARAMÈTRES D'APPRENTISSAGE ALTERNANT ===\n")
        params = {
            "n_agents": n_agents,
            "base_learning_rate": base_learning_rate,
            "max_episodes": max_episodes,
            "discount_factor": discount_factor,
            "exploration_rate": exploration_rate,
            "min_exploration_rate": min_exploration_rate,
            "exploration_decay": exploration_decay,
            "alternating_period": alternating_period,
            "learning_rate_ratio": learning_rate_ratio,
            "window_size": window_size
        }
        
        for key, value in params.items():
            results_file.write(f"{key}: {value}\n")

        # Visualiser l'environnement initial
        print("Environnement initial:")
        results_file.write("\nEnvironnement initial:\n")
        env_str = str(env.render())
        results_file.write(env_str + "\n" if env_str else "[Rendu de l'environnement non capturé]\n")

        # Créer l'instance d'apprentissage alternant
        alt_iql = AlternatingIQL(
            env,
            n_agents=n_agents,
            base_learning_rate=base_learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            min_exploration_rate=min_exploration_rate,
            exploration_decay=exploration_decay,
            alternating_period=alternating_period,
            learning_rate_ratio=learning_rate_ratio
        )

        # Entraîner les agents
        print("\nDébut de l'entraînement avec alternance...")
        results_file.write("\nDébut de l'entraînement avec alternance de taux d'apprentissage...\n")
        results = alt_iql.train(episodes=max_episodes, max_steps=100)

        alt_iql.render_policy()
        # Visualiser les résultats
        plt.figure(figsize=(15, 10))

        # Tracer les taux de récompense pour chaque agent
        window_size_plot = window_size
        for i in range(alt_iql.n_agents):
            plt.subplot(alt_iql.n_agents + 1, 1, i+1)  # +1 pour inclure le graphique des taux d'apprentissage

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
        
        # Ajouter un graphique pour les taux d'apprentissage
        plt.subplot(alt_iql.n_agents + 1, 1, alt_iql.n_agents + 1)
        for i in range(alt_iql.n_agents):
            # Calculer les moyennes des taux d'apprentissage par fenêtre pour réduire la taille du graphique
            lr_rates = results['learning_rates'][i]
            plt.plot(range(0, len(lr_rates) * window_size_plot, window_size_plot), 
                    lr_rates, label=f"Agent {i+1}")
        
        plt.title("Taux d'apprentissage des agents (alternance)")
        plt.xlabel("Épisodes")
        plt.ylabel("Taux d'apprentissage")
        plt.legend()

        plt.tight_layout()
        plot_filename = f"results/alternating_learning_plot_{timestamp}.png"
        plt.savefig(plot_filename)
        results_file.write(f"\nGraphique sauvegardé dans {plot_filename}\n")
        plt.show()
        
        print('Test de la politique obtenue')
        results_file.write('\nTest de la politique obtenue\n')
        
        # Utilisation de la même fonction evaluate_policy que pour IQL standard
        eval_results = evaluate_policy(alt_iql, env, n_episodes=1000, max_steps=100, results_file=results_file)
        
        # Sauvegarder les résultats d'évaluation
        eval_file = f"results/alternating_eval_{timestamp}.json"
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

    return alt_iql, results
if __name__ == "__main__":
    # Number of agents
    n_agents = 2

    # Environment setup
    env = FrozenLakeFlexibleAgentsEnvCol(
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
    iql, iql_results = test_independent_q_learning(
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
