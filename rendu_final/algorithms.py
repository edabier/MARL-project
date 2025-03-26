import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time
import random
import sys
import pygame
from collections import defaultdict
import os
import colorsys
from tqdm import tqdm


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
    def train(self, env, episodes=1000, max_steps=100, verbose=False, verbose_interval=100):
        """
        Entraîne l'agent dans l'environnement donné.
        
        Args:
            env: Environnement d'apprentissage (doit avoir des méthodes reset et step)
            episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximum d'étapes par épisode
            verbose: Si True, affiche des informations sur l'entraînement
            verbose_interval: Intervalle d'épisodes pour afficher les informations
        
        Returns:
            dict: Statistiques d'entraînement (récompenses, étapes, etc.)
        """
        rewards_history = []
        steps_history = []
        success_history = []
        
        for episode in range(episodes):
            # Réinitialiser l'environnement
            state, _ = env.reset()
            total_reward = 0
            done = False
            truncated = False
            
            for step in range(max_steps):
                # Sélectionner une action
                action = self.get_action(state)
                
                # Exécuter l'action
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Mettre à jour la table Q
                self.update(state, action, reward, next_state, done)
                
                # Mettre à jour l'état et la récompense
                state = next_state
                total_reward += reward
                
                # Sortir de la boucle si l'épisode est terminé
                if done or truncated:
                    break
            
            # Réduire epsilon
            self.decay_epsilon()
            
            # Enregistrer les statistiques
            rewards_history.append(total_reward)
            steps_history.append(step + 1)
            success_history.append(total_reward > 0)  # Considérer une récompense positive comme un succès
            
            # Afficher les statistiques
            if verbose and (episode + 1) % verbose_interval == 0:
                recent_rewards = rewards_history[-verbose_interval:]
                recent_success = success_history[-verbose_interval:]
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                success_rate = sum(recent_success) / len(recent_success) * 100
                
                print(f"Épisode {episode + 1}/{episodes}, "
                    f"Récompense moyenne: {avg_reward:.2f}, "
                    f"Taux de succès: {success_rate:.1f}%, "
                    f"Epsilon: {self.epsilon:.4f}")
        
        # Calculer les statistiques finales
        overall_success_rate = sum(success_history) / len(success_history) * 100
        
        if verbose:
            print(f"\nEntraînement terminé!")
            print(f"Récompense moyenne: {sum(rewards_history) / len(rewards_history):.2f}")
            print(f"Nombre moyen d'étapes: {sum(steps_history) / len(steps_history):.2f}")
            print(f"Taux de succès global: {overall_success_rate:.1f}%")
        
        return {
            'rewards': rewards_history,
            'steps': steps_history,
            'success_rate': overall_success_rate,
            'final_epsilon': self.epsilon
        }

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
                total_rewards=sum(recent_rewards)
                print(f"Épisode {episode + 1}/{episodes}, "
                    f"Récompenses moyennes agents: {[f'{reward:.2f}' for reward in recent_rewards]}, "
                    f"Epsilon: {self.agents[0].epsilon:.4f}",
                    f"Récompense Total {total_rewards:.2f}")

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
                      f"\nEpsilon: {self.agents[0].epsilon:.3f} "
                      f"total _reward: {np.sum(recent_rewards):.4f}"
                      )

        return {
            'rewards': rewards_history,
            'rewards_rates': rewards_rates,
            'steps': episode_steps,
            'learning_rates': learning_rates_history
        }


class RandomPolicy:
    
    '''
    This class creates a random policy in order to perform comparisons 
    with more complex policies
    '''
    
    def __init__(self, action_size=4):
        self.action_size = action_size
    
    def select_action(self, state):
        action1 = np.random.randint(0, self.action_size)
        action2 = np.random.randint(0, self.action_size)
        return (action1, action2)
    
class SingleGoalCentralQLearning:
    
    '''
    This class defines the central Q learning algorithm
    To be used on the OneGoalMultiAgentFrozenLake env
    '''
    
    def __init__(self, state_size, action_size, num_agents=2, learning_rate=0.1, discount_factor=0.99, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        # Add num_agents as a parameter
        self.num_agents = num_agents
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        
        # Create a joint Q-table that can handle any number of agents
        # We'll use a dictionary for states and a numpy array for joint actions
        self.q_table = defaultdict(lambda: np.zeros([action_size] * num_agents))
        
    def select_action(self, state):
        state_tuple = tuple(state)  # Convert state to tuple for dictionary key
        
        # Exploration-exploitation trade-off
        if np.random.random() < self.epsilon:
            # Random actions for all agents
            actions = tuple(np.random.randint(0, self.action_size) for _ in range(self.num_agents))
        else:
            # Greedy action selection
            # Use np.unravel_index with the shape based on num_agents
            actions = np.unravel_index(
                np.argmax(self.q_table[state_tuple]), 
                [self.action_size] * self.num_agents
            )   
        
        return actions
        
    def update(self, state, action, reward, next_state, done):
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)
        
        # Index into the Q-table using the full joint action
        # We need to convert action tuple to a tuple of ints for proper indexing
        action_tuple = tuple(int(a) for a in action)
        
        # Current Q-value
        current_q = self.q_table[state_tuple][action_tuple]
        
        # Next Q-value (maximum over all joint actions)
        next_q = np.max(self.q_table[next_state_tuple]) if not done else 0
        
        # Q-value update
        new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
        self.q_table[state_tuple][action_tuple] = new_q
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
