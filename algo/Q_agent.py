#@title Qagent
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


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

    def render_policy(self, grid_size):
        """
        Affiche la politique apprise (meilleure action pour chaque état)
        """
        print("\nPolitique apprise:")
        
        # Utiliser des symboles pour représenter les actions
        action_symbols = ['←', '↓', '→', '↑']
        
        # Récupérer la politique (meilleure action pour chaque état)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                state_idx = i * grid_size[1] + j
                
                # Vérifier si toutes les valeurs Q sont nulles pour cet état
                if np.all(self.q_table[state_idx] == 0):
                    print(" o ", end='')  # État jamais visité ou trou
                else:
                    # Récupérer la meilleure action pour cet état
                    best_action = np.argmax(self.q_table[state_idx])
                    print(f" {action_symbols[best_action]} ", end='')
            print()

from goals_env_4 import FrozenLake4goals
def test_qagent_frozen_lake():
    # Créer l'environnement FrozenLake
    # env = gym.make('FrozenLake-v1', is_slippery=False)
    env=FrozenLake4goals( num_agents=1, grid_size=(6, 6), slip_prob=0., hole_prob=0.2, seed=9, 
                 collaboration_bonus=0, collision_penalty=0)
    env.render()
    # Initialiser l'agent
         # Nombre d'actions possibles
    grid_size = env.grid_size
    state_size = grid_size[0] * grid_size[1]
    action_size = 4 

    learning_rate=0.3
    discount_factor=0.99
    exploration_rate=1.0
    min_exploration_rate=0.1
    exploration_decay=0.9995

    agent = QAgent(state_size, action_size,learning_rate=learning_rate, discount_factor=discount_factor,
                   exploration_rate=exploration_rate,
                 min_exploration_rate=min_exploration_rate, exploration_decay=exploration_decay)

    # Paramètres d'entraînement
    episodes = 10000
    max_steps = 100

    # Suivi des performances
    rewards_per_episode = []
    success_history = []  # Garder trace des succès (1) ou échecs (0)

    # Entraînement
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        for step in range(max_steps):
            # Choisir une action
            action = agent.get_action(state)

            # Exécuter l'action
            next_state_tuple, reward_list, done_list, truncated_list, _ = env.step([action])
            
            # Extract values for the first agent
            next_state = next_state_tuple[0]
            reward = reward_list[0]
            done = done_list[0]
            truncated = truncated_list[0]

            # Mettre à jour la table Q
            agent.update(state, action, reward, next_state, done)

            # Mise à jour de l'état et de la récompense
            state = next_state
            total_reward += reward

            if done or truncated:
                break

        # Enregistrer les résultats
        rewards_per_episode.append(total_reward)
        success_history.append(1 if total_reward > 0 else 0)

        # Réduire epsilon
        agent.decay_epsilon()

        # Afficher la progression
        if (episode + 1) % 1000 == 0:
            success_rate = sum(success_history[-1000:]) / 1000
            print(f"Épisode {episode + 1}/{episodes}, Taux de réussite sur les 1000 derniers épisodes: {success_rate:.2f}, Epsilon: {agent.epsilon}")

    # Calculer le taux de réussite moyen par fenêtre de 100 épisodes
    window_size = 100
    success_rates = []
    for i in range(0, episodes, window_size):
        if i + window_size <= episodes:
            window_success = sum(success_history[i:i+window_size]) / window_size
            success_rates.append(window_success)

    env.reset()
    env.render()
    agent.render_policy(grid_size)
    # Tracer les résultats
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, episodes, window_size), success_rates)
    plt.title('Success rate per 100 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success rate')

    # Visualize the Q-table (adapt for proper grid_size)
    plt.subplot(1, 2, 2)
    q_values = agent.q_table.max(axis=1).reshape(grid_size[0], grid_size[1])
    plt.imshow(q_values, cmap='hot')
    plt.colorbar()
    plt.title('Maximum Q-values per state')

    plt.tight_layout()
    plt.savefig('qagent_results.png')
    plt.show()
from multi_agent_frozen_env import FrozenLakeFlexibleAgentsEnvCol,FrozenLake4goals

def test_qagent_frozen_lake_flexible():
    # Créer l'environnement FrozenLake
    # env = gym.make('FrozenLake-v1', is_slippery=False)
    env=FrozenLakeFlexibleAgentsEnvCol( num_agents=1, grid_size=(8, 8), slip_prob=0., hole_prob=0.4, seed=7, 
                 collaboration_bonus=0, )
    env.render()
    # Initialiser l'agent
         # Nombre d'actions possibles
    grid_size = env.grid_size
    state_size = grid_size[0] * grid_size[1]
    action_size = 4 

    learning_rate=0.3
    discount_factor=0.99
    exploration_rate=1.0
    min_exploration_rate=0.05
    exploration_decay=0.9995

    agent = QAgent(state_size, action_size,learning_rate=learning_rate, discount_factor=discount_factor,
                   exploration_rate=exploration_rate,
                 min_exploration_rate=min_exploration_rate, exploration_decay=exploration_decay)

    # Paramètres d'entraînement
    episodes = 100000
    max_steps = 200

    # Suivi des performances
    rewards_per_episode = []
    success_history = []  # Garder trace des succès (1) ou échecs (0)

    # Entraînement
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        for step in range(max_steps):
            # Choisir une action
            action = agent.get_action(state)

            # Exécuter l'action
            next_state_tuple, reward_list, done_list, truncated_list, _ = env.step([action])
            
            # Extract values for the first agent
            next_state = next_state_tuple[0]
            reward = reward_list[0]
            done = done_list[0]
            truncated = truncated_list[0]

            # Mettre à jour la table Q
            agent.update(state, action, reward, next_state, done)

            # Mise à jour de l'état et de la récompense
            state = next_state
            total_reward += reward

            if done or truncated:
                break

        # Enregistrer les résultats
        rewards_per_episode.append(total_reward)
        success_history.append(1 if total_reward > 0 else 0)

        # Réduire epsilon
        agent.decay_epsilon()

        # Afficher la progression
        if (episode + 1) % 1000 == 0:
            success_rate = sum(success_history[-100:]) / 100
            print(f"Épisode {episode + 1}/{episodes}, Taux de réussite sur les 100 derniers épisodes: {success_rate:.2f}, Epsilon: {agent.epsilon}")

    # Calculer le taux de réussite moyen par fenêtre de 100 épisodes
    window_size = 100
    success_rates = []
    for i in range(0, episodes, window_size):
        if i + window_size <= episodes:
            window_success = sum(success_history[i:i+window_size]) / window_size
            success_rates.append(window_success)

    # Tracer les résultats
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, episodes, window_size), success_rates)
    plt.title('Success rate per 100 episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success rate')

    # Visualize the Q-table (adapt for proper grid_size)
    plt.subplot(1, 2, 2)
    q_values = agent.q_table.max(axis=1).reshape(grid_size[0], grid_size[1])
    plt.imshow(q_values, cmap='hot')
    plt.colorbar()
    plt.title('Maximum Q-values per state')

    plt.tight_layout()
    plt.savefig('qagent_results.png')
    plt.show()


if __name__ == "__main__":
    test_qagent_frozen_lake()
    # test_qagent_frozen_lake_flexible()