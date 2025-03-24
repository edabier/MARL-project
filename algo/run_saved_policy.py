from goals_env_4 import FrozenLake4goals
from IQL_CQL import CentralizedQLearning,IndependentQLearning,IndependentQLearningSave,AlternatingIQL,test_independent_learning_save,test_independent_q_learning,test_centralized_learning,test_alternating_learning
from test_collab_reward import visualize_policy_pygame,load_agent,list_saved_agents


"""goal run save policy the saved policy should have the same env has the env we are running now
"""
if __name__ == "__main__":
    
    
    agent_files = list_saved_agents("saved_agents")
    if agent_files:
        choice = int(input("\nChoisissez un agent à charger (entrez le numéro): "))
        if 1 <= choice <= len(agent_files):
            selected_file = agent_files[choice - 1]
            
            # Charger l'agent et les paramètres de l'environnement
            loaded_agent, env_params = load_agent(selected_file)
            
            # Recréer l'environnement avec les mêmes paramètres
            recreated_env = FrozenLake4goals(**env_params)
            
            if recreated_env:
                # Visualiser la politique de l'agent dans l'environnement recréé
                visualize_policy_pygame(
                    env=recreated_env, 
                    agent=loaded_agent, 
                    max_steps=50,
                    delay=0.5,
                    screen_size=600,
                    save_images=True
                )




