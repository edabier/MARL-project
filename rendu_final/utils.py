import os
import pickle
import datetime

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
