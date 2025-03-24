from multi_agent_frozen import FrozenLakeFlexibleAgentsEnvCol
from goals_env_4 import FrozenLake4goals
from IQL_CQL import test_independent_q_learning,IndependentQLearning,CentralizedQLearning,test_centralized_learning
from test_collab_reward import visualize_policy_pygame

def create_map_from_string(map_string, env_type="flexible"):
    """
    Create a map configuration from a string representation.
    
    The string uses the following symbols:
    'F' or '.' - Frozen surface (normal tile)
    'H' or 'h' - Hole
    'G' or 'g' - Goal
    'A' or 'a' - Agent start position (multiple allowed)
    '/' - Row separator
    
    Args:
        map_string: String representation of the map
        env_type: Type of environment to create ('flexible' or '4goals')
    
    Returns:
        An instance of the requested environment with the specified map
    """
    # Split the string into rows
    rows = map_string.strip().split('/')
    
    # Determine grid size
    height = len(rows)
    width = max(len(row) for row in rows)
    
    # Normalize row lengths
    normalized_rows = [row.ljust(width, '.') for row in rows]
    
    # Map tiles
    holes = []
    goals = []
    agent_starts = []
    
    # Parse the map
    for i, row in enumerate(normalized_rows):
        for j, cell in enumerate(row):
            if cell in 'Hh':
                holes.append((i, j))
            elif cell in 'Gg':
                goals.append((i, j))
            elif cell in 'Aa':
                agent_starts.append((i, j))
    
    # Create the environment based on the type
    if env_type.lower() == "flexible":
        # For FrozenLakeFlexibleAgentsEnvCol, set a single goal
        goal_pos = goals[0] if goals else None
        num_agents = len(agent_starts)
        
        env = FrozenLakeFlexibleAgentsEnvCol(
            num_agents=num_agents,
            grid_size=(height, width),
            slip_prob=0.0,  # Can be customized
            hole_prob=0.0,  # We'll set holes manually
            seed=None,
            collaboration_bonus=1.0,
            collision_penalty=0.0
        )
        
        # Override the default hole positions and goal
        env.holes = set(holes)
        env.goal_pos = goal_pos
        
        # Set custom agent start positions
        env.agent_starts = agent_starts
        
    elif env_type.lower() == "4goals":
        # For FrozenLake4goals
        num_agents = len(agent_starts)
        
        env = FrozenLake4goals(
            num_agents=num_agents,
            grid_size=(height, width),
            slip_prob=0.0,  # Can be customized
            hole_prob=0.0,  # We'll set holes manually
            seed=None,
            collaboration_bonus=1.0,
            collision_penalty=0.3
        )
        
        # Override the default hole positions
        env.holes = set(holes)
        
        # Set custom goal positions (up to 4)
        if goals:
            env.goal_positions = goals[:4]
        
        # Set custom agent start positions
        env.agent_starts = agent_starts
    
    # Reset the environment to apply changes
    env.reset()
    
    return env


# Example usage
if __name__ == "__main__":
    n_agents=2
    CQL=True
    IQL=True
    # Example map string for FrozenLakeFlexibleAgentsEnvCol
    # A = agent start, G = goal, H = hole, . or F = frozen surface
    flexible_map = "....H..A/A...H.../...H..../.....H../...H..../HHH...H./.H..H.H./...H...G"
 
    
    
    # Create environments from the maps
    env = create_map_from_string(flexible_map, "flexible")
    env.render()
    env.reset()

    if IQL:
        learning_rate_iql = 0.4
        max_episodes_iql = 300000
        discount_factor_iql = 0.99
        exploration_rate_iql = 1.0
        min_exploration_rate_iql = 0.05
        exploration_decay_iql = 0.99997
        window_size_iql = int(max_episodes_iql/200)
        env.reset()
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
    if CQL:
        env.reset()
        learning_rate = 0.4
        max_episodes=200000
        discount_factor = 0.99
        exploration_rate = 1.0
        min_exploration_rate = 0.05
        exploration_decay = 0.99999
        window_size=int(max_episodes/200)
        cql,results=test_centralized_learning(env,n_agents=2,
                                            learning_rate=learning_rate,
                discount_factor=discount_factor,
                exploration_rate=exploration_rate,
                min_exploration_rate=min_exploration_rate,
                exploration_decay=exploration_decay,
                max_episodes=max_episodes,
                window_size=window_size)
   
        
  
