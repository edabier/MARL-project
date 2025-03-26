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

# Define custom maps for testing purposes
MAPS = {
    "4x4": [
        "SFFF",
        "FHHS",
        "FFFH",
        "HFFG"
    ],
    "5x5": [
        "SFFFH",
        "HFFHF",
        "FFFFH",
        "HFHFF",
        "SFHFG"
    ],
    "6x6": [
        "SFFFHF",
        "HFFHFF",
        "FFHFHF",
        "HFFFHH",
        "FFHFHF",
        "SFFFFG"
    ],
    "7x7": [
        "SFFFFFH",
        "HFFHFHF",
        "FFHFFFH",
        "HFFFHFF",
        "FHFHFHF",
        "HFFHFHS",
        "GFFFFFH"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "SHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

def createMap(num_agent, size, map_name=None, seed=None):
    """
    Generate a random FrozenLake map with a feasible path for each agent to the goal.

    Args:
        num_agent (int): Number of agents (starting positions).
        size (int): Grid size (size x size).
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of strings representing the generated map.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if map_name is not None:
        return MAPS[map_name]
    
    # Define elements
    FLOOR, HOLE, START, GOAL = 'F', 'H', 'S', 'G'
    
    # Step 1: Create an empty grid filled with 'F' (frozen floor)
    grid = np.full((size, size), FLOOR)

    # Step 2: Place the goal randomly
    goal_pos = (random.randint(0, size - 1), random.randint(0, size - 1))
    grid[goal_pos] = GOAL

    # Step 3: Randomly place start positions, ensuring they are unique and not on the goal
    start_positions = set()
    while len(start_positions) < num_agent:
        start_pos = (random.randint(0, size - 1), random.randint(0, size - 1))
        if start_pos != goal_pos:
            start_positions.add(start_pos)

    for start_pos in start_positions:
        grid[start_pos] = START

    # Step 4: Randomly place holes
    num_holes = max(1, int(0.2 * size * size))  # 20% of the grid is holes
    hole_positions = set()
    
    while len(hole_positions) < num_holes:
        hole_pos = (random.randint(0, size - 1), random.randint(0, size - 1))
        if hole_pos != goal_pos and hole_pos not in start_positions:
            hole_positions.add(hole_pos)
    
    for hole_pos in hole_positions:
        grid[hole_pos] = HOLE

    # Step 5: Check if all agents can reach the goal
    def is_path_exists(start, goal, grid):
        """Check if there's a valid path using BFS."""
        rows, cols = grid.shape
        queue = [start]
        visited = set()

        while queue:
            x, y = queue.pop(0)
            if (x, y) == goal:
                return True
            visited.add((x, y))

            # Possible moves (Left, Down, Right, Up)
            for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                    if grid[nx, ny] != HOLE:
                        queue.append((nx, ny))

        return False

    # Ensure every agent can reach the goal
    for start in start_positions:
        if not is_path_exists(start, goal_pos, grid):
            return createMap(num_agent, size, seed + 1 if seed is not None else None)  # Retry with a different seed

    # Convert to list of strings
    return ["".join(row) for row in grid]

class FrozenLakeOneGoal(gym.Env):
    def __init__(self, map_, max_steps=100, num_agents=2):
        # Load the map
        self.desc = np.asarray(map_, dtype='c')
        self.nrow, self.ncol = self.desc.shape
        self.original_desc = self.desc.copy()  # Keep original for reference
        self.num_agents = num_agents
        
        # Define action and observation spaces
        # Create a tuple of Discrete(4) for each agent
        self.action_space = spaces.Tuple(tuple(spaces.Discrete(4) for _ in range(num_agents)))
        
        # State space: (agent1_row, agent1_col, agent2_row, agent2_col, ..., agentN_row, agentN_col)
        # Create a tuple of Discrete spaces for each agent's position
        self.observation_space = spaces.Tuple(
            tuple(spaces.Discrete(self.nrow) for _ in range(num_agents)) +  # rows
            tuple(spaces.Discrete(self.ncol) for _ in range(num_agents))    # columns
        )
        
        self.max_steps = max_steps 
        self.np_random = np.random.RandomState()
        
        # Need to import colorsys for agent color generation
        import colorsys
        self.colorsys = colorsys
        
        self.reset()
        
    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        self.desc = self.original_desc.copy()
        start_positions = np.argwhere(self.desc == b'S')
        goal_positions = np.argwhere(self.desc == b'G')
        
        # Initialize agent positions
        self.agent_positions = []
        
        # If we have enough start positions for all agents
        if len(start_positions) >= self.num_agents:
            for i in range(self.num_agents):
                self.agent_positions.append(start_positions[i % len(start_positions)])
        else:
            # If not enough start positions, place all agents at the first start position
            for i in range(self.num_agents):
                self.agent_positions.append(start_positions[0].copy())
        
        # Initialize goal positions
        if len(goal_positions) > 0:
            self.goal_pos = goal_positions[0]  # Use the first goal for simplicity
        else:
            raise ValueError("No goal position found on the map")
        
        self.steps = 0
        self.agents_goal_steps = [None] * self.num_agents
        
        # Flatten the state: [agent1_row, agent1_col, agent2_row, agent2_col, ...]
        self.state = tuple(pos for agent_pos in self.agent_positions for pos in agent_pos)
        
        # Initialize last actions for all agents (default to UP)
        self.last_actions = [3] * self.num_agents
        
        return self.state, {}
    
    def step(self, actions):
        self.steps += 1
        
        # Ensure actions is a list or tuple with the right length
        if isinstance(actions, (int, np.integer)):  # Handle single agent case
            actions = [actions]
        elif not isinstance(actions, (list, tuple, np.ndarray)) or len(actions) != self.num_agents:
            raise ValueError(f"Expected actions to be a list/tuple of length {self.num_agents}, got {actions}")
        
        # Store last actions for rendering
        self.last_actions = list(actions)
        
        # On first step, convert all 'S' to 'F' after agents have moved
        if self.steps == 1:
            start_positions = np.argwhere(self.desc == b'S')
            for pos in start_positions:
                self.desc[pos[0], pos[1]] = b'F'
        
        # Move all agents
        for i, action in enumerate(actions):
            if i < len(self.agent_positions):  # Ensure we don't go out of bounds
                self.agent_positions[i] = self._move_agent(self.agent_positions[i], action)
        
        # Check if agents have reached goals
        dones = []
        for i, agent_pos in enumerate(self.agent_positions):
            done = np.array_equal(agent_pos, self.goal_pos)
            dones.append(done)
            if done and self.agents_goal_steps[i] is None:
                self.agents_goal_steps[i] = self.steps
        
        # Update state
        self.state = tuple(pos for agent_pos in self.agent_positions for pos in agent_pos)
        
        # Check if any agent has fallen into a hole
        fallen_agents = []
        for i, agent_pos in enumerate(self.agent_positions):
            if self.desc[agent_pos[0], agent_pos[1]] == b'H':
                fallen_agents.append(i)
        
        # Determine overall done condition
        done = (all(dones) or self.steps >= self.max_steps or len(fallen_agents) > 0)
        
        # Calculate reward
        if len(fallen_agents) > 0:
            reward = -5.0  # Immediate failure if any agent falls
        elif all(dones):
            # Check if all agents reached the goal at the same time
            goal_steps = [step for step in self.agents_goal_steps if step is not None]
            if len(goal_steps) == self.num_agents and len(set(goal_steps)) == 1:
                reward = 1.0  # Full reward if they all arrive together
            else:
                reward = 0.5  # Partial reward if they arrive at different times
        elif any(dones):
            reward = -0.2  # Penalize reaching the goal alone
        else:
            reward = -0.001  # Small penalty to encourage movement
        
        truncated = self.steps >= self.max_steps
        
        return self.state, reward, done, truncated, {}
    
    def _move_agent(self, position, action):
        # Get new position
        new_position = position.copy()
        
        # 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
        if action == 0:  # LEFT
            new_position[1] = max(0, position[1] - 1)
        elif action == 1:  # DOWN
            new_position[0] = min(self.nrow - 1, position[0] + 1)
        elif action == 2:  # RIGHT
            new_position[1] = min(self.ncol - 1, position[1] + 1)
        elif action == 3:  # UP
            new_position[0] = max(0, position[0] - 1)
        
        # Check if new position is a hole or valid
        if self.desc[new_position[0], new_position[1]] != b'H':
            return new_position
        else:
            # If it's a hole, agent falls in
            return new_position

    def render_pygame(self, screen_size=400):
        """Render the environment using pygame with the original gym images"""
        # Initialize pygame if not already done
        if not hasattr(self, 'pygame_initialized') or not self.pygame_initialized:
            pygame.init()
            self.pygame_initialized = True
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            pygame.display.set_caption(f"Multi-Agent Frozen Lake ({self.num_agents} agents)")
            self.cell_size = screen_size // max(self.nrow, self.ncol)
            self.running = True  # Flag to control pygame loop
            
            # Load images from gym repository
            img_dir = "img/"
            self.images = {
                'F': pygame.image.load(img_dir + "ice.png"),
                'H': pygame.image.load(img_dir + "hole.png"),
                'G': pygame.image.load(img_dir + "ice.png"),  # Use ice as background for goal
                'S': pygame.image.load(img_dir + "stool.png")
            }
            
            # Load goal sprite separately to overlay on ice
            self.goal_sprite = pygame.image.load(img_dir + "goal.png")
            self.goal_sprite = pygame.transform.scale(self.goal_sprite, (self.cell_size, self.cell_size))
            
            # Load agent images for different directions
            self.agent_images = {
                'up': pygame.image.load(img_dir + "elf_up.png"),
                'down': pygame.image.load(img_dir + "elf_down.png"),
                'left': pygame.image.load(img_dir + "elf_left.png"),
                'right': pygame.image.load(img_dir + "elf_right.png")
            }
            
            # Resize images to fit the cell size
            for key in self.images:
                self.images[key] = pygame.transform.scale(self.images[key], (self.cell_size, self.cell_size))
            
            for key in self.agent_images:
                self.agent_images[key] = pygame.transform.scale(self.agent_images[key], (self.cell_size, self.cell_size))
            
            # Create red-tinted version for collision
            self.collision_images = {}
            
            for key, img in self.agent_images.items():
                # Create red-tinted version for collision
                self.collision_images[key] = img.copy()
                red_surface = pygame.Surface(img.get_size(), pygame.SRCALPHA)
                red_surface.fill((255, 0, 0, 100))  # Red tint
                self.collision_images[key].blit(red_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Define agent indicator colors - generate a unique color for each agent
            self.agent_colors = []
            for i in range(self.num_agents):
                # Generate a unique color based on the agent index
                hue = i / max(1, self.num_agents)
                rgb = self.colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                self.agent_colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
        
        # Don't render if pygame has been closed
        if not hasattr(self, 'running') or not self.running:
            return
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw grid
        for i in range(self.nrow):
            for j in range(self.ncol):
                tile_char = self.desc[i, j].decode('utf-8')
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size,
                                self.cell_size, self.cell_size)
                # Draw tile
                self.screen.blit(self.images[tile_char], rect)
                
                # Overlay goal sprite on ice if this is the goal position
                if tile_char == 'G':
                    self.screen.blit(self.goal_sprite, rect)
        
        # Draw grid lines
        grid_color = (50, 50, 50)  # Dark gray
        for i in range(self.nrow + 1):
            pygame.draw.line(self.screen, grid_color, 
                            (0, i * self.cell_size), 
                            (self.ncol * self.cell_size, i * self.cell_size), 2)
        for j in range(self.ncol + 1):
            pygame.draw.line(self.screen, grid_color, 
                            (j * self.cell_size, 0), 
                            (j * self.cell_size, self.nrow * self.cell_size), 2)
        
        # Map actions to directions for agent images
        direction_map = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
        
        # Check for collisions (where 2 or more agents occupy the same cell)
        # Create a dictionary to track cell occupancy
        cell_occupancy = {}
        for i, agent_pos in enumerate(self.agent_positions):
            pos_tuple = tuple(agent_pos)
            if pos_tuple in cell_occupancy:
                cell_occupancy[pos_tuple].append(i)
            else:
                cell_occupancy[pos_tuple] = [i]
        
        # Draw agents
        for i, agent_pos in enumerate(self.agent_positions):
            pos_tuple = tuple(agent_pos)
            agent_rect = pygame.Rect(agent_pos[1] * self.cell_size, 
                                   agent_pos[0] * self.cell_size,
                                   self.cell_size, self.cell_size)
            
            # Make sure we have a valid action index
            action_idx = min(self.last_actions[i], 3) if i < len(self.last_actions) else 3
            
            # Check if this agent is colliding with others
            is_collision = len(cell_occupancy[pos_tuple]) > 1
            
            if is_collision:
                # Use red-tinted image for collisions
                collision_img = self.collision_images[direction_map[action_idx]]
                self.screen.blit(collision_img, agent_rect)
                
                # Draw indicators for all colliding agents
                colliding_agents = cell_occupancy[pos_tuple]
                for idx, colliding_agent_idx in enumerate(colliding_agents):
                    # Position indicators at different corners
                    indicator_size = max(4, int(self.cell_size / 8))
                    
                    # Calculate position based on index (up to 4 agents per cell)
                    corner_idx = idx % 4
                    if corner_idx == 0:  # Top-left
                        pos = (agent_rect.left + indicator_size, agent_rect.top + indicator_size)
                    elif corner_idx == 1:  # Top-right
                        pos = (agent_rect.right - indicator_size, agent_rect.top + indicator_size)
                    elif corner_idx == 2:  # Bottom-left
                        pos = (agent_rect.left + indicator_size, agent_rect.bottom - indicator_size)
                    else:  # Bottom-right
                        pos = (agent_rect.right - indicator_size, agent_rect.bottom - indicator_size)
                    
                    # Draw the indicator
                    if colliding_agent_idx < len(self.agent_colors):
                        pygame.draw.circle(self.screen, self.agent_colors[colliding_agent_idx], pos, indicator_size)
            else:
                # Draw agent normally
                agent_img = self.agent_images[direction_map[action_idx]]
                self.screen.blit(agent_img, agent_rect)
                
                # Draw a small indicator for agent identification
                indicator_size = max(4, int(self.cell_size / 8))
                pygame.draw.circle(self.screen, self.agent_colors[i], 
                                (agent_rect.left + indicator_size, agent_rect.top + indicator_size), 
                                indicator_size)
        
        # Update display
        pygame.display.flip()
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return
            
    def close(self):
        """Properly close the pygame window"""
        if hasattr(self, 'pygame_initialized') and self.pygame_initialized:
            self.running = False
            pygame.quit()
            self.pygame_initialized = False      

class FrozenLake4goals(gym.Env):
    """
    A multi-agent FrozenLake environment where multiple agents navigate a frozen grid.
    The agents receive a bonus reward if they reach the goal at the same time.
    Agents are penalized for collisions when they occupy the same cell.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_agents=2, grid_size=(5, 5), slip_prob=0.2, hole_prob=0.2, seed=None, 
                 collaboration_bonus=1.0, collision_penalty=0.3):
        super(FrozenLake4goals, self).__init__()
        self.grid_size = grid_size
        self.slip_prob = slip_prob  # Probability of slipping
        self.hole_prob = hole_prob  # Probability of each cell being a hole
        self.collaboration_bonus = collaboration_bonus  # Bonus reward for simultaneous goal arrival
        self.num_agents = num_agents
        self.collision_penalty = collision_penalty  # Penalty for agents colliding

        # Set seed for reproducibility
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            self.random_gen = random.Random(seed)
        else:
            self.np_random = np.random.RandomState()
            self.random_gen = random.Random()

        # Actions for all agents (0: left, 1: down, 2: right, 3: up)
        self.action_space = spaces.MultiDiscrete([4] * num_agents)

        # Observation space for all agents (position of each agent)
        self.observation_space = spaces.MultiDiscrete([grid_size[0] * grid_size[1]] * num_agents)

        # Define goal position first (all agents share the same goal in this case)
        self.goal_positions = [
        (self.grid_size[0]-1, 0),  # en bas à gauche
        (0, 0),                    # en haut à gauche
        (self.grid_size[0]-1, self.grid_size[1]-1),  # en bas à droite
        (0, self.grid_size[1]-1)   # en haut à droite
        ]

        # Define starting positions for agents
        self.agent_starts = self._generate_start_positions()

        # Generate holes only once during initialization
        self.holes = self._generate_holes()

        # Initialize agent positions and states
        self.agent_positions = None
        self.agent_done = None
        self.reached_goal = None

        # Complete initialization with reset
        self.reset()

    def _generate_start_positions(self):
        """Generate starting positions for all agents"""
        positions = []
        # For any additional agents, distribute them randomly
        cells = [(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
                if (i, j) not in positions and (i, j) not in self.goal_positions]

    # If we need more positions

        # Shuffle to get random positions
        self.random_gen.shuffle(cells)
        positions.extend(cells[:self.num_agents])

        return positions

    def _path_exists(self, start, goal, holes):
        """Vérifie s'il existe un chemin entre start et goal en évitant les trous"""
        queue = [start]
        visited = {start}

        while queue:
            x, y = queue.pop(0)

            if (x, y) == goal:
                return True

            # Vérifier les 4 directions
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Gauche, Bas, Droite, Haut
                nx, ny = x + dx, y + dy

                if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and
                    (nx, ny) not in holes and (nx, ny) not in visited):
                    queue.append((nx, ny))
                    visited.add((nx, ny))

        return False

    def _generate_holes(self):
        """Generate holes once and ensure paths exist for all agents"""
        holes = set()
        max_holes = int(self.grid_size[0] * self.grid_size[1] * self.hole_prob)

        # Exclude start positions and goal from hole candidates
        excluded_positions = set(self.agent_starts + self.goal_positions)
        hole_candidates = [(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
                          if (i, j) not in excluded_positions]

        # Shuffle candidates to avoid bias in hole selection
        self.random_gen.shuffle(hole_candidates)

        # Add holes randomly but check that a path exists after each addition
        for candidate in hole_candidates[:max_holes*2]:  # Try up to twice as many candidates
            if len(holes) >= max_holes:
                break

            # Temporarily add the hole
            holes.add(candidate)

            # Check if paths still exist for all agents to all the goals
            paths_exist = all(all(self._path_exists(start, goal, holes) for goal in self.goal_positions) for start in self.agent_starts)


            if not paths_exist:
                # If any path doesn't exist, remove the hole
                holes.remove(candidate)

        return holes

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            # Note: This only affects the gym random state, not our holes

        # Reset agent positions to starting positions
        self.agent_positions = list(self.agent_starts)

        # Reset agent done states
        self.agent_done = [False] * self.num_agents

        # Reset goal reaching states
        self.reached_goal = [False] * self.num_agents
        
        # Reset goals reached by agents
        self.goals_reached_by_agent = [set() for _ in range(self.num_agents)]
        
        # Reset goal rewards to initial values
        self.goal_rewards = {goal: 100 for goal in self.goal_positions}
        
        # Reset goal visit counts
        self.goal_visit_count = {goal: 0 for goal in self.goal_positions}

        return self.get_state(), {}

    def _check_collisions(self, positions):
        """
        Check for collisions between agents and return a list of agents involved in collisions.
        """
        collision_agents = set()
        # Create a dictionary where keys are positions and values are lists of agent indices
        position_to_agents = {}
        
        for i, pos in enumerate(positions):
            if not self.agent_done[i]:  # Only check active agents
                if pos in position_to_agents:
                    # Add all agents at this position to the collision set
                    for agent_idx in position_to_agents[pos]:
                        collision_agents.add(agent_idx)
                    collision_agents.add(i)
                    position_to_agents[pos].append(i)
                else:
                    position_to_agents[pos] = [i]
                    
        return list(collision_agents)

    def step(self, actions):
        # Ensure agent states are initialized
        if not hasattr(self, 'agent_done') or self.agent_done is None:
            self.agent_done = [False] * self.num_agents

        if not hasattr(self, 'reached_goal') or self.reached_goal is None:
            self.reached_goal = [False] * self.num_agents
            
        # Initialize a set to track which goals each agent has reached
        if not hasattr(self, 'goals_reached_by_agent') or self.goals_reached_by_agent is None:
            self.goals_reached_by_agent = [set() for _ in range(self.num_agents)]
            
        # Initialize a dictionary to track the reward value for each goal
        if not hasattr(self, 'goal_rewards') or self.goal_rewards is None:
            self.goal_rewards = {goal: 100 for goal in self.goal_positions}  # Initial reward of 100 for each goal
            
        # Initialize a counter for each goal to track order of arrival
        if not hasattr(self, 'goal_visit_count') or self.goal_visit_count is None:
            self.goal_visit_count = {goal: 0 for goal in self.goal_positions}

        # Calculate new positions for agents that aren't already done
        new_positions = []
        for i in range(self.num_agents):
            if not self.agent_done[i]:
                new_positions.append(self.move(self.agent_positions[i], actions[i]))
            else:
                new_positions.append(self.agent_positions[i])

        rewards = [0] * self.num_agents
        done = [False] * self.num_agents
        truncated = [False] * self.num_agents
        info = {"simultaneous_arrival": False, "collisions": False}

        # Track agents arriving at each goal in this step
        goals_arrivals = {goal: [] for goal in self.goal_positions}
        
        # First pass: check which agents are arriving at goals
        for i in range(self.num_agents):
            if not self.agent_done[i] and new_positions[i] in self.goal_positions:
                goals_arrivals[new_positions[i]].append(i)

        # Second pass: Distribute rewards based on arrivals and order
        for goal, arriving_agents in goals_arrivals.items():
            num_arrivals = len(arriving_agents)
            
            if num_arrivals > 0:
                # For simultaneous arrivals, all agents get the same current reward value
                if num_arrivals > 1:
                    # Trier les agents par ordre d'indice
                    sorted_agents = sorted(arriving_agents)
                    
                    # Attribuer les récompenses séquentiellement (chacun reçoit la moitié du précédent)
                    current_reward = self.goal_rewards[goal]
                    for idx, agent_idx in enumerate(sorted_agents):
                        # Le premier agent obtient la récompense complète
                        if idx == 0:
                            rewards[agent_idx] = current_reward
                        else:
                            # Les agents suivants obtiennent la moitié du précédent
                            current_reward /= 2
                            rewards[agent_idx] = current_reward
                        
                        # Ajouter l'objectif aux objectifs atteints par cet agent
                        self.goals_reached_by_agent[agent_idx].add(goal)
                        
                        # Marquer l'agent comme terminé
                        self.agent_done[agent_idx] = True
                        done[agent_idx] = True
                        self.reached_goal[agent_idx] = True
                    
                    # Mettre à jour la récompense de l'objectif pour le prochain agent
                    self.goal_rewards[goal] = current_reward / 2
                    self.goal_visit_count[goal] += num_arrivals
                    
                    info["simultaneous_arrival"] = True
                    info["num_simultaneous_arrivals"] = num_arrivals
                else:
                    # Single agent arrival
                    agent_idx = arriving_agents[0]
                    reward_per_agent = self.goal_rewards[goal]
                    
                    # Assign reward to the agent
                    rewards[agent_idx] = reward_per_agent
                    
                    # Add this goal to the set of goals reached by this agent
                    self.goals_reached_by_agent[agent_idx].add(goal)
                    
                    # Mark the agent as done
                    self.agent_done[agent_idx] = True
                    done[agent_idx] = True
                    self.reached_goal[agent_idx] = True
                    
                    # Update the goal reward (halved) for the next agent
                    self.goal_rewards[goal] /= 2
                    self.goal_visit_count[goal] += 1

        # Check for holes and update other agent states
        for i in range(self.num_agents):
            if not self.agent_done[i]:
                if new_positions[i] in self.holes:
                    rewards[i] = 0
                    self.agent_done[i] = True
                    done[i] = True
            else:
                # If agent was already done, keep the done flag set
                done[i] = True

        # Check for collisions among active agents
        collision_agents = self._check_collisions(new_positions)
        if collision_agents:
            # Apply collision penalty to agents involved in collisions
            for agent_idx in collision_agents:
                if not self.agent_done[agent_idx]:  # Only penalize active agents
                    rewards[agent_idx] -= self.collision_penalty
            
            info["collisions"] = True
            info["collision_agents"] = collision_agents

        # Update agent positions
        self.agent_positions = new_positions
        
        # Add information about how many unique goals each agent has reached
        info["goals_reached"] = [len(goals) for goals in self.goals_reached_by_agent]
        info["goal_rewards"] = self.goal_rewards.copy()
        info["goal_visit_count"] = self.goal_visit_count.copy()

        return self.get_state(), rewards, done, truncated, info


    def move(self, position, action):
        if self.np_random.rand() < self.slip_prob:
            if action == 0:  # LEFT
                side_actions = [1, 3]  # UP, DOWN
            elif action == 1:  # DOWN
                side_actions = [0, 2]  # LEFT, RIGHT
            elif action == 2:  # RIGHT
                side_actions = [1, 3]  # UP, DOWN
            elif action == 3:  # UP
                side_actions = [0, 2]  # LEFT, RIGHT

            # Choose randomly between the two side directions
            action = self.np_random.choice(side_actions)  # Slip to a random action

        x, y = position
        if action == 0 and y > 0:
            y -= 1  # Left
        elif action == 1 and x < self.grid_size[0] - 1:
            x += 1  # Down
        elif action == 2 and y < self.grid_size[1] - 1:
            y += 1  # Right
        elif action == 3 and x > 0:
            x -= 1  # Up

        return (x, y)

    def get_state(self):
        return tuple(pos[0] * self.grid_size[1] + pos[1] for pos in self.agent_positions)

    def render(self, mode="human"):
        grid = np.full(self.grid_size, ".")
        for hole in self.holes:
            grid[hole] = "H"
        for goal in self.goal_positions:
            grid[goal] = "G"

        # Render each agent position
        for i, pos in enumerate(self.agent_positions):
            # If multiple agents are on the same position, show a special character
            if any(j != i and self.agent_positions[j] == pos for j in range(self.num_agents)):
                # Collision detected, mark with a special character
                grid[pos] = "C"
            else:
                grid[pos] = f"{i+1}"

        print("\n".join(" ".join(row) for row in grid))
        print()
    def render_pygame(self, screen_size=400, save_path=None):
        '''
        Render the environment using pygame with sprites
        
        Parameters:
        -----------
        screen_size : int
            Size of the screen in pixels
        save_path : str, optional
            If provided, saves the rendered image to this path
        '''
        try:
            import pygame
        except ImportError:
            print("Pygame module not found. Install it using 'pip install pygame'")
            return
            
        # Initialize pygame if not already done
        if not hasattr(self, 'pygame_initialized'):
            pygame.init()
            self.pygame_initialized = True
            self.screen = pygame.display.set_mode((screen_size, screen_size))
            pygame.display.set_caption("Multi-Agent Frozen Lake")
            self.cell_size = screen_size // max(self.grid_size[0], self.grid_size[1])
            self.font = pygame.font.Font(None, self.cell_size // 2)
            
            # Load images
            img_dir = "../img/"
            self.images = {
                'F': pygame.image.load(img_dir + "ice.png"),
                'H': pygame.image.load(img_dir + "hole.png"),
                'G': pygame.image.load(img_dir + "ice.png"),  # Use ice as background for goal
                'S': pygame.image.load(img_dir + "stool.png")
            }
            
            # Scale images to cell size
            for key in self.images:
                self.images[key] = pygame.transform.scale(self.images[key], (self.cell_size, self.cell_size))
            
            # Load goal sprite separately to overlay on ice
            self.goal_sprite = pygame.image.load(img_dir + "goal.png")
            self.goal_sprite = pygame.transform.scale(self.goal_sprite, (self.cell_size, self.cell_size))
            
            # Load agent images for different directions
            self.agent_images = {
                'up': pygame.image.load(img_dir + "elf_up.png"),
                'down': pygame.image.load(img_dir + "elf_down.png"),
                'left': pygame.image.load(img_dir + "elf_left.png"),
                'right': pygame.image.load(img_dir + "elf_right.png")
            }
            
            # Scale agent images
            for key in self.agent_images:
                self.agent_images[key] = pygame.transform.scale(self.agent_images[key], (self.cell_size, self.cell_size))
                
            # Store last actions for each agent to determine which direction sprite to use
            self.last_actions = [1] * self.num_agents  # Default: facing down
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw grid
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, 
                                self.cell_size, self.cell_size)
                
                # Draw ice on all cells as base
                self.screen.blit(self.images['F'], rect)
                
                # Overlay with hole or goal as needed
                pos = (i, j)
                if pos in self.holes:
                    self.screen.blit(self.images['H'], rect)
                elif pos in self.goal_positions:
                    self.screen.blit(self.goal_sprite, rect)
                
                # Draw grid lines - plus visible now
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)  # Increased line thickness to 2
        
        # Check for collisions (agents on same position)
        position_counts = {}
        for i, pos in enumerate(self.agent_positions):
            if pos in position_counts:
                position_counts[pos].append(i)
            else:
                position_counts[pos] = [i]
        
        # Agent colors - use different colors for each agent label
        agent_label_colors = [
            (255, 0, 0),    # Red for agent 1
            (0, 0, 255),    # Blue for agent 2
            (255, 165, 0),  # Orange for agent 3
            (0, 128, 0)     # Green for agent 4
        ]
            
        # Draw agents
        for pos, agent_indices in position_counts.items():
            i, j = pos
            rect = pygame.Rect(j * self.cell_size, i * self.cell_size, 
                            self.cell_size, self.cell_size)
            
            # If collision (multiple agents at same position)
            if len(agent_indices) > 1:
                # Draw a larger purple circle for collision
                pygame.draw.circle(self.screen, (128, 0, 128), rect.center, self.cell_size // 2)
                
                # Create text showing which agents are colliding
                agent_nums = [str(idx+1) for idx in agent_indices]
                collision_text = "+".join(agent_nums)  # Example: "1+2" for agents 1 and 2
                
                # Draw collision text in bright red for visibility
                text = self.font.render(collision_text, True, (255, 0, 0))
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
            else:
                # Draw normal agent with direction
                agent_idx = agent_indices[0]
                
                # Determine which direction sprite to use based on last action
                direction_map = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
                
                # Use the last action if available, otherwise default to 'down'
                if hasattr(self, 'last_actions') and agent_idx < len(self.last_actions):
                    direction = direction_map[self.last_actions[agent_idx]]
                else:
                    direction = 'down'
                    
                self.screen.blit(self.agent_images[direction], rect)
                
                # Draw agent number with colored background for better visibility
                color = agent_label_colors[min(agent_idx, len(agent_label_colors)-1)]
                
                # Create a small colored circle as background for the text
                circle_radius = self.cell_size // 6
                pygame.draw.circle(self.screen, color, 
                                (rect.left + circle_radius + 2, rect.top + circle_radius+2), 
                                circle_radius)
                
                # Draw agent number in white on colored background
                text = self.font.render(f"{agent_idx+1}", True, (255, 255, 255))
                text_rect = text.get_rect(center=(rect.left + circle_radius + 2, rect.top + circle_radius+2))
                self.screen.blit(text, text_rect)
        
        # Save the image if a path is provided
        if save_path is not None:
            pygame.image.save(self.screen, save_path)
        
        # Update display
        pygame.display.flip()
        
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.iconify()

    def close(self):
        pass

