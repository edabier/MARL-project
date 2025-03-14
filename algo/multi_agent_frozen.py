#@title 2 agent env
import gymnasium as gym
import numpy as np 
import random
from gymnasium import spaces

#@title multi-n_agent_env

class FrozenLakeFlexibleAgentsEnv(gym.Env):
    """
    A multi-agent FrozenLake environment where multiple agents navigate a frozen grid.
    The agents receive a bonus reward if they reach the goal at the same time.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_agents=2, grid_size=(5, 5), slip_prob=0.2, hole_prob=0.2, seed=None, collaboration_bonus=1.0):
        super(FrozenLakeFlexibleAgentsEnv, self).__init__()
        self.grid_size = grid_size
        self.slip_prob = slip_prob  # Probability of slipping
        self.hole_prob = hole_prob  # Probability of each cell being a hole
        self.collaboration_bonus = collaboration_bonus  # Bonus reward for simultaneous goal arrival
        self.num_agents = num_agents

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
        self.goal_pos = (self.grid_size[0]-1, 0)

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

        # First agent starts at top-left
        positions.append((0, 0))

        # Last agent starts at bottom-right (if there are at least 2 agents)
        if self.num_agents > 1:
            positions.append((self.grid_size[0]-1, self.grid_size[1]-1))

        # For any additional agents, distribute them randomly
        cells = [(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
                if (i, j) not in positions and (i, j) != self.goal_pos]

        # If we need more positions
        if self.num_agents > 2:
            # Shuffle to get random positions
            self.random_gen.shuffle(cells)
            positions.extend(cells[:self.num_agents - 2])

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
        excluded_positions = set(self.agent_starts + [self.goal_pos])
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

            # Check if paths still exist for all agents
            paths_exist = all(self._path_exists(start, self.goal_pos, holes) for start in self.agent_starts)

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

        return self.get_state(), {}

    def step(self, actions):
        # Ensure agent states are initialized
        if not hasattr(self, 'agent_done') or self.agent_done is None:
            self.agent_done = [False] * self.num_agents

        if not hasattr(self, 'reached_goal') or self.reached_goal is None:
            self.reached_goal = [False] * self.num_agents

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
        info = {"simultaneous_arrival": False}

        # Temporary flags to detect simultaneous goal arrival in this step
        just_reached_goal = [False] * self.num_agents

        # Check status for each agent
        for i in range(self.num_agents):
            if not self.agent_done[i]:
                if new_positions[i] in self.holes:
                    rewards[i] = 0
                    self.agent_done[i] = True
                    done[i] = True
                elif new_positions[i] == self.goal_pos:
                    rewards[i] = 1
                    self.agent_done[i] = True
                    done[i] = True
                    self.reached_goal[i] = True
                    just_reached_goal[i] = True
            else:
                # If agent was already done, keep the done flag set
                done[i] = True

        # Check if multiple agents reached the goal in this step
        simultaneous_arrivals = sum(just_reached_goal)
        if simultaneous_arrivals >= 2:
            # Calculate progressive bonus based on number of simultaneous arrivals
            # More agents arriving together = higher bonus per agent
            progressive_bonus = self.collaboration_bonus * (simultaneous_arrivals - 1)

            # Apply the progressive collaboration bonus to all agents who arrived simultaneously
            for i in range(self.num_agents):
                if just_reached_goal[i]:
                    rewards[i] += progressive_bonus

            info["simultaneous_arrival"] = True
            info["num_simultaneous_arrivals"] = simultaneous_arrivals
            info["progressive_bonus"] = progressive_bonus

        # Update agent positions
        self.agent_positions = new_positions

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
        grid[self.goal_pos] = "G"

        # Render each agent position
        for i, pos in enumerate(self.agent_positions):
            # If multiple agents are on the same position, just show the most recent one
            grid[pos] = f"A{i+1}"

        print("\n".join(" ".join(row) for row in grid))
        print()

    def close(self):
        pass