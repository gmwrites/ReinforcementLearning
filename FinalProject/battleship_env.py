import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class BattleshipEnv(gym.Env):
    """
    Custom Battleship Environment that follows gym interface.
    This environment simulates a game of Battleship where the agent tries to sink all ships.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_size=5, ships_config=None):
        super(BattleshipEnv, self).__init__()

        # Default ship configuration for 5x5 grid if not specified
        if ships_config is None:
            self.ships_config = [3, 2]  # One ship of size 3 and one ship of size 2
        else:
            self.ships_config = ships_config

        self.grid_size = grid_size

        # Check if ship configuration is valid for the grid size
        self._validate_ship_config()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(grid_size * grid_size)
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(grid_size, grid_size),
                                            dtype=np.int32)

        # Initialize game state
        self.reset()

    def _validate_ship_config(self):
        """Validate if the ship configuration fits on the grid."""
        # Calculate total ship cells
        total_ship_cells = sum(self.ships_config)

        # Calculate available cells
        total_cells = self.grid_size * self.grid_size

        if total_ship_cells > total_cells:
            raise ValueError(
                f"Ship configuration {self.ships_config} is too large for a {self.grid_size}x{self.grid_size} grid.")

        # More advanced check: Can each ship fit on the grid?
        max_ship_size = max(self.ships_config)
        if max_ship_size > self.grid_size:
            raise ValueError(f"Ship of size {max_ship_size} cannot fit on a {self.grid_size}x{self.grid_size} grid.")

        return True

    def reset(self):
        """Reset the environment to start a new game."""
        # Initialize empty grid
        # 0: unknown, 1: miss, 2: hit, 3: ship (hidden from observation)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place ships randomly
        self._place_ships()

        # Initialize game variables
        self.shots_fired = 0
        self.hits = 0
        self.ships_sunk = 0
        self.total_ship_cells = sum(self.ships_config)
        self.game_over = False
        self.action_history = []

        # Return initial observation (player's view of the grid)
        return self._get_observation()

    def _place_ships(self):
        """Place ships randomly on the grid."""
        # Make a separate grid for ship placement (hidden from observation)
        self.ship_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        for ship_size in self.ships_config:
            placed = False
            attempts = 0
            max_attempts = 100  # Prevent infinite loop

            while not placed and attempts < max_attempts:
                # Randomly choose orientation: 0 for horizontal, 1 for vertical
                orientation = np.random.randint(0, 2)

                if orientation == 0:  # Horizontal
                    # Choose starting point ensuring ship fits within grid
                    x = np.random.randint(0, self.grid_size)
                    y = np.random.randint(0, self.grid_size - ship_size + 1)

                    # Check if position is valid (no overlap with other ships)
                    valid = True
                    for i in range(ship_size):
                        if self.ship_grid[x, y + i] != 0:
                            valid = False
                            break

                    # Place ship if valid
                    if valid:
                        for i in range(ship_size):
                            self.ship_grid[x, y + i] = ship_size  # Mark with ship size for identification
                        placed = True

                else:  # Vertical
                    # Choose starting point ensuring ship fits within grid
                    x = np.random.randint(0, self.grid_size - ship_size + 1)
                    y = np.random.randint(0, self.grid_size)

                    # Check if position is valid (no overlap with other ships)
                    valid = True
                    for i in range(ship_size):
                        if self.ship_grid[x + i, y] != 0:
                            valid = False
                            break

                    # Place ship if valid
                    if valid:
                        for i in range(ship_size):
                            self.ship_grid[x + i, y] = ship_size  # Mark with ship size for identification
                        placed = True

                attempts += 1

            if not placed:
                # If we can't place a ship after max attempts, try a different configuration
                self.ship_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
                self._place_ships()  # Recursively try again
                return

    def step(self, action):
        """
        Execute one step in the environment.
        Args:
            action: An integer representing the cell to fire at (row*grid_size + col)
        Returns:
            observation: The player's view of the grid
            reward: The reward for the action
            done: Whether the game is over
            info: Additional information
        """
        if self.game_over:
            return self._get_observation(), 0, True, {"message": "Game already over"}

        # Convert action to grid coordinates
        x, y = divmod(action, self.grid_size)

        # Check if this cell has already been fired at
        if self.grid[x, y] != 0:
            return self._get_observation(), -1, False, {"message": "Cell already fired at"}

        self.action_history.append(action)
        self.shots_fired += 1

        # Check if hit
        if self.ship_grid[x, y] != 0:
            self.grid[x, y] = 2  # Mark as hit
            self.hits += 1

            # Check if ship is sunk
            ship_size = self.ship_grid[x, y]
            ship_id = self.ship_grid[x, y]

            # Count hits on this specific ship
            ship_hits = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.ship_grid[i, j] == ship_id and self.grid[i, j] == 2:
                        ship_hits += 1

            # If all cells of this ship are hit, it's sunk
            if ship_hits == ship_size:
                self.ships_sunk += 1
                reward = 2  # Bonus for sinking a ship
            else:
                reward = 1  # Regular hit
        else:
            self.grid[x, y] = 1  # Mark as miss
            reward = 0  # Miss

        # Check if game is over (all ships sunk)
        if self.hits == self.total_ship_cells:
            self.game_over = True
            reward += 5  # Bonus for winning

        return self._get_observation(), reward, self.game_over, {
            "shots_fired": self.shots_fired,
            "hits": self.hits,
            "ships_sunk": self.ships_sunk,
            "message": "Hit" if self.grid[x, y] == 2 else "Miss"
        }

    def _get_observation(self):
        """Return the player's view of the grid (without revealing ship locations)."""
        # Return only information visible to the player (0: unknown, 1: miss, 2: hit)
        return self.grid.copy()



    def render(self, mode='human'):
        """Render the game state."""
        if mode == 'human':
            # Create a new figure
            plt.figure(figsize=(10, 5))

            # Plot player's view
            plt.subplot(1, 2, 1)

            # Use pcolormesh for proper grid alignment
            x = np.arange(0, self.grid_size + 1)
            y = np.arange(0, self.grid_size + 1)
            X, Y = np.meshgrid(x, y)

            # Create a masked array for proper coloring
            cmap = plt.cm.Blues
            masked_grid = np.ma.masked_array(self.grid, self.grid < 0)

            plt.pcolormesh(X, Y, masked_grid, cmap=cmap, vmin=0, vmax=2, edgecolors='k', linewidth=1)
            plt.title("Player's View")
            plt.axis('equal')
            plt.xticks(np.arange(0.5, self.grid_size + 0.5), np.arange(self.grid_size))
            plt.yticks(np.arange(0.5, self.grid_size + 0.5), np.arange(self.grid_size))

            # Add labels
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.grid[i, j] == 1:
                        plt.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', color='black')
                    elif self.grid[i, j] == 2:
                        plt.text(j + 0.5, i + 0.5, 'H', ha='center', va='center', color='red', weight='bold')

            # Set limits to show the entire grid properly
            plt.xlim(0, self.grid_size)
            plt.ylim(self.grid_size, 0)  # Invert y-axis for traditional grid view

            # Plot true ship positions
            plt.subplot(1, 2, 2)

            # Create a binary ship grid for visualization
            ship_view = (self.ship_grid > 0).astype(float)

            plt.pcolormesh(X, Y, ship_view, cmap='YlOrRd', vmin=0, vmax=1, edgecolors='k', linewidth=1)
            plt.title("Ship Positions")
            plt.axis('equal')
            plt.xticks(np.arange(0.5, self.grid_size + 0.5), np.arange(self.grid_size))
            plt.yticks(np.arange(0.5, self.grid_size + 0.5), np.arange(self.grid_size))

            # Add labels for hits and misses on the ship grid
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.grid[i, j] == 1:
                        plt.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', color='blue')
                    elif self.grid[i, j] == 2:
                        plt.text(j + 0.5, i + 0.5, 'H', ha='center', va='center', color='red', weight='bold')

            # Set limits to show the entire grid properly
            plt.xlim(0, self.grid_size)
            plt.ylim(self.grid_size, 0)  # Invert y-axis for traditional grid view

            plt.tight_layout()
            plt.show()

        elif mode == 'rgb_array':
            # For machine learning, return an RGB array representation
            # (not implemented in this basic version)
            return np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)

    def close(self):
        """Clean up resources."""
        plt.close()

    def get_ship_grid(self):
        """Return the ship grid (for evaluation/debugging)."""
        return self.ship_grid.copy()

    def get_game_state(self):
        """Return the current game state as a dictionary."""
        return {
            "grid": self.grid.copy(),
            "ship_grid": self.ship_grid.copy(),
            "shots_fired": self.shots_fired,
            "hits": self.hits,
            "ships_sunk": self.ships_sunk,
            "game_over": self.game_over,
            "action_history": self.action_history.copy()
        }


def create_test_scenario(grid_size=5, ships_config=None):
    """
    Create a test scenario with fixed ship positions.
    Returns an environment with predetermined ship positions.
    """
    # Create standard environment
    env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)

    # Override random ship placement with fixed positions
    env.ship_grid = np.zeros((grid_size, grid_size), dtype=np.int32)

    if grid_size == 5 and ships_config == [3, 2]:
        # Place ship of size 3 horizontally at (1,0), (1,1), (1,2)
        env.ship_grid[1, 0] = 3
        env.ship_grid[1, 1] = 3
        env.ship_grid[1, 2] = 3

        # Place ship of size 2 vertically at (2,4), (3,4)
        env.ship_grid[2, 4] = 2
        env.ship_grid[3, 4] = 2
    else:
        # For other configurations, create a deterministic but reasonable layout
        ship_idx = 0
        for ship_size in ships_config:
            # Alternate between horizontal and vertical placement
            if ship_idx % 2 == 0:  # Horizontal
                row = (ship_idx // 2) % grid_size
                col = 0
                # Ensure it fits
                if col + ship_size > grid_size:
                    col = grid_size - ship_size
                for i in range(ship_size):
                    env.ship_grid[row, col + i] = ship_size
            else:  # Vertical
                row = 0
                col = (ship_idx // 2) % grid_size
                # Ensure it fits
                if row + ship_size > grid_size:
                    row = grid_size - ship_size
                for i in range(ship_size):
                    env.ship_grid[row + i, col] = ship_size
            ship_idx += 1

    # Reset other environment variables to be consistent with the ship placement
    env.total_ship_cells = sum(ships_config)
    env.hits = 0
    env.ships_sunk = 0
    env.shots_fired = 0
    env.game_over = False
    env.action_history = []

    return env


def setup_results_directory(model_name, test_type="training"):
    """
    Set up a directory for saving results with a timestamp.
    Returns the path to the directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "battleship_results"
    test_dir = os.path.join(base_dir, test_type)
    results_dir = os.path.join(test_dir, f"{model_name}_{timestamp}")

    # Create directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    return results_dir