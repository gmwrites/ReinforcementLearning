import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from battleship_env import BattleshipEnv, setup_results_directory


class EnhancedQLearningAgent:
    """
    Enhanced Q-Learning agent for the Battleship game with improved state representation
    and exploration strategy.
    """

    def __init__(self, grid_size=5, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, exploration_decay=0.9999, min_exploration_rate=0.1):
        self.grid_size = grid_size
        self.name = "EnhancedQLearningAgent"
        self.action_size = grid_size * grid_size

        # Q-learning hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Initialize state
        self.reset()

    def reset(self):
        """Reset the agent's state for a new game."""
        self.available_actions = set(range(self.grid_size * self.grid_size))
        self.q_table = {}  # Dictionary to store state-action values
        self.hit_cells = []  # Track positions of hits
        self.miss_cells = []  # Track positions of misses
        self.last_hit = None  # Last cell where a hit was found

    def _state_to_key(self, observation):
        """
        Convert the observation to a more compact and meaningful state representation.

        Args:
            observation: The current state observation (player's view of the grid)

        Returns:
            A hashable key representing the state
        """
        # Count known information
        unknowns = np.sum(observation == 0)
        misses = np.sum(observation == 1)
        hits = np.sum(observation == 2)

        # Get patterns of hits (connected hits are likely from the same ship)
        hit_patterns = []

        # Track all hit positions
        hit_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if observation[i, j] == 2:
                    hit_positions.append((i, j))

        # Find connected components (ships) among hits
        while hit_positions:
            # Start with one hit position
            position_queue = [hit_positions.pop(0)]
            current_pattern = []

            while position_queue:
                i, j = position_queue.pop(0)
                current_pattern.append((i, j))

                # Check adjacent cells
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if (ni, nj) in hit_positions:
                        hit_positions.remove((ni, nj))
                        position_queue.append((ni, nj))

            # Add the length of this pattern
            hit_patterns.append(len(current_pattern))

        # Create a state key that captures important aspects of the game state
        # The number of patterns and their sizes help identify different ships
        # Sort the patterns to ensure consistent state representation
        state_key = (unknowns, misses, hits, tuple(sorted(hit_patterns)))
        return state_key

    def _get_q_value(self, state_key, action):
        """
        Get Q-value for a state-action pair.

        Args:
            state_key: Hashable state key
            action: Action index

        Returns:
            Q-value for the state-action pair
        """
        if (state_key, action) not in self.q_table:
            # Initialize new state-action pairs with small random values
            # to break ties and encourage exploration of unvisited states
            self.q_table[(state_key, action)] = np.random.uniform(0, 0.1)

        return self.q_table[(state_key, action)]

    def _set_q_value(self, state_key, action, value):
        """
        Set Q-value for a state-action pair.

        Args:
            state_key: Hashable state key
            action: Action index
            value: New Q-value
        """
        self.q_table[(state_key, action)] = value

    def _calculate_hit_proximity(self, observation):
        """
        Calculate proximity of each cell to known hits.
        This helps the agent prioritize cells adjacent to hits.

        Args:
            observation: Current grid observation

        Returns:
            Array of proximity values for each cell
        """
        hit_proximity = np.zeros(self.grid_size * self.grid_size)

        # Find all hit cells
        hit_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if observation[i, j] == 2:
                    hit_cells.append((i, j))

        # No hits yet, return zeros
        if not hit_cells:
            return hit_proximity

        # Calculate proximity for each valid action
        for action in self.available_actions:
            r, c = divmod(action, self.grid_size)

            # Check if adjacent to any hit
            for hit_i, hit_j in hit_cells:
                # Adjacent horizontally or vertically
                if (abs(r - hit_i) == 1 and c == hit_j) or (r == hit_i and abs(c - hit_j) == 1):
                    hit_proximity[action] = 1.0
                    break

        return hit_proximity

    def act(self, observation):
        """
        Select an action based on the current state using an enhanced epsilon-greedy policy
        with priority for cells adjacent to hits.

        Args:
            observation: The current state observation (player's view of the grid)

        Returns:
            Action to take (cell to fire at)
        """
        state_key = self._state_to_key(observation)

        # Filter actions to only include available ones
        valid_actions = list(self.available_actions)

        if not valid_actions:
            # If no valid actions remain, return a random cell
            # (should never happen in a normal game)
            return np.random.randint(0, self.grid_size * self.grid_size)

        # Calculate proximity to hits to guide exploration
        hit_proximity = self._calculate_hit_proximity(observation)

        # Make decision: explore or exploit
        if np.random.rand() < self.exploration_rate:
            # Exploration: prefer cells adjacent to hits when available
            if np.any(hit_proximity > 0):
                # Get actions adjacent to hits
                priority_actions = [a for a in valid_actions if hit_proximity[a] > 0]
                if priority_actions:
                    # Choose randomly among priority actions
                    return np.random.choice(priority_actions)

            # Otherwise, choose randomly from all valid actions
            return np.random.choice(valid_actions)
        else:
            # Exploitation: choose action with highest Q-value
            q_values = [self._get_q_value(state_key, a) for a in valid_actions]

            # Apply hit proximity as a bonus to Q-values for cells adjacent to hits
            # This helps break ties in favor of cells more likely to contain ships
            adjusted_q_values = [q + 0.01 * hit_proximity[a] for q, a in zip(q_values, valid_actions)]

            # Find actions with maximum adjusted Q-value
            max_indices = np.where(adjusted_q_values == np.max(adjusted_q_values))[0]

            # Select a random action among those with maximum value
            chosen_index = np.random.choice(max_indices)
            action = valid_actions[chosen_index]

        # Remove the chosen action from available actions
        self.available_actions.remove(action)

        return action

    def update(self, observation, action, reward, next_observation, done, info):
        """
        Update the Q-table based on the outcome of the action.

        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
            done: Whether the episode is done
            info: Additional information
        """
        # Update our understanding of the grid
        row, col = divmod(action, self.grid_size)

        # Track hits and misses for future reference
        if next_observation[row, col] == 2:  # Hit
            self.hit_cells.append((row, col))
            self.last_hit = (row, col)
        elif next_observation[row, col] == 1:  # Miss
            self.miss_cells.append((row, col))

        # Get current and next state representations
        state_key = self._state_to_key(observation)
        next_state_key = self._state_to_key(next_observation)

        # Get current Q value
        current_q = self._get_q_value(state_key, action)

        # Calculate new Q value using Q-learning update rule
        if done:
            # If game is done, no future reward
            # Add a bonus for finishing the game
            new_q = current_q + self.learning_rate * (reward + 5 - current_q)
        else:
            # Get maximum Q value for next state
            valid_actions = list(self.available_actions)
            if valid_actions:
                next_q_values = [self._get_q_value(next_state_key, a) for a in valid_actions]
                max_next_q = np.max(next_q_values) if next_q_values else 0
            else:
                max_next_q = 0

            # Calculate updated Q value
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

        # Update Q table
        self._set_q_value(state_key, action, new_q)

        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)

    def save(self, filepath):
        """Save the agent and its Q-table."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'grid_size': self.grid_size,
                'q_table': self.q_table,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'exploration_decay': self.exploration_decay,
                'min_exploration_rate': self.min_exploration_rate
            }, f)

    def load(self, filepath):
        """Load the agent from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.name = data['name']
            self.grid_size = data['grid_size']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.exploration_rate = data['exploration_rate']
            self.exploration_decay = data['exploration_decay']
            self.min_exploration_rate = data['min_exploration_rate']

            # Reset and restore Q-table
            self.reset()
            self.q_table = data['q_table']


def train_enhanced_qlearning_agent(episodes=10000, grid_size=5, ships_config=None,
                                   learning_rate=0.1, discount_factor=0.9,
                                   exploration_rate=1.0, exploration_decay=0.9999,
                                   min_exploration_rate=0.1, save_interval=1000):
    """
    Train the enhanced Q-learning agent.

    Args:
        episodes: Number of games to play
        grid_size: Size of the game grid
        ships_config: Configuration of ships [ship1_size, ship2_size, ...]
        learning_rate: Learning rate for Q-learning
        discount_factor: Discount factor for future rewards
        exploration_rate: Initial exploration rate
        exploration_decay: Rate at which exploration decays
        min_exploration_rate: Minimum exploration rate
        save_interval: How often to save checkpoints

    Returns:
        agent: The trained agent
        results: Dictionary containing performance metrics
    """
    if ships_config is None:
        ships_config = [3, 2]  # Default ship configuration for 5x5 grid

    env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)
    agent = EnhancedQLearningAgent(grid_size=grid_size, learning_rate=learning_rate,
                                   discount_factor=discount_factor,
                                   exploration_rate=exploration_rate,
                                   exploration_decay=exploration_decay,
                                   min_exploration_rate=min_exploration_rate)

    # Set up results directory
    results_dir = setup_results_directory(agent.name, "training")

    # Create a subdirectory for checkpoints
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Initialize metrics tracking
    shots_history = []
    rewards_history = []
    win_rate = 0

    # Track moving averages
    window_size = 100
    moving_avg_shots = []
    moving_avg_rewards = []

    start_time = time.time()

    for episode in range(episodes):
        observation = env.reset()
        agent.reset()
        done = False
        episode_shots = 0
        episode_reward = 0

        while not done:
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)

            # Apply enhanced rewards
            if info['message'] == 'Hit':
                if 'ships_sunk' in info and info['ships_sunk'] > 0:
                    reward = 5  # Bonus for sinking a ship
                else:
                    reward = 2  # Standard hit
            else:  # Miss
                reward = -0.1  # Small penalty for missing

            agent.update(observation, action, reward, next_observation, done, info)

            observation = next_observation
            episode_shots += 1
            episode_reward += reward

        # Record metrics
        shots_history.append(episode_shots)
        rewards_history.append(episode_reward)
        if env.hits == env.total_ship_cells:
            win_rate += 1

        # Calculate moving averages
        if episode >= window_size:
            moving_avg_shots.append(np.mean(shots_history[-window_size:]))
            moving_avg_rewards.append(np.mean(rewards_history[-window_size:]))
        else:
            moving_avg_shots.append(np.mean(shots_history))
            moving_avg_rewards.append(np.mean(rewards_history))

        # Print progress and metrics
        if (episode + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed_time if elapsed_time > 0 else 0
            time_remaining = (episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0

            print(f"Episode {episode + 1}/{episodes} | " +
                  f"Exploration rate: {agent.exploration_rate:.4f} | " +
                  f"Recent avg shots: {moving_avg_shots[-1]:.2f} | " +
                  f"Recent avg reward: {moving_avg_rewards[-1]:.2f} | " +
                  f"Time elapsed: {elapsed_time:.1f}s | " +
                  f"Est. time remaining: {time_remaining:.1f}s")

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{episode + 1}.pkl")
            agent.save(checkpoint_path)

            # Create interim plots
            if (episode + 1) % (save_interval * 5) == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, episode + 2), moving_avg_shots)
                plt.title(f'Moving Average Shots (Window: {window_size})')
                plt.xlabel('Episode')
                plt.ylabel('Average Shots')
                plt.grid(True)
                plt.savefig(os.path.join(results_dir, f'interim_shots_{episode + 1}.png'))
                plt.close()

                plt.figure(figsize=(10, 6))
                plt.plot(range(1, episode + 2), moving_avg_rewards)
                plt.title(f'Moving Average Rewards (Window: {window_size})')
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.grid(True)
                plt.savefig(os.path.join(results_dir, f'interim_rewards_{episode + 1}.png'))
                plt.close()

    # Calculate final metrics
    win_rate = win_rate / episodes
    avg_shots = np.mean(shots_history)
    avg_reward = np.mean(rewards_history)

    # Save results
    results = {
        'episodes': episodes,
        'win_rate': win_rate,
        'avg_shots': avg_shots,
        'avg_reward': avg_reward,
        'shots_history': shots_history,
        'rewards_history': rewards_history,
        'moving_avg_shots': moving_avg_shots,
        'moving_avg_rewards': moving_avg_rewards
    }

    # Save agent
    agent_path = os.path.join(results_dir, 'enhanced_qlearning_agent.pkl')
    agent.save(agent_path)

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, episodes + 1),
        'shots': shots_history,
        'reward': rewards_history,
        'moving_avg_shots': moving_avg_shots,
        'moving_avg_rewards': moving_avg_rewards
    })
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)

    # Plot final learning curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot shots curve
    ax1.plot(range(1, episodes + 1), shots_history, alpha=0.3, color='blue')
    ax1.plot(range(1, episodes + 1), moving_avg_shots, linewidth=2, color='blue')
    ax1.set_title(f'Enhanced Q-Learning Agent Shots Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Shots to Complete Game')
    ax1.grid(True)
    ax1.legend(['Raw data', f'Moving avg (window={window_size})'])

    # Plot rewards curve
    ax2.plot(range(1, episodes + 1), rewards_history, alpha=0.3, color='green')
    ax2.plot(range(1, episodes + 1), moving_avg_rewards, linewidth=2, color='green')
    ax2.set_title(f'Enhanced Q-Learning Agent Rewards\nAvg. Reward: {avg_reward:.2f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Episode Reward')
    ax2.grid(True)
    ax2.legend(['Raw data', f'Moving avg (window={window_size})'])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance.png'))
    plt.close()

    # Plot exploration rate decay
    episodes_arr = np.arange(episodes)
    exploration_rates = exploration_rate * (exploration_decay ** episodes_arr)
    exploration_rates = np.maximum(exploration_rates, min_exploration_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), exploration_rates)
    plt.title('Exploration Rate Decay')
    plt.xlabel('Episode')
    plt.ylabel('Exploration Rate')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'exploration_decay.png'))
    plt.close()

    print(f"\nTraining completed for Enhanced Q-Learning Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Results saved to: {results_dir}")

    return agent, results


def evaluate_enhanced_qlearning_agent(agent, test_episodes=50, grid_size=5, ships_config=None, fixed_test=True):
    """
    Evaluate the enhanced Q-learning agent on test episodes.

    Args:
        agent: The agent to evaluate
        test_episodes: Number of test episodes
        grid_size: Size of the game grid
        ships_config: Configuration of ships
        fixed_test: Whether to use fixed ship positions for testing

    Returns:
        results: Dictionary containing performance metrics
    """
    import time
    from battleship_env import create_test_scenario

    if ships_config is None:
        ships_config = [3, 2]  # Default ship configuration for 5x5 grid

    # Set up results directory
    results_dir = setup_results_directory(agent.name, "testing")

    # Initialize metrics tracking
    shots_history = []
    rewards_history = []
    win_rate = 0

    # Set exploration rate to minimum for evaluation
    original_exploration_rate = agent.exploration_rate
    agent.exploration_rate = agent.min_exploration_rate

    start_time = time.time()

    for episode in range(test_episodes):
        if fixed_test:
            # Use the standard test scenario
            env = create_test_scenario(grid_size=grid_size, ships_config=ships_config)
        else:
            # Use random ship placements
            env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)

        agent.reset()
        observation = env.reset()  # This is redundant for fixed_test but kept for consistency
        done = False
        episode_shots = 0
        episode_reward = 0

        while not done:
            action = agent.act(observation)

            # Apply enhanced rewards
            next_observation, reward, done, info = env.step(action)
            if info['message'] == 'Hit':
                if 'ships_sunk' in info and info['ships_sunk'] > 0:
                    reward = 5  # Bonus for sinking a ship
                else:
                    reward = 2  # Standard hit
            else:  # Miss
                reward = -0.1  # Small penalty for missing

            observation = next_observation
            episode_shots += 1
            episode_reward += reward

            # Prevent infinite loops
            if episode_shots > grid_size * grid_size:
                break

        # Record metrics
        shots_history.append(episode_shots)
        rewards_history.append(episode_reward)
        if env.hits == env.total_ship_cells:
            win_rate += 1

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Evaluated {episode + 1}/{test_episodes} episodes | " +
                  f"Avg shots: {np.mean(shots_history[-10:]):.2f}")

    # Restore original exploration rate
    agent.exploration_rate = original_exploration_rate

    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.1f}s")

    # Calculate final metrics
    win_rate = win_rate / test_episodes
    avg_shots = np.mean(shots_history)
    avg_reward = np.mean(rewards_history)

    # Save results
    results = {
        'test_episodes': test_episodes,
        'win_rate': win_rate,
        'avg_shots': avg_shots,
        'avg_reward': avg_reward,
        'shots_history': shots_history,
        'rewards_history': rewards_history,
        'fixed_test': fixed_test
    }

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, test_episodes + 1),
        'shots': shots_history,
        'reward': rewards_history
    })
    metrics_df.to_csv(os.path.join(results_dir, 'test_metrics.csv'), index=False)

    # Plot histogram of shots distribution
    plt.figure(figsize=(10, 6))
    plt.hist(shots_history, bins=range(min(shots_history), max(shots_history) + 2), alpha=0.7)
    plt.title(f'Enhanced Q-Learning Agent Test Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    plt.xlabel('Shots to Complete Game')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Add mean line
    plt.axvline(avg_shots, color='r', linestyle='dashed', linewidth=2)
    plt.text(avg_shots + 0.5, plt.ylim()[1] * 0.9, f'Mean: {avg_shots:.2f}', color='r')

    plt.savefig(os.path.join(results_dir, 'test_histogram.png'))
    plt.close()

    print(f"\nEvaluation completed for Enhanced Q-Learning Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Results saved to: {results_dir}")

    return results


def progressive_training():
    """
    Train the agent progressively on increasingly complex environments.
    This helps the agent build up knowledge incrementally.
    """
    import time
    import os

    results_dir = setup_results_directory("EnhancedQLearningAgent", "progressive_training")

    # Start with a small grid and a single ship
    print("\n=== Stage 1: Training on 3x3 grid with a single ship ===")
    agent, _ = train_enhanced_qlearning_agent(
        episodes=5000,
        grid_size=3,
        ships_config=[2],
        exploration_decay=0.9998,
        save_interval=1000
    )

    # Save stage 1 agent
    stage1_path = os.path.join(results_dir, 'stage1_agent.pkl')
    agent.save(stage1_path)

    # Medium grid with two small ships
    print("\n=== Stage 2: Training on 4x4 grid with two ships ===")
    agent, _ = train_enhanced_qlearning_agent(
        episodes=10000,
        grid_size=4,
        ships_config=[2, 2],
        exploration_rate=0.8,  # Start with lower exploration
        exploration_decay=0.9999,
        save_interval=2000
    )

    # Save stage 2 agent
    stage2_path = os.path.join(results_dir, 'stage2_agent.pkl')
    agent.save(stage2_path)

    # Full-sized grid with standard configuration
    print("\n=== Stage 3: Training on 5x5 grid with standard configuration ===")
    agent, _ = train_enhanced_qlearning_agent(
        episodes=20000,
        grid_size=5,
        ships_config=[3, 2],
        exploration_rate=0.5,  # Further reduce initial exploration
        exploration_decay=0.9999,
        save_interval=4000
    )

    # Save final agent
    final_path = os.path.join(results_dir, 'final_agent.pkl')
    agent.save(final_path)

    # Evaluate the final agent
    print("\n=== Evaluating final agent ===")
    results = evaluate_enhanced_qlearning_agent(agent, test_episodes=100)

    return agent, results


import time  # Add this at the top of the file

if __name__ == "__main__":
    # Choose one of the training approaches

    # Option 1: Standard training
    agent, results = train_enhanced_qlearning_agent(episodes=20000)
    test_results = evaluate_enhanced_qlearning_agent(agent, test_episodes=100)

    # Option 2: Progressive training (recommended for better results)
    # agent, results = progressive_training()