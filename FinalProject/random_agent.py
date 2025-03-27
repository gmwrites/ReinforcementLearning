import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from battleship_env import BattleshipEnv, setup_results_directory


class RandomAgent:
    """
    Agent that makes random guesses in the Battleship game.
    """

    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.name = "RandomAgent"
        self.reset()

    def reset(self):
        """Reset the agent's state for a new game."""
        self.available_actions = list(range(self.grid_size * self.grid_size))

    def act(self, observation=None):
        """
        Select a random action from available actions.

        Args:
            observation: The current state observation (not used by random agent)

        Returns:
            Action to take (cell to fire at)
        """
        if not self.available_actions:
            # If somehow we ran out of actions, return a random valid action
            return np.random.randint(0, self.grid_size * self.grid_size)

        # Choose a random action from remaining available actions
        action_idx = np.random.randint(0, len(self.available_actions))
        action = self.available_actions.pop(action_idx)

        return action

    def update(self, observation, action, reward, next_observation, done, info):
        """
        Update the agent's knowledge based on the action's outcome.
        (Not used for random agent as it doesn't learn)
        """
        pass

    def save(self, filepath):
        """Save the agent (not much to save for random agent)."""
        with open(filepath, 'wb') as f:
            pickle.dump({'name': self.name, 'grid_size': self.grid_size}, f)

    def load(self, filepath):
        """Load the agent from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.name = data['name']
            self.grid_size = data['grid_size']
            self.reset()


def train_random_agent(episodes=100, grid_size=5, ships_config=None):
    """
    Train the random agent (no actual training, just collecting performance data).

    Args:
        episodes: Number of games to play
        grid_size: Size of the game grid
        ships_config: Configuration of ships [ship1_size, ship2_size, ...]

    Returns:
        agent: The trained agent
        results: Dictionary containing performance metrics
    """
    if ships_config is None:
        ships_config = [3, 2]  # Default ship configuration for 5x5 grid

    env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)
    agent = RandomAgent(grid_size=grid_size)

    # Set up results directory
    results_dir = setup_results_directory(agent.name, "training")

    # Initialize metrics tracking
    shots_history = []
    win_rate = 0

    for episode in range(episodes):
        observation = env.reset()
        agent.reset()
        done = False
        episode_shots = 0

        while not done:
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)
            agent.update(observation, action, reward, next_observation, done, info)

            observation = next_observation
            episode_shots += 1

        # Record metrics
        shots_history.append(episode_shots)
        if env.hits == env.total_ship_cells:
            win_rate += 1

        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")
            print(f"Recent average shots: {np.mean(shots_history[-10:]):.2f}")

    # Calculate final metrics
    win_rate = win_rate / episodes
    avg_shots = np.mean(shots_history)

    # Save results
    results = {
        'episodes': episodes,
        'win_rate': win_rate,
        'avg_shots': avg_shots,
        'shots_history': shots_history
    }

    # Save agent
    agent_path = os.path.join(results_dir, 'random_agent.pkl')
    agent.save(agent_path)

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, episodes + 1),
        'shots': shots_history
    })
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), shots_history)
    plt.title(f'Random Agent Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Shots to Complete Game')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'performance.png'))
    plt.close()

    # Plot moving average
    window_size = min(10, episodes)
    moving_avg = [np.mean(shots_history[max(0, i - window_size):i])
                  for i in range(1, episodes + 1)]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), moving_avg)
    plt.title(f'Random Agent Moving Average ({window_size} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Shots to Complete Game')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'moving_average.png'))
    plt.close()

    print(f"\nTraining completed for Random Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Results saved to: {results_dir}")

    return agent, results


def evaluate_random_agent(agent, test_episodes=50, grid_size=5, ships_config=None, fixed_test=True):
    """
    Evaluate the random agent on test episodes.

    Args:
        agent: The agent to evaluate
        test_episodes: Number of test episodes
        grid_size: Size of the game grid
        ships_config: Configuration of ships
        fixed_test: Whether to use fixed ship positions for testing

    Returns:
        results: Dictionary containing performance metrics
    """
    from battleship_env import create_test_scenario

    if ships_config is None:
        ships_config = [3, 2]  # Default ship configuration for 5x5 grid

    # Set up results directory
    results_dir = setup_results_directory(agent.name, "testing")

    # Initialize metrics tracking
    shots_history = []
    win_rate = 0

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

        while not done:
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            episode_shots += 1

            # Prevent infinite loops
            if episode_shots > grid_size * grid_size:
                break

        # Record metrics
        shots_history.append(episode_shots)
        if env.hits == env.total_ship_cells:
            win_rate += 1

    # Calculate final metrics
    win_rate = win_rate / test_episodes
    avg_shots = np.mean(shots_history)

    # Save results
    results = {
        'test_episodes': test_episodes,
        'win_rate': win_rate,
        'avg_shots': avg_shots,
        'shots_history': shots_history,
        'fixed_test': fixed_test
    }

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, test_episodes + 1),
        'shots': shots_history
    })
    metrics_df.to_csv(os.path.join(results_dir, 'test_metrics.csv'), index=False)

    # Plot histogram of shots distribution
    plt.figure(figsize=(10, 6))
    plt.hist(shots_history, bins=range(min(shots_history), max(shots_history) + 2))
    plt.title(f'Random Agent Test Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    plt.xlabel('Shots to Complete Game')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'test_histogram.png'))
    plt.close()

    print(f"\nEvaluation completed for Random Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Results saved to: {results_dir}")

    return results


if __name__ == "__main__":
    # Example usage
    # Train the random agent
    agent, train_results = train_random_agent(episodes=100)

    # Evaluate on fixed test scenario
    test_results_fixed = evaluate_random_agent(agent, test_episodes=50, fixed_test=True)

    # Evaluate on random scenarios
    test_results_random = evaluate_random_agent(agent, test_episodes=50, fixed_test=False)