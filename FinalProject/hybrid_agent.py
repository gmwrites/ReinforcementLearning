import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import time
from battleship_env import BattleshipEnv, create_test_scenario, setup_results_directory
from smart_agent import SmartAgent
from enhanced_qlearning_agent import EnhancedQLearningAgent


class HybridAgent:
    """
    A hybrid agent that combines Q-learning for exploration and Smart Agent logic for exploitation.
    Uses Q-learning to find initial hits, then switches to the Smart Agent's strategy to sink ships.
    """

    def __init__(self, grid_size=5, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=0.3):
        self.grid_size = grid_size
        self.name = "HybridAgent"

        # Initialize both underlying agents
        self.q_agent = EnhancedQLearningAgent(
            grid_size=grid_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            # Use fixed exploration rate for consistency
            exploration_decay=1.0,
            min_exploration_rate=exploration_rate
        )

        self.smart_agent = SmartAgent(grid_size=grid_size)

        # Initialize state
        self.reset()

    def reset(self):
        """Reset the agent's state for a new game."""
        self.available_actions = set(range(self.grid_size * self.grid_size))
        self.q_agent.reset()
        self.smart_agent.reset()
        self.mode = "explore"  # Start in exploration mode
        self.hit_count = 0
        self.last_hit = None

    def act(self, observation):
        """
        Select an action based on the current mode.
        Uses Q-learning for exploration and Smart Agent for exploitation.
        """
        # Update mode based on the observation
        self._update_mode(observation)

        # Ensure available actions are synced between component agents
        self.q_agent.available_actions = self.available_actions.copy()
        self.smart_agent.available_actions = list(self.available_actions)

        # Select action based on current mode
        if self.mode == "explore":
            # Use Q-learning agent for exploration
            action = self.q_agent.act(observation)
        else:  # mode == "exploit"
            # Use Smart Agent for exploitation
            action = self.smart_agent.act(observation)

        # Remove the action from available actions
        if action in self.available_actions:
            self.available_actions.remove(action)

        return action

    def _update_mode(self, observation):
        """
        Update the agent's mode based on the current observation.
        Switches to exploit mode when a hit is found, and back to explore mode when a ship is sunk.
        """
        # Count hits in the current observation
        current_hits = np.sum(observation == 2)

        # If we have new hits, switch to exploit mode
        if current_hits > self.hit_count:
            self.mode = "exploit"

            # Find the new hit location
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if observation[i, j] == 2 and (i, j) != self.last_hit:
                        self.last_hit = (i, j)
                        # Also update the Smart Agent's internal state
                        self.smart_agent.last_hit = i * self.grid_size + j
                        self.smart_agent.mode = "exploit"

                        # Update Smart Agent's hit queue with adjacent cells
                        adjacent_cells = []
                        # Check up
                        if i > 0 and (i - 1) * self.grid_size + j in self.available_actions:
                            adjacent_cells.append((i - 1) * self.grid_size + j)
                        # Check down
                        if i < self.grid_size - 1 and (i + 1) * self.grid_size + j in self.available_actions:
                            adjacent_cells.append((i + 1) * self.grid_size + j)
                        # Check left
                        if j > 0 and i * self.grid_size + (j - 1) in self.available_actions:
                            adjacent_cells.append(i * self.grid_size + (j - 1))
                        # Check right
                        if j < self.grid_size - 1 and i * self.grid_size + (j + 1) in self.available_actions:
                            adjacent_cells.append(i * self.grid_size + (j + 1))

                        self.smart_agent.hit_queue = adjacent_cells

            # Update hit count
            self.hit_count = current_hits

        # Check if we just finished sinking a ship (Smart Agent's hit queue is empty)
        if self.mode == "exploit" and (
                not hasattr(self.smart_agent, 'hit_queue') or len(self.smart_agent.hit_queue) == 0):
            # If no more cells to try around hits, switch back to explore
            self.mode = "explore"

    def update(self, observation, action, reward, next_observation, done, info):
        """Update both underlying agents."""
        # Always update the Q-learning agent so it can learn
        self.q_agent.update(observation, action, reward, next_observation, done, info)

        # Update the Smart Agent
        self.smart_agent.update(observation, action, reward, next_observation, done, info)

        # Update hit count and mode
        self._update_mode(next_observation)

        # Sync available actions
        self.available_actions = self.q_agent.available_actions.intersection(
            set(self.smart_agent.available_actions)
        )

    def save(self, filepath):
        """Save the hybrid agent."""
        q_agent_path = filepath + ".q_agent"
        smart_agent_path = filepath + ".smart_agent"

        self.q_agent.save(q_agent_path)
        self.smart_agent.save(smart_agent_path)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'grid_size': self.grid_size,
                'q_agent_path': q_agent_path,
                'smart_agent_path': smart_agent_path
            }, f)

    def load(self, filepath):
        """Load the hybrid agent."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.name = data['name']
            self.grid_size = data['grid_size']

            self.q_agent.load(data['q_agent_path'])
            self.smart_agent.load(data['smart_agent_path'])

            self.reset()


def train_hybrid_agent(episodes=10000, grid_size=5, ships_config=None,
                       learning_rate=0.1, discount_factor=0.9,
                       exploration_rate=0.3, save_interval=1000):
    """
    Train the hybrid agent.

    Args:
        episodes: Number of games to play
        grid_size: Size of the game grid
        ships_config: Configuration of ships [ship1_size, ship2_size, ...]
        learning_rate: Learning rate for Q-learning component
        discount_factor: Discount factor for future rewards
        exploration_rate: Fixed exploration rate for Q-learning component
        save_interval: How often to save checkpoints

    Returns:
        agent: The trained agent
        results: Dictionary containing performance metrics
    """
    if ships_config is None:
        ships_config = [3, 2]  # Default ship configuration for 5x5 grid

    env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)
    agent = HybridAgent(grid_size=grid_size, learning_rate=learning_rate,
                        discount_factor=discount_factor,
                        exploration_rate=exploration_rate)

    # Set up results directory
    results_dir = setup_results_directory(agent.name, "training")

    # Create a subdirectory for checkpoints
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Initialize metrics tracking
    shots_history = []
    rewards_history = []
    win_rate = 0
    mode_switches = []  # Track how often the agent switches modes

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
        episode_mode_switches = 0
        current_mode = "explore"

        while not done:
            action = agent.act(observation)

            # Count mode switches
            if agent.mode != current_mode:
                episode_mode_switches += 1
                current_mode = agent.mode

            # Apply enhanced rewards
            next_observation, reward, done, info = env.step(action)
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
        mode_switches.append(episode_mode_switches)
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

            avg_mode_switches = np.mean(mode_switches[-100:])

            print(f"Episode {episode + 1}/{episodes} | " +
                  f"Mode: {agent.mode} | " +
                  f"Recent avg shots: {moving_avg_shots[-1]:.2f} | " +
                  f"Recent avg mode switches: {avg_mode_switches:.2f} | " +
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
    avg_mode_switches = np.mean(mode_switches)

    # Save results
    results = {
        'episodes': episodes,
        'win_rate': win_rate,
        'avg_shots': avg_shots,
        'avg_reward': avg_reward,
        'avg_mode_switches': avg_mode_switches,
        'shots_history': shots_history,
        'rewards_history': rewards_history,
        'mode_switches': mode_switches,
        'moving_avg_shots': moving_avg_shots,
        'moving_avg_rewards': moving_avg_rewards
    }

    # Save agent
    agent_path = os.path.join(results_dir, 'hybrid_agent.pkl')
    agent.save(agent_path)

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, episodes + 1),
        'shots': shots_history,
        'reward': rewards_history,
        'mode_switches': mode_switches,
        'moving_avg_shots': moving_avg_shots,
        'moving_avg_rewards': moving_avg_rewards
    })
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)

    # Plot final learning curves
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot shots curve
    ax1.plot(range(1, episodes + 1), shots_history, alpha=0.3, color='blue')
    ax1.plot(range(1, episodes + 1), moving_avg_shots, linewidth=2, color='blue')
    ax1.set_title(f'Hybrid Agent Shots Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Shots to Complete Game')
    ax1.grid(True)
    ax1.legend(['Raw data', f'Moving avg (window={window_size})'])

    # Plot rewards curve
    ax2.plot(range(1, episodes + 1), rewards_history, alpha=0.3, color='green')
    ax2.plot(range(1, episodes + 1), moving_avg_rewards, linewidth=2, color='green')
    ax2.set_title(f'Hybrid Agent Rewards\nAvg. Reward: {avg_reward:.2f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Episode Reward')
    ax2.grid(True)
    ax2.legend(['Raw data', f'Moving avg (window={window_size})'])

    # Plot mode switches
    moving_avg_switches = [np.mean(mode_switches[max(0, i - window_size):i])
                           for i in range(1, episodes + 1)]

    ax3.plot(range(1, episodes + 1), mode_switches, alpha=0.3, color='purple')
    ax3.plot(range(1, episodes + 1), moving_avg_switches, linewidth=2, color='purple')
    ax3.set_title(f'Mode Switches Per Episode\nAvg. Switches: {avg_mode_switches:.2f}')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Number of Mode Switches')
    ax3.grid(True)
    ax3.legend(['Raw data', f'Moving avg (window={window_size})'])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance.png'))
    plt.close()

    print(f"\nTraining completed for Hybrid Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average mode switches: {avg_mode_switches:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Results saved to: {results_dir}")

    return agent, results


def evaluate_hybrid_agent(agent, test_episodes=50, grid_size=5, ships_config=None, fixed_test=True):
    """
    Evaluate the hybrid agent on test episodes.

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
    rewards_history = []
    win_rate = 0
    mode_history = []  # Track which mode was used more

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
        explore_count = 0
        exploit_count = 0

        while not done:
            action = agent.act(observation)

            # Track which mode was used
            if agent.mode == "explore":
                explore_count += 1
            else:
                exploit_count += 1

            # Apply enhanced rewards
            next_observation, reward, done, info = env.step(action)
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

            # Prevent infinite loops
            if episode_shots > grid_size * grid_size:
                break

        # Record metrics
        shots_history.append(episode_shots)
        rewards_history.append(episode_reward)
        mode_history.append((explore_count, exploit_count))
        if env.hits == env.total_ship_cells:
            win_rate += 1

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Evaluated {episode + 1}/{test_episodes} episodes | " +
                  f"Avg shots: {np.mean(shots_history[-10:]):.2f}")

    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.1f}s")

    # Calculate final metrics
    win_rate = win_rate / test_episodes
    avg_shots = np.mean(shots_history)
    avg_reward = np.mean(rewards_history)

    # Calculate mode usage statistics
    total_shots = sum(shots_history)
    total_explore = sum(e for e, _ in mode_history)
    total_exploit = sum(x for _, x in mode_history)

    explore_pct = (total_explore / total_shots) * 100 if total_shots > 0 else 0
    exploit_pct = (total_exploit / total_shots) * 100 if total_shots > 0 else 0

    # Save results
    results = {
        'test_episodes': test_episodes,
        'win_rate': win_rate,
        'avg_shots': avg_shots,
        'avg_reward': avg_reward,
        'shots_history': shots_history,
        'rewards_history': rewards_history,
        'explore_percentage': explore_pct,
        'exploit_percentage': exploit_pct,
        'fixed_test': fixed_test
    }

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, test_episodes + 1),
        'shots': shots_history,
        'reward': rewards_history,
        'explore_count': [e for e, _ in mode_history],
        'exploit_count': [x for _, x in mode_history]
    })
    metrics_df.to_csv(os.path.join(results_dir, 'test_metrics.csv'), index=False)

    # Plot histogram of shots distribution
    plt.figure(figsize=(10, 6))
    plt.hist(shots_history, bins=range(min(shots_history), max(shots_history) + 2), alpha=0.7)
    plt.title(f'Hybrid Agent Test Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    plt.xlabel('Shots to Complete Game')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Add mean line
    plt.axvline(avg_shots, color='r', linestyle='dashed', linewidth=2)
    plt.text(avg_shots + 0.5, plt.ylim()[1] * 0.9, f'Mean: {avg_shots:.2f}', color='r')

    plt.savefig(os.path.join(results_dir, 'test_histogram.png'))
    plt.close()

    # Plot pie chart of mode usage
    plt.figure(figsize=(8, 8))
    plt.pie([explore_pct, exploit_pct], labels=['Explore', 'Exploit'],
            autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    plt.title('Mode Usage Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.savefig(os.path.join(results_dir, 'mode_distribution.png'))
    plt.close()

    print(f"\nEvaluation completed for Hybrid Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Mode distribution: {explore_pct:.1f}% Explore, {exploit_pct:.1f}% Exploit")
    print(f"Results saved to: {results_dir}")

    return results


def compare_with_other_agents(hybrid_agent, episodes=50, grid_size=5, ships_config=None):
    """
    Compare the hybrid agent with random, smart, and enhanced Q-learning agents.

    Args:
        hybrid_agent: The trained hybrid agent
        episodes: Number of test episodes
        grid_size: Size of the game grid
        ships_config: Configuration of ships

    Returns:
        results: Dictionary containing comparison metrics
    """
    from random_agent import RandomAgent
    from smart_agent import SmartAgent
    from enhanced_qlearning_agent import EnhancedQLearningAgent
    from tournament_comparison import TournamentComparison

    # Create other agents
    random_agent = RandomAgent(grid_size=grid_size)
    smart_agent = SmartAgent(grid_size=grid_size)
    q_agent = EnhancedQLearningAgent(grid_size=grid_size)

    # Create tournament with custom names
    agents = [random_agent, smart_agent, q_agent, hybrid_agent]
    custom_names = ["Random", "Smart", "Q-Learning", "Hybrid"]

    tournament = TournamentComparison(agents)

    # Run tournament
    results = tournament.run_tournament(episodes=episodes)

    return results, tournament


if __name__ == "__main__":
    # Train the hybrid agent
    agent, train_results = train_hybrid_agent(
        episodes=5000,
        grid_size=5,
        ships_config=[3, 2],
        exploration_rate=0.3
    )

    # Evaluate the agent
    test_results = evaluate_hybrid_agent(
        agent,
        test_episodes=100,
        fixed_test=False
    )

    # Compare with other agents
    comparison_results, tournament = compare_with_other_agents(
        agent,
        episodes=50
    )