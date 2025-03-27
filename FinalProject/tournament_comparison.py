import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from datetime import datetime
from battleship_env import BattleshipEnv, create_test_scenario, setup_results_directory
from random_agent import RandomAgent
from smart_agent import SmartAgent
from qlearning_agent import QLearningAgent


class TournamentComparison:
    """
    Class to run tournament-style comparisons between multiple Battleship agents.
    Each agent plays on identical board configurations, and their performance is tracked.
    """

    def __init__(self, agents, grid_size=5, ships_config=None, results_dir=None):
        """
        Initialize the tournament comparison.

        Args:
            agents: List of agent objects to compare
            grid_size: Size of the game grid
            ships_config: Configuration of ships [ship1_size, ship2_size, ...]
            results_dir: Directory to save results (if None, a new one will be created)
        """
        self.agents = agents
        self.agent_names = [agent.name for agent in agents]
        self.grid_size = grid_size

        if ships_config is None:
            self.ships_config = [3, 2]  # Default ship configuration for 5x5 grid
        else:
            self.ships_config = ships_config

        if results_dir is None:
            self.results_dir = setup_results_directory("tournament", "comparison")
        else:
            self.results_dir = results_dir
            os.makedirs(self.results_dir, exist_ok=True)

        # Initialize metrics tracking
        self.shots_history = {name: [] for name in self.agent_names}
        self.wins = {name: 0 for name in self.agent_names}
        self.matchups = {name: {} for name in self.agent_names}

        # For each agent, initialize matchup stats against all other agents
        for name1 in self.agent_names:
            for name2 in self.agent_names:
                if name1 != name2:
                    self.matchups[name1][name2] = {'wins': 0, 'total': 0}

    def run_tournament(self, episodes=100, fixed_test=False):
        """
        Run a tournament between all agents.

        Args:
            episodes: Number of episodes (games) to run
            fixed_test: Whether to use fixed ship positions or random ones

        Returns:
            Dictionary containing tournament results
        """
        global BattleshipEnv
        for episode in range(episodes):
            # Generate a single board configuration for this episode
            if fixed_test:
                # Use a standard test scenario
                base_env = create_test_scenario(grid_size=self.grid_size, ships_config=self.ships_config)
                ship_grid = base_env.ship_grid.copy()
            else:
                # Create a random board configuration
                base_env = BattleshipEnv(grid_size=self.grid_size, ships_config=self.ships_config)
                ship_grid = base_env.ship_grid.copy()

            # Track shots for each agent on this episode
            episode_shots = {}

            # Each agent plays on the same board configuration
            for agent in self.agents:
                # Reset the agent
                agent.reset()
                import numpy as np
                import matplotlib.pyplot as plt
                import pandas as pd
                import os
                import pickle
                from datetime import datetime
                from battleship_env import BattleshipEnv, create_test_scenario, setup_results_directory
                from random_agent import RandomAgent
                from smart_agent import SmartAgent
                from qlearning_agent import QLearningAgent

                class TournamentComparison:
                    """
                    Class to run tournament-style comparisons between multiple Battleship agents.
                    Each agent plays on identical board configurations, and their performance is tracked.
                    """

                    def __init__(self, agents, agent_names=None, grid_size=5, ships_config=None, results_dir=None):
                        """
                        Initialize the tournament comparison.

                        Args:
                            agents: List of agent objects to compare
                            agent_names: Optional list of custom names for the agents (must match length of agents)
                            grid_size: Size of the game grid
                            ships_config: Configuration of ships [ship1_size, ship2_size, ...]
                            results_dir: Directory to save results (if None, a new one will be created)
                        """
                        self.agents = agents

                        # Use custom agent names if provided, otherwise use the agent object names
                        if agent_names is not None:
                            if len(agent_names) != len(agents):
                                raise ValueError("Number of agent names must match number of agents")
                            self.agent_names = agent_names
                        else:
                            self.agent_names = [agent.name for agent in agents]

                        # Create a lookup dict to map display names to agent objects
                        self.agent_lookup = {name: agent for name, agent in zip(self.agent_names, self.agents)}

                        self.grid_size = grid_size

                        if ships_config is None:
                            self.ships_config = [3, 2]  # Default ship configuration for 5x5 grid
                        else:
                            self.ships_config = ships_config

                        if results_dir is None:
                            self.results_dir = setup_results_directory("tournament", "comparison")
                        else:
                            self.results_dir = results_dir
                            os.makedirs(self.results_dir, exist_ok=True)

                        # Initialize metrics tracking
                        self.shots_history = {name: [] for name in self.agent_names}
                        self.wins = {name: 0 for name in self.agent_names}
                        self.matchups = {name: {} for name in self.agent_names}

                        # For each agent, initialize matchup stats against all other agents
                        for name1 in self.agent_names:
                            for name2 in self.agent_names:
                                if name1 != name2:
                                    self.matchups[name1][name2] = {'wins': 0, 'total': 0}

                    def run_tournament(self, episodes=100, fixed_test=False):
                        """
                        Run a tournament between all agents.

                        Args:
                            episodes: Number of episodes (games) to run
                            fixed_test: Whether to use fixed ship positions or random ones

                        Returns:
                            Dictionary containing tournament results
                        """
                        for episode in range(episodes):
                            # Generate a single board configuration for this episode
                            if fixed_test:
                                # Use a standard test scenario
                                base_env = create_test_scenario(grid_size=self.grid_size,
                                                                ships_config=self.ships_config)
                                ship_grid = base_env.ship_grid.copy()
                            else:
                                # Create a random board configuration
                                base_env = BattleshipEnv(grid_size=self.grid_size, ships_config=self.ships_config)
                                ship_grid = base_env.ship_grid.copy()

                            # Track shots for each agent on this episode
                            episode_shots = {}

                            # Each agent plays on the same board configuration
                            for agent in self.agents:
                                # Reset the agent
                                agent.reset()

                                # Create a new environment with the same ship configuration
                                env = BattleshipEnv(grid_size=self.grid_size, ships_config=self.ships_config)
                                env.ship_grid = ship_grid.copy()
                                env.reset()
                                env.ship_grid = ship_grid.copy()  # Ensure ship positions are preserved

                                # Run the agent on this environment
                                observation = env.grid.copy()
                                done = False
                                agent_shots = 0

                                while not done:
                                    action = agent.act(observation)
                                    next_observation, reward, done, info = env.step(action)

                                    # If the agent supports update method, use it
                                    if hasattr(agent, 'update') and callable(getattr(agent, 'update')):
                                        agent.update(observation, action, reward, next_observation, done, info)

                                    observation = next_observation
                                    agent_shots += 1

                                    # Prevent infinite loops
                                    if agent_shots > self.grid_size * self.grid_size:
                                        break

                                # Record metrics for this agent
                                self.shots_history[agent.name].append(agent_shots)
                                episode_shots[agent.name] = agent_shots

                            # Determine the winner for this episode (agent with fewest shots)
                            winner = min(episode_shots, key=episode_shots.get)
                            self.wins[winner] += 1

                            # Update head-to-head matchups
                            for name1 in self.agent_names:
                                for name2 in self.agent_names:
                                    if name1 != name2:
                                        self.matchups[name1][name2]['total'] += 1
                                        # name1 wins if it used fewer shots than name2
                                        if episode_shots[name1] < episode_shots[name2]:
                                            self.matchups[name1][name2]['wins'] += 1

                            # Print progress
                            if (episode + 1) % 10 == 0:
                                print(f"Tournament episode {episode + 1}/{episodes} completed.")

                        # Calculate final statistics
                        win_percentages = {name: (self.wins[name] / episodes) * 100 for name in self.agent_names}
                        avg_shots = {name: np.mean(self.shots_history[name]) for name in self.agent_names}

                        # Save and visualize results
                        self._save_results(episodes)
                        self._visualize_results(episodes)

                        return {
                            'wins': self.wins,
                            'win_percentages': win_percentages,
                            'avg_shots': avg_shots,
                            'shots_history': self.shots_history,
                            'matchups': self.matchups
                        }

                    def _save_results(self, episodes):
                        """Save tournament results to files."""
                        # Save overall statistics
                        stats_data = {
                            'Agent': self.agent_names,
                            'Wins': [self.wins[name] for name in self.agent_names],
                            'Win Percentage': [(self.wins[name] / episodes) * 100 for name in self.agent_names],
                            'Avg Shots': [np.mean(self.shots_history[name]) for name in self.agent_names],
                            'Min Shots': [np.min(self.shots_history[name]) for name in self.agent_names],
                            'Max Shots': [np.max(self.shots_history[name]) for name in self.agent_names],
                            'Std Dev': [np.std(self.shots_history[name]) for name in self.agent_names]
                        }

                        stats_df = pd.DataFrame(stats_data)
                        stats_df.to_csv(os.path.join(self.results_dir, 'tournament_stats.csv'), index=False)

                        # Save episode-by-episode shots
                        shots_df = pd.DataFrame({
                            'Episode': range(1, episodes + 1),
                            **{name: self.shots_history[name] for name in self.agent_names}
                        })
                        shots_df.to_csv(os.path.join(self.results_dir, 'tournament_shots.csv'), index=False)

                        # Save matchup statistics
                        matchup_rows = []
                        for name1 in self.agent_names:
                            for name2 in self.agent_names:
                                if name1 != name2:
                                    matchup = self.matchups[name1][name2]
                                    win_pct = (matchup['wins'] / matchup['total']) * 100 if matchup['total'] > 0 else 0
                                    matchup_rows.append({
                                        'Agent': name1,
                                        'Opponent': name2,
                                        'Wins': matchup['wins'],
                                        'Total': matchup['total'],
                                        'Win Percentage': win_pct
                                    })

                        matchups_df = pd.DataFrame(matchup_rows)
                        matchups_df.to_csv(os.path.join(self.results_dir, 'head_to_head.csv'), index=False)

                    def _visualize_results(self, episodes):
                        """Create visualizations of tournament results."""
                        # 1. Bar chart of wins
                        plt.figure(figsize=(10, 6))

                        positions = range(len(self.agent_names))
                        bars = plt.bar(positions, [self.wins[name] for name in self.agent_names])

                        plt.title(f'Tournament Wins ({episodes} Episodes)')
                        plt.xlabel('Agent')
                        plt.ylabel('Number of Wins')
                        plt.xticks(positions, self.agent_names)
                        plt.grid(True, axis='y')

                        # Add value labels
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                                     f'{height}', ha='center', va='bottom')

                        plt.savefig(os.path.join(self.results_dir, 'tournament_wins.png'))
                        plt.close()

                        # 2. Box plot of shots distribution
                        plt.figure(figsize=(12, 8))

                        plt.boxplot([self.shots_history[name] for name in self.agent_names], labels=self.agent_names)

                        plt.title(f'Shots Distribution Comparison ({episodes} Episodes)')
                        plt.xlabel('Agent')
                        plt.ylabel('Shots to Complete Game')
                        plt.grid(True, axis='y')

                        # Add avg shots text
                        avg_shots = [np.mean(self.shots_history[name]) for name in self.agent_names]
                        for i, avg in enumerate(avg_shots):
                            plt.text(i + 1, avg, f'Avg: {avg:.1f}',
                                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

                        plt.savefig(os.path.join(self.results_dir, 'shots_distribution.png'))
                        plt.close()

                        # 3. Head-to-head matchup matrix
                        agent_count = len(self.agent_names)
                        win_matrix = np.zeros((agent_count, agent_count))

                        for i, name1 in enumerate(self.agent_names):
                            for j, name2 in enumerate(self.agent_names):
                                if name1 == name2:
                                    win_matrix[i, j] = 0  # Self matchup
                                else:
                                    matchup = self.matchups[name1][name2]
                                    win_matrix[i, j] = (matchup['wins'] / matchup['total']) * 100 if matchup[
                                                                                                         'total'] > 0 else 0

                        plt.figure(figsize=(10, 8))
                        plt.imshow(win_matrix, cmap='RdYlGn', vmin=0, vmax=100)
                        plt.colorbar(label='Win Percentage %')

                        plt.title('Head-to-Head Win Percentages')
                        plt.xlabel('Opponent')
                        plt.ylabel('Agent')

                        plt.xticks(range(agent_count), self.agent_names, rotation=45)
                        plt.yticks(range(agent_count), self.agent_names)

                        # Add percentage text in each cell
                        for i in range(agent_count):
                            for j in range(agent_count):
                                if i != j:  # Skip diagonal (self matchups)
                                    plt.text(j, i, f'{win_matrix[i, j]:.1f}%',
                                             ha='center', va='center',
                                             color='black' if 20 < win_matrix[i, j] < 80 else 'white')

                        plt.tight_layout()
                        plt.savefig(os.path.join(self.results_dir, 'head_to_head_matrix.png'))
                        plt.close()

                        # 4. Shots per episode line graph
                        plt.figure(figsize=(12, 8))

                        for name in self.agent_names:
                            plt.plot(range(1, episodes + 1), self.shots_history[name], label=name)

                        plt.title('Shots per Episode')
                        plt.xlabel('Episode')
                        plt.ylabel('Shots to Complete Game')
                        plt.legend()
                        plt.grid(True)

                        plt.savefig(os.path.join(self.results_dir, 'shots_per_episode.png'))
                        plt.close()

                    @staticmethod
                    def load_agent_from_file(filepath):
                        """
                        Load an agent from a file based on its type.

                        Args:
                            filepath: Path to the agent file

                        Returns:
                            The loaded agent
                        """
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                            agent_name = data.get('name', '')

                            if 'RandomAgent' in agent_name:
                                agent = RandomAgent()
                            elif 'SmartAgent' in agent_name:
                                agent = SmartAgent()
                            elif 'QLearningAgent' in agent_name:
                                agent = QLearningAgent()
                            else:
                                raise ValueError(f"Unknown agent type: {agent_name}")

                            agent.load(filepath)
                            return agent

                def example_tournament():
                    """Run an example tournament between different agents."""
                    print("Training agents for tournament...")

                    from random_agent import train_random_agent
                    from smart_agent import train_smart_agent
                    from qlearning_agent import train_qlearning_agent

                    # Train a random agent
                    print("Training Random Agent...")
                    random_agent, _ = train_random_agent(episodes=50)

                    # Train a smart agent
                    print("\nTraining Smart Agent...")
                    smart_agent, _ = train_smart_agent(episodes=50)

                    # Train a Q-learning agent with fewer episodes for quick demonstration
                    print("\nTraining Q-Learning Agent...")
                    qlearning_agent, _ = train_qlearning_agent(episodes=200)

                    # Run tournament with custom names
                    print("\nRunning tournament...")
                    agents = [random_agent, smart_agent, qlearning_agent]
                    custom_names = ["Random Shooter", "Smart Explorer", "Q-Learning Master"]
                    tournament = TournamentComparison(agents, agent_names=custom_names)
                    results = tournament.run_tournament(episodes=100)

                    # Print summary
                    print("\nTournament Results:")
                    print(f"Total episodes: 100")

                    print("\nWins:")
                    for name, wins in results['wins'].items():
                        print(f"{name}: {wins} wins ({results['win_percentages'][name]:.1f}%)")

                    print("\nAverage Shots:")
                    for name, avg in results['avg_shots'].items():
                        print(f"{name}: {avg:.2f} shots on average")

                    return tournament, results

                if __name__ == "__main__":
                    tournament, results = example_tournament()

                    # Alternative: load pre-trained agents from files
                    """
                    # Load agents from files
                    agents = [
                        TournamentComparison.load_agent_from_file('path/to/random_agent.pkl'),
                        TournamentComparison.load_agent_from_file('path/to/smart_agent.pkl'),
                        TournamentComparison.load_agent_from_file('path/to/qlearning_agent.pkl')
                    ]

                    # Run tournament with loaded agents
                    tournament = TournamentComparison(agents)
                    results = tournament.run_tournament(episodes=100)
                    """
                # Create a new environment with the same ship configuration
                env = BattleshipEnv(grid_size=self.grid_size, ships_config=self.ships_config)
                env.ship_grid = ship_grid.copy()
                env.reset()
                env.ship_grid = ship_grid.copy()  # Ensure ship positions are preserved

                # Run the agent on this environment
                observation = env.grid.copy()
                done = False
                agent_shots = 0

                while not done:
                    action = agent.act(observation)
                    next_observation, reward, done, info = env.step(action)

                    # If the agent supports update method, use it
                    if hasattr(agent, 'update') and callable(getattr(agent, 'update')):
                        agent.update(observation, action, reward, next_observation, done, info)

                    observation = next_observation
                    agent_shots += 1

                    # Prevent infinite loops
                    if agent_shots > self.grid_size * self.grid_size:
                        break

                # Record metrics for this agent
                self.shots_history[agent.name].append(agent_shots)
                episode_shots[agent.name] = agent_shots

            # Determine the winner for this episode (agent with fewest shots)
            winner = min(episode_shots, key=episode_shots.get)
            self.wins[winner] += 1

            # Update head-to-head matchups
            for name1 in self.agent_names:
                for name2 in self.agent_names:
                    if name1 != name2:
                        self.matchups[name1][name2]['total'] += 1
                        # name1 wins if it used fewer shots than name2
                        if episode_shots[name1] < episode_shots[name2]:
                            self.matchups[name1][name2]['wins'] += 1

            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Tournament episode {episode + 1}/{episodes} completed.")

        # Calculate final statistics
        win_percentages = {name: (self.wins[name] / episodes) * 100 for name in self.agent_names}
        avg_shots = {name: np.mean(self.shots_history[name]) for name in self.agent_names}

        # Save and visualize results
        self._save_results(episodes)
        self._visualize_results(episodes)

        return {
            'wins': self.wins,
            'win_percentages': win_percentages,
            'avg_shots': avg_shots,
            'shots_history': self.shots_history,
            'matchups': self.matchups
        }

    def _save_results(self, episodes):
        """Save tournament results to files."""
        # Save overall statistics
        stats_data = {
            'Agent': self.agent_names,
            'Wins': [self.wins[name] for name in self.agent_names],
            'Win Percentage': [(self.wins[name] / episodes) * 100 for name in self.agent_names],
            'Avg Shots': [np.mean(self.shots_history[name]) for name in self.agent_names],
            'Min Shots': [np.min(self.shots_history[name]) for name in self.agent_names],
            'Max Shots': [np.max(self.shots_history[name]) for name in self.agent_names],
            'Std Dev': [np.std(self.shots_history[name]) for name in self.agent_names]
        }

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(os.path.join(self.results_dir, 'tournament_stats.csv'), index=False)

        # Save episode-by-episode shots
        shots_df = pd.DataFrame({
            'Episode': range(1, episodes + 1),
            **{name: self.shots_history[name] for name in self.agent_names}
        })
        shots_df.to_csv(os.path.join(self.results_dir, 'tournament_shots.csv'), index=False)

        # Save matchup statistics
        matchup_rows = []
        for name1 in self.agent_names:
            for name2 in self.agent_names:
                if name1 != name2:
                    matchup = self.matchups[name1][name2]
                    win_pct = (matchup['wins'] / matchup['total']) * 100 if matchup['total'] > 0 else 0
                    matchup_rows.append({
                        'Agent': name1,
                        'Opponent': name2,
                        'Wins': matchup['wins'],
                        'Total': matchup['total'],
                        'Win Percentage': win_pct
                    })

        matchups_df = pd.DataFrame(matchup_rows)
        matchups_df.to_csv(os.path.join(self.results_dir, 'head_to_head.csv'), index=False)

    def _visualize_results(self, episodes):
        """Create visualizations of tournament results."""
        # 1. Bar chart of wins
        plt.figure(figsize=(10, 6))

        positions = range(len(self.agent_names))
        bars = plt.bar(positions, [self.wins[name] for name in self.agent_names])

        plt.title(f'Tournament Wins ({episodes} Episodes)')
        plt.xlabel('Agent')
        plt.ylabel('Number of Wins')
        plt.xticks(positions, self.agent_names)
        plt.grid(True, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{height}', ha='center', va='bottom')

        plt.savefig(os.path.join(self.results_dir, 'tournament_wins.png'))
        plt.close()

        # 2. Box plot of shots distribution
        plt.figure(figsize=(12, 8))

        plt.boxplot([self.shots_history[name] for name in self.agent_names], labels=self.agent_names)

        plt.title(f'Shots Distribution Comparison ({episodes} Episodes)')
        plt.xlabel('Agent')
        plt.ylabel('Shots to Complete Game')
        plt.grid(True, axis='y')

        # Add avg shots text
        avg_shots = [np.mean(self.shots_history[name]) for name in self.agent_names]
        for i, avg in enumerate(avg_shots):
            plt.text(i + 1, avg, f'Avg: {avg:.1f}',
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))

        plt.savefig(os.path.join(self.results_dir, 'shots_distribution.png'))
        plt.close()

        # 3. Head-to-head matchup matrix
        agent_count = len(self.agent_names)
        win_matrix = np.zeros((agent_count, agent_count))

        for i, name1 in enumerate(self.agent_names):
            for j, name2 in enumerate(self.agent_names):
                if name1 == name2:
                    win_matrix[i, j] = 0  # Self matchup
                else:
                    matchup = self.matchups[name1][name2]
                    win_matrix[i, j] = (matchup['wins'] / matchup['total']) * 100 if matchup['total'] > 0 else 0

        plt.figure(figsize=(10, 8))
        plt.imshow(win_matrix, cmap='RdYlGn', vmin=0, vmax=100)
        plt.colorbar(label='Win Percentage %')

        plt.title('Head-to-Head Win Percentages')
        plt.xlabel('Opponent')
        plt.ylabel('Agent')

        plt.xticks(range(agent_count), self.agent_names, rotation=45)
        plt.yticks(range(agent_count), self.agent_names)

        # Add percentage text in each cell
        for i in range(agent_count):
            for j in range(agent_count):
                if i != j:  # Skip diagonal (self matchups)
                    plt.text(j, i, f'{win_matrix[i, j]:.1f}%',
                             ha='center', va='center',
                             color='black' if 20 < win_matrix[i, j] < 80 else 'white')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'head_to_head_matrix.png'))
        plt.close()

        # 4. Shots per episode line graph
        plt.figure(figsize=(12, 8))

        for name in self.agent_names:
            plt.plot(range(1, episodes + 1), self.shots_history[name], label=name)

        plt.title('Shots per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Shots to Complete Game')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.results_dir, 'shots_per_episode.png'))
        plt.close()

    @staticmethod
    def load_agent_from_file(filepath):
        """
        Load an agent from a file based on its type.

        Args:
            filepath: Path to the agent file

        Returns:
            The loaded agent
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            agent_name = data.get('name', '')

            if 'RandomAgent' in agent_name:
                agent = RandomAgent()
            elif 'SmartAgent' in agent_name:
                agent = SmartAgent()
            elif 'QLearningAgent' in agent_name:
                agent = QLearningAgent()
            else:
                raise ValueError(f"Unknown agent type: {agent_name}")

            agent.load(filepath)
            return agent


def example_tournament():
    """Run an example tournament between different agents."""
    print("Training agents for tournament...")

    from random_agent import train_random_agent
    from smart_agent import train_smart_agent
    from qlearning_agent import train_qlearning_agent

    # Train a random agent
    print("Training Random Agent...")
    random_agent, _ = train_random_agent(episodes=50)

    # Train a smart agent
    print("\nTraining Smart Agent...")
    smart_agent, _ = train_smart_agent(episodes=50)

    # Train a Q-learning agent with fewer episodes for quick demonstration
    print("\nTraining Q-Learning Agent...")
    qlearning_agent, _ = train_qlearning_agent(episodes=200)

    # Run tournament
    print("\nRunning tournament...")
    agents = [random_agent, smart_agent, qlearning_agent]
    tournament = TournamentComparison(agents)
    results = tournament.run_tournament(episodes=100)

    # Print summary
    print("\nTournament Results:")
    print(f"Total episodes: 100")

    print("\nWins:")
    for name, wins in results['wins'].items():
        print(f"{name}: {wins} wins ({results['win_percentages'][name]:.1f}%)")

    print("\nAverage Shots:")
    for name, avg in results['avg_shots'].items():
        print(f"{name}: {avg:.2f} shots on average")

    return tournament, results


if __name__ == "__main__":
    tournament, results = example_tournament()

    # Alternative: load pre-trained agents from files
    """
    # Load agents from files
    agents = [
        TournamentComparison.load_agent_from_file('path/to/random_agent.pkl'),
        TournamentComparison.load_agent_from_file('path/to/smart_agent.pkl'),
        TournamentComparison.load_agent_from_file('path/to/qlearning_agent.pkl')
    ]

    # Run tournament with loaded agents
    tournament = TournamentComparison(agents)
    results = tournament.run_tournament(episodes=100)
    """