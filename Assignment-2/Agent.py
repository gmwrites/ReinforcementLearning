import numpy as np
import random

class Agent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table
        self.q_table = np.zeros((num_states, num_actions))

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit best known action

    def train(self, env, episodes=5000):
        """Train the agent using Q-learning"""
        for episode in range(episodes):
            state, _ = env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = env.step(action)

                # Update Q-table using Bellman Equation
                self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + \
                                              self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]))

                state = next_state