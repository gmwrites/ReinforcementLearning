import random
import torch

class AgentGPU:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table as a PyTorch tensor on GPU
        self.q_table = torch.zeros((num_states, num_actions), dtype=torch.float32).cuda()

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Explore
        else:
            return torch.argmax(self.q_table[state]).item()  # Exploit best known action

    def train(self, env, episodes=5000):
        """Train the agent using Q-learning"""
        for episode in range(episodes):
            state, _ = env.reset()
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = env.step(action)

                # GPU-accelerated Q-table update using PyTorch
                with torch.cuda.device(0):
                    target_q = reward + self.gamma * torch.max(self.q_table[next_state])
                    self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * target_q

                state = next_state