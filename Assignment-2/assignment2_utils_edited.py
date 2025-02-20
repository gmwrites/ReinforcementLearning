#!/usr/bin/python3
import time
import gym

#---------------------------
# Helper functions
#---------------------------
class Taxi:
    '''@brief Describes the environment actions, observation states, and reward range
    '''
    def describe_env(self, env: gym.Env):
        num_actions = env.action_space.n
        obs = env.observation_space
        num_obs = env.observation_space.n
        reward_range = env.reward_range
        action_desc = { 
            0: "Move south (down)",
            1: "Move north (up)",
            2: "Move east (right)",
            3: "Move west (left)",
            4: "Pickup passenger",
            5: "Drop off passenger"
        }
        print("Observation space: ", obs)
        print("Observation space size: ", num_obs)
        print("Reward Range: ", reward_range)
        
        print("Number of actions: ", num_actions)
        print("Action description: ", action_desc)
        return num_obs, num_actions


    '''@brief Get the string description of the action
    '''
    def get_action_description(self, action):
        action_desc = { 
            0: "Move south (down)",
            1: "Move north (up)",
            2: "Move east (right)",
            3: "Move west (left)",
            4: "Pickup passenger",
            5: "Drop off passenger"
        }
        return action_desc[action]

    '''@brief print full description of current observation
    '''
    def describe_obs(self, obs):
        obs_desc = {
            0: "Red",
            1: "Green",
            2: "Yellow",
            3: "Blue",
            4: "In taxi"
        }
        obs_dict = self.breakdown_obs(obs)
        print("Passenger is at: {0}, wants to go to {1}. Taxi currently at ({2}, {3})".format(
            obs_desc[obs_dict["passenger_location"]], 
            obs_desc[obs_dict["destination"]], 
            obs_dict["taxi_row"], 
            obs_dict["taxi_col"]))

    '''@brief Takes an observation for the 'taxi-v3' environment and returns details observation space description
        @details returns a dict with "destination", "passenger_location", "taxi_col", "taxi_row"
        @see: https://gymnasium.farama.org/environments/toy_text/taxi/
    '''
    def breakdown_obs(self, obs):
        # ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination = X
        # X % 4 --> destination
        destination = obs % 4
        # X -= remainder, X /= 4
        obs -= destination
        obs /= 4
        # X % 5 --> passenger_location
        passenger_location = obs % 5
        # X -= remainder, X /= 5
        obs -= passenger_location
        obs /= 5
        # X % 5 --> taxi_col
        taxi_col = obs % 5
        # X -= remainder, X /=5 
        obs -= taxi_col
        # X --> taxi_row
        taxi_row = obs
        observation_dict= {
            "destination": destination, 
            "passenger_location": passenger_location,
            "taxi_row": taxi_row, 
            "taxi_col": taxi_col
        }
        return observation_dict


    '''@brief simulate the environment with the agents taught policy
    '''
    def simulate_episodes(self, env, agent, num_episodes=3):
        for _ in range(num_episodes):
            done = False
            state, _ = env.reset()
            self.describe_obs(state)
            env.render()
            while not done:
                # Random choice from behavior policy
                action = agent.select_action(state)
                # take a step
                env.render()
                time.sleep(0.1)
                next_state, _, done, _, _ = env.step(action)
                state = next_state
            time.sleep(1.0)

    def main(self):
        # Note: Use v3 for the latest version
        env = gym.make('Taxi-v3')
        num_obs, num_actions = self.describe_env(env)


        # TODO: Train
        agent = Agent(num_obs, num_actions)
        agent.train(env, 5000)
        
        # TODO: Simulate
        # Note how here, we change the render mode for testing/simulation
        env2 = gym.make('Taxi-v3', render_mode="human")
        self.simulate_episodes(env2, agent)

    if __name__=="__main__":
        main()


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
