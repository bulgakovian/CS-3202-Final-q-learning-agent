# CSPB 3202 Intro to AI
# Final project: Q-Learning Agent by Andrew "Metz" Metzroth
#
#
# This file uses Gymnasium and the Arcade Learning Environment (ALE) to create and train
# a q-learning agent to play Centipede in a simulated Atari envrionment.
# 
# Some code repurposed from Berkeley CS 188 pac-man reinforcement learning project.
# Also thanks to Gymnasium Blackjack q-learning tutorial for guidance.


import gymnasium as gym
import ale_py
import functools
from collections import defaultdict
import numpy as np
import random, util, math
import csv
from tqdm import tqdm
    
        

class CentipedeQAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
    @functools.lru_cache(maxsize = None)
    def get_action(self, env, obs) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

        
    @functools.lru_cache(maxsize = None)
    def update(
        self,
        obs: tuple,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def stop_learning(self): 
        """End learning and use only exploitation"""
        self.epsilon = 0


# Create environment
env = gym.make('ALE/Centipede-v5')


# hyperparameters
learning_rate = 0.01
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = CentipedeQAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

env = gym.make('ALE/Centipede-v5',frameskip=1)
env = gym.wrappers.AtariPreprocessing(env,frame_skip=4,terminal_on_life_loss = True)
env = gym.wrappers.FlattenObservation(env)
log = []
ep_num = 0
for episode in tqdm(range(n_episodes)):
    tot_reward = 0
    tot_steps = 0
    # pre-process

    obs, info = env.reset()
    curr_lives = info['lives']
    done = False
    # play one episode
    while not done:
        action = agent.get_action(env, tuple(obs))
        next_obs, reward, terminated, truncated, info = env.step(action)
        

        # update the agent

        # reward firing
        if action in [1,10,11,12,13,14,15,16,17]:
            reward +=100
        else:
            reward -=10


        agent.update(tuple(obs), action, reward, terminated, tuple(next_obs))
        tot_reward += reward
        tot_steps +=1
        # update if the environment is done and the current obs
        done = terminated or truncated or (curr_lives != info['lives'])
        obs = next_obs

    # update decay and log results
    agent.decay_epsilon()
    log.append((ep_num,tot_steps,tot_reward))
    ep_num +=1

# Log episodes
with open('output.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode','Total steps','Total reward'])
    writer.writerows(log)


# Training done Play one full game (all lives) with learning
agent.stop_learning()
total_reward = 0.0
total_steps = 0
env = gym.make('ALE/Centipede-v5', frameskip=1, render_mode='rgb_array')
env = gym.wrappers.AtariPreprocessing(env,frame_skip=4,terminal_on_life_loss = True)
env = gym.wrappers.RecordVideo(env, video_folder = 'centipede_agents', name_prefix = 'qagent', episode_trigger=lambda x: True)
env = gym.wrappers.FlattenObservation(env)
obs, info = env.reset()
done = False



obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
while not done:
    action = agent.get_action(env, tuple(obs))
    next_obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    total_steps += 1
    env.render()

    # update if the environment is done and the current obs
    done = terminated or truncated
    obs = next_obs
print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
env.close()