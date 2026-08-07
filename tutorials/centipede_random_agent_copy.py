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

class CentipedeRandomAgent:
    def __init__(
            self,
            env
        ):
            """Initialize a Random agent
            """
    def get_action(self, env, obs) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        return env.action_space.sample()

    


# Create environment
gym.register_envs(ale_py)
env = gym.make('ALE/Centipede-v5')
agent = CentipedeRandomAgent(env)



# Play one full game (all lives) with learning
total_reward = 0.0
total_steps = 0
env = gym.make('ALE/Centipede-v5', frameskip = 1)
env = gym.wrappers.AtariPreprocessing(env,frame_skip=4,terminal_on_life_loss = True)
obs, info = env.reset()
done = False
log = []
ep_num = 0
n_episodes = 100

for episode in tqdm(range(n_episodes)):
    done=False
    tot_reward = 0
    tot_steps = 0
    obs,info = env.reset()

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    while not done:
        action = agent.get_action(env, tuple(obs))
        next_obs, reward, terminated, truncated, info = env.step(action)
        tot_reward += reward
        tot_steps += 1


        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs
    log.append((ep_num,tot_steps,tot_reward))
    ep_num +=1

# Log episodes
with open('output.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode','Total steps','Total reward'])
    writer.writerows(log)
