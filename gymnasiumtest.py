import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make('ALE/Centipede-v5', render_mode='human')
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
## print (obs,reward,terminated,truncated,info) # Show initial state



total_reward = 0.0
total_steps = 0
obs = env.reset()
curr_lives = info['lives']

# Main loop with one life as an episode

while curr_lives == info['lives']:
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    total_steps += 1
    env.render()
    if done:
        break

# post-episode info
print ("Final state:")
print (obs,reward,terminated,truncated,info)
print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
env.close()

