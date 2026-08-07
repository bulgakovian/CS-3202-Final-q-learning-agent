# CSPB 3202 Final Project - Actor-Critic CNN Reinforcement learning
# Andrew Metzroth
# 
# A learning agent that uses a Convolution Neural network (CNN)
# To learn the Atari game Centipede.
# 
# Much of the code base for this project comes from the extremely helpful tutorial:
# https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
# 
# My major alterations include:
# - Adapting the code base to run on the Arcade Learning Environment in Gymnasium
# - Adding a CNN to the Actor-Critic model
# - Creating replay logs and adding statistics
# - Adapting parameters and reward requirements to fit the Centipede problem


import collections
import gymnasium as gym
import ale_py
import numpy as np
import statistics
import tensorflow as tf
import tqdm

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


# Set seed for experiment reproducibility
#seed = 42
#tf.random.set_seed(seed)
#np.random.seed(seed)

############### Classes and Functions ###############

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network. Modified to use a Convolution neural Network"""

  def __init__(
      self,
      num_actions: int,
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    # Convolutional layers with pooling layers
    self.conv1 = layers.Conv2D(84,(3,3), activation='relu', input_shape=(1,84,84,1))
    self.pool1 = layers.MaxPooling2D((2, 2))
    self.conv2 = layers.Conv2D(64, (3,3), strides=2, activation="relu")
    self.pool2 = layers.MaxPooling2D((2, 2))
    self.conv3 = layers.Conv2D(32, (3,3), strides=1, activation="relu")
    
    # Flatten features
    self.flatten = layers.Flatten()
    self.dense = layers.Dense(512, activation="relu")

    # Actor and critic head
    self.actor_output = layers.Dense(num_actions, name="actor")
    self.critic_output = layers.Dense(1, name="critic")

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Step through the neural network.
    The final layer returns a series of 18 logits for the actor
    and a single reward modifier for the critic.
    """
    inputs = tf.expand_dims(inputs,0)
    x = self.conv1(inputs)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.dense(x)
    return self.actor_output(x), self.critic_output(x)
# Wrap Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

@tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32])
def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, truncated, info = env.step(action)
  return (state.astype(np.float32),
          np.array(reward, np.int32),
          np.array(done, np.int32))

def run_episode(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state
  done = False

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    # state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    # Softmax here gives us a probability distribution.
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = env_step(action)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards

@tf.function
def get_expected_return(
    rewards: tf.Tensor,
    gamma: float,
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) /
               (tf.math.reduce_std(returns) + eps))

  return returns



# We are using L1 loss here because it is not as sensitive to outliers.
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

@tf.function
def compute_loss(
    action_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined Actor-Critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode)

    # Calculate the expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

    # Calculate the loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = int(tf.math.reduce_sum(rewards))

  return episode_reward

def render_episode(env: gym.Env, model: tf.keras.Model, max_steps: int):
  state, _ = env.reset()
  state = tf.constant(state, dtype=tf.float32)
  for i in range(1, max_steps + 1):
    state = tf.expand_dims(state, axis = -1)
    action_probs, _ = model(state)
    action = np.argmax(np.squeeze(action_probs))

    state, reward, done, truncated, info = env.step(action)
    state = tf.constant(state, dtype=tf.float32)
    env.render()
    if done:
      break
  env.close()
  return


############### Running Code ###############

# Create the environment
env = gym.make("ALE/Centipede-v5", frameskip=1)
env = gym.wrappers.AtariPreprocessing(env,frame_skip=4,terminal_on_life_loss = True, grayscale_newaxis = True)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

num_actions = env.action_space.n 
num_hidden_units = 128

model = ActorCritic(int(num_actions), num_hidden_units)
model.summary()

# Trial variables
min_episodes_criterion = 100
max_episodes = 5000
max_steps_per_episode = 7500


# Training will stop if the agent converges to this average reward after the minimum number of episodes.
reward_threshold = 40000
running_reward = 0

# The discount factor for future rewards
gamma = 0.99

# Keep the last episode's reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
t = tqdm.trange(max_episodes)
with tf.device("GPU:0"):
  for i in t:
      initial_state, info = env.reset()
      initial_state = tf.constant(initial_state, dtype=tf.float32)
      episode_reward = int(train_step(
          initial_state, model, optimizer, gamma, max_steps_per_episode))

      episodes_reward.append(episode_reward)
      running_reward = statistics.mean(episodes_reward)


      t.set_postfix(
          episode_reward=episode_reward, running_reward=running_reward)

      # Show the average episode reward every 100 episodes
      if i % 100 == 0:
        print()
        print(f'Episode {i}: average reward: {running_reward}')

      if running_reward > reward_threshold and i >= min_episodes_criterion:
          break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
model.summary()


# Render an episode and save as an MP4
env = gym.make('ALE/Centipede-v5', frameskip=1, render_mode='rgb_array')
env = gym.wrappers.AtariPreprocessing(env,frame_skip=4,terminal_on_life_loss = True)
env = gym.wrappers.RecordVideo(env, video_folder = 'centipede_agents', name_prefix = 'cnnAgent_pool'+str(max_episodes), episode_trigger=lambda x: True)

render_episode(env, model, max_steps_per_episode)