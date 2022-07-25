import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from utils import reward_engineering
import tensorflow as tf

NUM_EPISODES = 30  # Number of episodes used for evaluation
fig_format = 'png'  # Format used for saving matplotlib's figures
# fig_format = 'eps'
# fig_format = 'svg'

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.compat.v1.disable_eager_execution()

# Initiating environment
env = gym.make('LunarLander-v2')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Creating the DQN agent (with greedy policy, suited for evaluation)
agent = DQNAgent(state_size, action_size, epsilon=0.0, epsilon_min=0.0)

# Checking if weights from previous learning session exists
if os.path.exists('lunar_lander.h5'):
    print('Loading weights from previous learning session.')
    agent.load("lunar_lander.h5")
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)
return_history = []

for episodes in range(1, NUM_EPISODES + 1):
    # Reset the environment
    state = env.reset()
    # This reshape is needed to keep compatibility with Keras
    state = np.reshape(state, [1, state_size])
    # Cumulative reward is the return since the beginning of the episode
    cumulative_reward = 0.0
    for time in range(1, 500):
        # Render the environment for visualization
        env.render()
        # Select action
        action = agent.act(state)
        # Take action, observe reward and new state
        next_state, reward, done, _ = env.step(action)
        # Reshaping to keep compatibility with Keras
        next_state = np.reshape(next_state, [1, state_size])
        # Making reward engineering to keep compatibility with how training was done
        reward = reward_engineering(state[0], action, reward, next_state[0], done, time)
        state = next_state
        # Accumulate reward
        cumulative_reward = agent.gamma * cumulative_reward + reward
        if done:
            print("episode: {}/{}, time: {}, score: {:.6}, epsilon: {:.3}"
                  .format(episodes, NUM_EPISODES, time, cumulative_reward, agent.epsilon))
            break
    return_history.append(cumulative_reward)

# Prints mean return
print('Mean return: ', np.mean(return_history))

# Plots return history
plt.plot(return_history, 'b')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('dqn_evaluation.' + fig_format, fig_format=fig_format)
