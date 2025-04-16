import time
import numpy as np
from env import StockMarketEnv

env = StockMarketEnv(save_to_csv=True)
env.reset()

# Espacio de acciones
print("Espacio de acciones es {} ".format(env.action_space))

# Espacio de observacions
print("Espacio de estados es {} ".format(env.observation_space))

# Set the number of episodes
episodes = 1
total_rewards = []

for episode in range(1, episodes+1):
    state = env.reset()  # Reset the environment at the start of each episode
    done = False
    score = 0
    step = 0

    while not done:
        action = env.action_space.sample()  # Randomly sample an action
        n_state, reward, terminated, truncated, info = env.step(action)  # Take the action
        done = terminated or truncated  # Check if the episode is done
        score += reward  # Accumulate the reward
        step += 1

    total_rewards.append(score)  # Store the reward for the episode
    #print(f'Episode {episode} finished with score: {score}')
    #print('------------------------------------')

# Calculate and print the mean reward over all episodes
mean_reward = np.mean(total_rewards)
print(f'\nMean reward over {episodes} episodes: {mean_reward}')