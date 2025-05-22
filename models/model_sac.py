from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from env_v3 import StockMarketEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from ProfitCallback import ProfitLoggingCallback
import torch
import numpy as np
import os


env = StockMarketEnv(step='test')

model = SAC.load(os.getcwd() + "\\best_models\SAC")
mean_reward_episode = []
final_profit_episode = []
for _ in range(10):
    obs = env.reset()[0]
    dones = False
    profits = []
    rewards = []
    while not dones:
        action, _states = model.predict(obs)
        obs, reward, dones, truncated, profit = env.step(action)
        profits.append(profit)
        rewards.append(reward)
    mean_reward_episode.append(np.array(rewards).mean())
    final_profit_episode.append(profits[-1])
print("Mean reward of all episodes: " + str(np.array(mean_reward_episode).mean()))
print("Mean final profit of all episodes: " + str(np.array(final_profit_episode).mean()))