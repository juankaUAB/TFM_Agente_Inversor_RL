from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env_v2 import StockMarketEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from ProfitCallback import ProfitLoggingCallback
import torch
import os

torch.set_num_threads(os.cpu_count())

# Supón que tienes un entorno compatible con Gym: `TradingEnv`
env = DummyVecEnv([lambda: StockMarketEnv()])
eval_env = DummyVecEnv([lambda: StockMarketEnv()])

def lr_schedule(progress):
    # Reduce el learning rate según el progreso
    return 0.002 * (1 - progress)

model = PPO("MlpPolicy", env, verbose = 1, learning_rate = 0.0003, tensorboard_log="./ppo_stocktrading_tensorboard/")
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs-PPO/',
                             log_path='./logs-PPO/', eval_freq=25000,
                             deterministic=True, render=False)
profit_callback = ProfitLoggingCallback()
callback = CallbackList([profit_callback, eval_callback])
model.learn(total_timesteps=200_000, log_interval=5, callback=callback, progress_bar=True, tb_log_name = "PPO_200k_lrStable")

# Guardar y cargar el modelo
model.save("ppo_trading")
model = PPO.load("ppo_trading")