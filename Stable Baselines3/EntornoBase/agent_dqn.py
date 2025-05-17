from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockMarketEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.buffers import ReplayBuffer
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

model = DQN("MlpPolicy", env, verbose = 1, exploration_fraction=0.2, batch_size=64, learning_rate=0.0003, learning_starts=7500, buffer_size=75000, tensorboard_log="./dqn_stocktrading_tensorboard/", replay_buffer_class=ReplayBuffer)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs-DQN/',
                             log_path='./logs-DQN/', eval_freq=25000,
                             deterministic=True, render=False)
profit_callback = ProfitLoggingCallback()
callback = CallbackList([profit_callback, eval_callback])
model.learn(total_timesteps=200_000, log_interval=5, callback=callback, progress_bar=True, tb_log_name = "DQN_200k_lrStable")

# Guardar y cargar el modelo
model.save("DQN_trading")
model = DQN.load("DQN_trading")
