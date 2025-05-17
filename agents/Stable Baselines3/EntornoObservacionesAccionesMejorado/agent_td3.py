from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from env_v3 import StockMarketEnv
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

model = TD3("MlpPolicy", env, verbose = 1, learning_rate=0.0003, learning_starts=10000, buffer_size=100000, tensorboard_log="./TD3_stocktrading_tensorboard/", replay_buffer_class=ReplayBuffer, device="cuda")
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs-TD3/',
                             log_path='./logs-TD3/', eval_freq=25000,
                             deterministic=True, render=False)
profit_callback = ProfitLoggingCallback()
callback = CallbackList([profit_callback, eval_callback])
model.learn(total_timesteps=2_000_000, log_interval=5, callback=callback, progress_bar=True, tb_log_name = "TD3_2M_lrStable")

# Guardar y cargar el modelo
model.save("TD3_trading")
model = TD3.load("TD3_trading")
