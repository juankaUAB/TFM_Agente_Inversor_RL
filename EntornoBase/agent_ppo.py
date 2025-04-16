from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import StockMarketEnv
from stable_baselines3.common.callbacks import EvalCallback

# Supón que tienes un entorno compatible con Gym: `TradingEnv`
env = DummyVecEnv([lambda: StockMarketEnv()])
eval_env = DummyVecEnv([lambda: StockMarketEnv()])

def lr_schedule(progress):
    # Reduce el learning rate según el progreso
    return 0.002 * (1 - progress)

model = PPO("MlpPolicy", env, verbose = 1, learning_rate = lr_schedule, tensorboard_log="./ppo_stocktrading_tensorboard/")
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs-PPO/',
                             log_path='./logs-PPO/', eval_freq=5000,
                             deterministic=True, render=False)
model.learn(total_timesteps=500_000, log_interval=5, callback=eval_callback, progress_bar=True)

# Guardar y cargar el modelo
model.save("ppo_trading")
model = PPO.load("ppo_trading")