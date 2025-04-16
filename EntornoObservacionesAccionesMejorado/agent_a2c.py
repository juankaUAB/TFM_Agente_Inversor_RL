from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from env_v3 import StockMarketEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from ProfitCallback import ProfitLoggingCallback

# Supón que tienes un entorno compatible con Gym: `TradingEnv`
env = DummyVecEnv([lambda: StockMarketEnv()])
eval_env = DummyVecEnv([lambda: StockMarketEnv()])

def lr_schedule(progress):
    # Reduce el learning rate según el progreso
    return 0.002 * (1 - progress)

model = A2C("MlpPolicy", env, verbose = 1, learning_rate = 0.0003, tensorboard_log="./a2c_stocktrading_tensorboard/", n_steps=2048, ent_coef=0.001)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs-A2C/',
                             log_path='./logs-A2C/', eval_freq=5000,
                             deterministic=True, render=False)
profit_callback = ProfitLoggingCallback()
callback = CallbackList([profit_callback, eval_callback])
model.learn(total_timesteps=1_000_000, log_interval=5, callback=callback, progress_bar=True, tb_log_name = "A2C_1M_lrStable_entCoefModif")

# Guardar y cargar el modelo
model.save("A2C_trading")
model = A2C.load("A2C_trading")