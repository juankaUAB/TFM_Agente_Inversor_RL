from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time

class ProfitLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_profits = []
        self.start_time = None
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Escribir el tiempo de entrenamiento
        elapsed_min = (time.time() - self.start_time) / 60.0
        self.logger.record("training/elapsed_minutes", elapsed_min)
        
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]
        truncated = info.get("truncated", False)

        if (done or truncated) and "profit" in info:
            episode_profit = info["profit"]
            self.episode_profits.append(episode_profit)

            avg_profit = np.mean(self.episode_profits[-100:])

            self.logger.record("custom/profit_last_episode", episode_profit)
            self.logger.record("custom/avg_profit", avg_profit)

        return True