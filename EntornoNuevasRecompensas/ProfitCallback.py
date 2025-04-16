from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ProfitLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_profits = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]
        truncated = info.get("truncated", False)

        if (done or truncated) and "profit" in info:
            episode_profit = info["profit"]
            self.episode_profits.append(episode_profit)
            self.logger.record("custom/profit_last_episode", episode_profit)

        return True