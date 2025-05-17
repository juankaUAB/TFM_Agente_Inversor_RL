from env_v2 import StockMarketEnv
from ray import tune
from ray.tune.registry import register_env
import ray
import os

ray.init()

def my_env_creator(env_config):
    return StockMarketEnv(env_config)

register_env("StockMarketEnv-v2", my_env_creator)

working_dir = os.path.dirname(os.path.abspath(__file__))

tune.run("PPO", config={
    "env": "StockMarketEnv-v2",
    "lr": 0.0003,
    "buffer_size": 100000,
    "evaluation_interval": 25000,
},
         stop={"num_env_steps_sampled_lifetime": 500000},
         checkpoint_freq=15,
         storage_path="C:/ray_results")

