from env_v3 import StockMarketEnv
from ray import tune
from ray.tune.registry import register_env
import ray
import os

ray.init()

def my_env_creator(env_config):
    return StockMarketEnv(env_config)

register_env("StockMarketEnv-v3", my_env_creator)

working_dir = os.path.dirname(os.path.abspath(__file__))

tune.run("SAC", config={
    "env": "StockMarketEnv-v3",
    "lr": None,
    "actor_lr": 0.0003,
    "critic_lr": 0.0003,
    "alpha_lr": 0.0003,
    "buffer_size": 100000,
    "evaluation_interval": 25000,
},
         stop={"num_env_steps_sampled_lifetime": 100_000},
         checkpoint_freq=50,
         storage_path="C:/ray_results")

