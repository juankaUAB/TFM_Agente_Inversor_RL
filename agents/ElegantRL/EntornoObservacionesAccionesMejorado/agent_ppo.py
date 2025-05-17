from elegantrl.train.config import Config
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train.run import train_agent
from env_v3 import StockMarketEnv
import os

env_args = {
    'env_name': '2ksteps',
    'state_dim': 8*6,
    'action_dim': 1,
    'if_discrete': False,
}

args = Config(AgentPPO, StockMarketEnv, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
args.break_step = int(2e5)  # break training if 'total_step > break_step'
args.learning_rate = 0.0003
args.eval_per_step = 25000
args.save_gap = 50000  # Frecuencia de guardado (en steps)
args.buffer_size = 100000 # Ajustamos tamaño del buffer de experiencias para los algoritmos off-policy
args.buffer_init_size = 10000 # Empezar a acumular experiencias después de 10k pasos
args.repeat_times = 16.0 # repeatedly update network using ReplayBuffer to keep critic's loss small
args.gpu_id = -1
args.num_threads = os.cpu_count()
train_agent(args=args)


