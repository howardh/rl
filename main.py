from experiments.msc.maincp import run_all
#from experiments.msc.cartpole.exp3 import run_trial
from experiments.dqn.main import run_trial_steps as run_trial
from agent.dqn_agent import QNetwork
import utils

import torch

#utils.set_results_directory("/home/howard/tmp/results/test")
#run_all(1)
run_trial(0.9,0.1,0.1,0.1,'/home/howard/tmp/results/test',max_steps=1000000,epoch=1000,verbose=True)
#x1 = torch.rand([1,3,210,160])
#x2 = torch.rand([1,3,84,84])
#q = QNetwork()
