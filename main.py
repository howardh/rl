import os
import torch

from experiments.msc.maincp import run_all
#from experiments.msc.cartpole.exp3 import run_trial
from experiments.dqn.main import run_trial_steps as run_trial
from agent.dqn_agent import QNetwork
import utils

#utils.set_results_directory("/home/howard/tmp/results/test")
#run_all(1)
directory = os.path.join(utils.get_results_directory(),"atari")
run_trial(gamma=0.9,alpha=1e-3,eps_b=0.1,eps_t=0,directory=directory,max_steps=1000000,epoch=1000,verbose=True)
#x1 = torch.rand([1,3,210,160])
#x2 = torch.rand([1,3,84,84])
#q = QNetwork()
