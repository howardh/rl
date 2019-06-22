import os
import torch

from experiments.msc.maincp import run_all
#from experiments.msc.cartpole.exp3 import run_trial
from experiments.dqn.main import run_trial_steps as run_trial
from agent.dqn_agent import QNetwork
import utils

##utils.set_results_directory("/home/howard/tmp/results/test")
##run_all(1)
#directory = os.path.join(utils.get_results_directory(),"atari")
#run_trial(gamma=0.9,alpha=1e-3,eps_b=0.1,eps_t=0,tau=1,directory=directory,env_name='Pong-v0',max_steps=1000000,epoch=1000,min_replay_buffer_size=10000,verbose=True)
##x1 = torch.rand([1,3,210,160])
##x2 = torch.rand([1,3,84,84])
##q = QNetwork()

from experiments.policy_gradient.main import run_trial, run_trial_steps
directory = os.path.join(utils.get_results_directory(),"ddpg")
run_trial(gamma=0.9,actor_lr=1e-4,critic_lr=1e-3,polyak_rate=1e-3,directory=directory,env_name='MountainCarContinuous-v0',epoch=1,verbose=True)
#run_trial_steps(gamma=0.9,actor_lr=1e-4,critic_lr=1e-3,polyak_rate=1e-3,directory=directory,env_name='RoboschoolHalfCheetah-v1',epoch=1000,min_replay_buffer_size=0,verbose=True)
