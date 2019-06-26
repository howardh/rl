import os
import torch
import gym

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

from experiments.policy_gradient.main import run_trial, run_trial_steps, load_partial_q_model, find_linear_mapping

results_directory = os.path.join(
        utils.get_results_directory(),"ddpg","results")
model_directory = os.path.join(
        utils.get_results_directory(),"ddpg","models")
#run_trial(
#        gamma=0.9,actor_lr=1e-4,critic_lr=1e-3,polyak_rate=1e-3,results_directory=results_directory,model_directory=model_directory,env_name='MountainCarContinuous-v0',epoch=1,max_iters=1,verbose=True)
#run_trial_steps(
#        gamma=0.9,actor_lr=1e-4,critic_lr=1e-3,polyak_rate=1e-3,results_directory=results_directory,model_directory=model_directory,env_name='MountainCarContinuous-v0',epoch=100,max_steps=500,min_replay_buffer_size=0,verbose=True)

env = gym.make('MountainCarContinuous-v0')
model1 = load_partial_q_model('/home/howard/tmp/results/2019-06-25_18-53-20/ddpg/models/model-0.pt', env)
model2 = load_partial_q_model('/home/howard/tmp/results/2019-06-25_18-53-20/ddpg/models/model-1.pt', env)
find_linear_mapping(model1, model2, env)
