import pytest
import os

def test_run_trial_steps(tmp_path):
    from experiments.hrl.gridsearch import run_trial
    run_trial(gamma=0.9,alpha=1e-3,eps_b=0.1,eps_t=0,tau=1,directory=tmp_path,env_name='FrozenLake-v0',max_steps=10,epoch=2,min_replay_buffer_size=1,verbose=True)
    
def test_gs_h_run_trial_steps(tmp_path):
    from experiments.hrl.gridsearch_hierarchical import run_trial
    from experiments.hrl.gridsearch_hierarchical import plot
    results_directory = os.path.join(tmp_path,'results')
    plot_directory = os.path.join(tmp_path,'plot')
    run_trial(gamma=0.9,alpha=1e-3,eps_b=0.1,eps_t=0,tau=1,net_structure=(),num_options=2,directory=results_directory,env_name='FrozenLake-v0',batch_size=3,max_steps=10,epoch=2,min_replay_buffer_size=2,verbose=True)
    plot(results_directory,plot_directory)
