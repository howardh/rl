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

def test_transfer(tmp_path):
    from experiments.hrl.transfer_nrooms import run_trial
    results_directory = os.path.join(tmp_path,'results')
    plot_directory = os.path.join(tmp_path,'plot')

    params = {
        'agent_name': 'ActorCritic',
        'gamma': 0.9,
        'controller_learning_rate': 1e-3,
        'subpolicy_learning_rate': 1e-3,
        'q_net_learning_rate': 1e-3,
        'eps_b': 0,
        'polyak_rate': 0.001,
        'batch_size': 2,
        'min_replay_buffer_size': 5,
        'steps_per_task': 3,
        'total_steps': 12,
        'epoch': 3,
        'test_iters': 1,
        'verbose': True,
        'num_options': 2,
        'cnet_n_layers': 0,
        'cnet_layer_size': 0,
        'snet_n_layers': 0,
        'snet_layer_size': 0,
        'qnet_n_layers': 0,
        'qnet_layer_size': 1,
        'directory': results_directory
    }
    run_trial(**params)
