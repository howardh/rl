import pytest

from experiments.hrl.gridsearch import run_trial

def test_run_trial_steps(tmp_path):
    run_trial(gamma=0.9,alpha=1e-3,eps_b=0.1,eps_t=0,tau=1,directory=tmp_path,env_name='FrozenLake-v0',max_steps=10,epoch=2,min_replay_buffer_size=1,verbose=True)
