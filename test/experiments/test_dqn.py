import pytest

from experiments.dqn.main import run_trial_steps as run_trial, run_trial_steps

@pytest.mark.skip(reason="Unimportant for now. Fix later.")
def test_run_trial(tmp_path):
    run_trial(gamma=0.9,alpha=1e-3,eps_b=0.1,eps_t=0,tau=1,directory=tmp_path,env_name='Pong-v0',max_steps=10,epoch=2,min_replay_buffer_size=1,verbose=True)

@pytest.mark.skip(reason="Unimportant for now. Fix later.")
def test_run_trial_steps(tmp_path):
    run_trial_steps(gamma=0.9,alpha=1e-3,eps_b=0.1,eps_t=0,tau=1,directory=tmp_path,env_name='Pong-v0',max_steps=10,epoch=2,min_replay_buffer_size=1,verbose=True)
