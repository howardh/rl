import os
import torch

from experiments.lstd.collect_data import run_trial
import utils

directory = os.path.join(
        utils.get_results_directory(),"lstd","results")
print('Saving results to %s' % directory)
data = run_trial(0.9, 1, 0.1, 0.01, sigma=1, lam=0, points_per_step=500, directory=directory, env_name='CartPole-v0',plot_results=True,verbose=True)
