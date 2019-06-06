from experiments.msc.maincp import run_all
from experiments.msc.cartpole.exp3 import run_trial
import utils

utils.set_results_directory("/home/howard/tmp/results/test")
run_all(1)
