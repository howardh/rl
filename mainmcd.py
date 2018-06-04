import mountaincardiscrete

from mountaincardiscrete import exp2
from mountaincardiscrete import exp3
from mountaincardiscrete import graph

import experiments
import utils

def run_all(proc=20):
    # SGD
    experiments.run1(exp2, n=1, proc=proc)
    #for _ in range(100):
    #    experiments.run2(exp2, n=5, m=20, proc=proc)
    #experiments.run3(exp2, n=100, proc=proc)
    #experiments.plot_best(exp2)
    #experiments.plot_final_rewards(exp2)

    ## LSTD
    #experiments.run1(exp3, n=1, proc=proc)
    #for _ in range(100):
    #    experiments.run2(exp3, n=1, m=100, proc=proc)
    #experiments.run3(exp3, n=100, proc=proc)
    #experiments.plot_best(exp3)
    #experiments.plot_final_rewards(exp3)

    graph.graph_all()

if __name__ == "__main__":
    utils.set_results_directory("/home/ml/hhuang63/results/final")
    #utils.skip_new_files(True)
    #utils.set_results_directory("/home/hhuang63/scratch/results/final")
    run_all(40)
    #experiments.run1(exp3, n=1, proc=1)
    #experiments.run2(exp3, n=10, m=75, proc=1)
    #graph.graph_all()
    #for _ in range(100):
    #    experiments.run2(exp3, n=5, m=75, proc=30)
    #experiments.plot_gridsearch(exp2)
    #experiments.plot_gridsearch(exp3)
    #utils.get_all_series(exp2.get_directory())
    #experiments.plot_best_trials(exp2,3)
    #experiments.plot_best_trials(exp3,3)
