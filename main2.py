from tqdm import tqdm

import frozenlake2
from frozenlake2 import exp2
from frozenlake2 import exp3
from frozenlake2 import graph

import experiments
import utils

def run_all(proc=20):
    ## SGD
    #experiments.run1(exp2, n=1, proc=proc)
    #for _ in tqdm(range(100)):
    #    experiments.run2(exp2, n=10, m=10, proc=proc)
    #experiments.run3(exp2, n=15, proc=proc)
    #experiments.plot_best(exp2)
    #experiments.plot_final_rewards(exp2)

    ### LSTD
    #experiments.run1(exp3, n=2, proc=proc)
    for _ in tqdm(range(100)):
        experiments.run2(exp3, n=10, m=10, proc=proc)
    #experiments.run3(exp3, n=15, proc=proc)
    experiments.plot_best(exp3)
    experiments.plot_final_rewards(exp3)

    graph.graph_all()

if __name__ == "__main__":
    utils.set_results_directory("/home/ml/hhuang63/results/final")
    #experiments.run3(exp2, n=30, proc=30, params=params)
    #run_all(10)
    #graph.graph_all()
    #for _ in tqdm(range(100)):
    #    experiments.run2(exp2, n=10, m=10, proc=40)
    #utils.skip_new_files(True)
    #s=utils.get_all_series(exp2.get_directory())
    experiments.plot_gridsearch(exp2)
