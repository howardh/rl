from tqdm import tqdm

import frozenlake2
from frozenlake2 import exp2
from frozenlake2 import exp3
from frozenlake2 import exp4
from frozenlake2 import exp5
from frozenlake2 import exp3decay
from frozenlake2 import exp5decay
from frozenlake2 import graph

import experiments
import utils

def run_all(proc=20):
    ## SGD
    #experiments.run1(exp2, n=1, proc=proc)
    #for _ in tqdm(range(1)):
    #    experiments.run2(exp2, n=10, m=10, proc=proc)
    experiments.run3(exp2, n=40, proc=proc)
    #experiments.plot_best(exp2)
    #experiments.plot_final_rewards(exp2)

    ### LSTD
    #experiments.run1(exp3, n=2, proc=proc)
    #for _ in tqdm(range(1)):
    #    experiments.run2(exp3, n=10, m=10, proc=proc)
    experiments.run3(exp3, n=40, proc=proc)
    #experiments.plot_best(exp3)
    #experiments.plot_final_rewards(exp3)

    #graph.graph_all()

if __name__ == "__main__":
    utils.set_results_directory("/home/ml/hhuang63/results/final")
    #experiments.run3(exp2, n=30, proc=30, params=params)
    #while True:
    #    run_all(40)
    #for _ in tqdm(range(100)):
    #    experiments.run2(exp2, n=10, m=10, proc=40)
    #utils.skip_new_files(True)
    #while True:
    #    experiments.run3(exp3, n=20, proc=1,
    #            score_functions=[experiments.get_mean_rewards_first100])
    #    graph.graph_lstd()
    #    graph.graph_all()
    #while True:
    #    experiments.run3(exp4, n=20, proc=3)
    #    experiments.run2(exp2, n=10, m=1, proc=40)
    #    graph.graph_sgd()
    #experiments.plot_gridsearch(exp3decay)
    #experiments.plot_best(exp3decay)
    #graph.graph_sgd()
    #while True:
    #    experiments.run3(exp5, n=5, proc=10)
    #    #graph.graph_lstd()
    #experiments.run3(exp5, n=300, proc=30)
    #graph.graph_lstd()
    #graph.graph_sgd()
    #graph.graph_all()
    #s=utils.get_all_series(exp2.get_directory())
    #experiments.plot_gridsearch(exp3)
    #experiments.plot_best_trials(exp2,3)
    #experiments.plot_best_trials(exp3,3)
    #experiments.plot_best(exp5)
    #experiments.plot_gridsearch(exp3decay)
    #experiments.plot_best(exp3decay,
    #        score_functions=[experiments.get_mean_rewards])
    #experiments.plot_best(exp3decay,
    #        score_functions=[experiments.get_mean_rewards_first_n(10)])
    while True:
        experiments.run1(exp3decay, n=1, proc=10)
        #experiments.run2(exp3decay, n=1, m=10, proc=3,
        #        score_functions=[experiments.get_mean_rewards_first_n(10)])
        #experiments.run3(exp3decay, n=3, proc=10,
        #        score_functions=[experiments.get_mean_rewards_first_n(10)])

        #experiments.plot_best(exp3decay,
        #        score_functions=[experiments.get_mean_rewards])
        #experiments.plot_best(exp3decay,
        #        score_functions=[experiments.get_mean_rewards_first_n(10)])
    #while True:
    #    experiments.run3(exp5decay, n=5, proc=3)
    #experiments.plot_best(exp5decay)
