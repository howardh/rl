import mountaincar 

from mountaincar import exp2
from mountaincar import exp3
from mountaincar import graph

import experiments
import utils

def run_all(proc=20):
    ## SGD
    #experiments.run1(exp2, n=1, proc=proc)
    #for _ in range(100):
    #    experiments.run2(exp2, n=1, m=100, proc=proc)
    #experiments.run3(exp2, n=100, proc=proc)
    #exp2.plot_final_rewards()

    ## LSTD
    #experiments.run1(exp3, n=1, proc=proc)
    #for _ in range(100):
    #    experiments.run2(exp3, n=1, m=100, proc=proc)
    #experiments.run3(exp3, n=100, proc=proc)
    #exp3.plot_final_rewards()

    graph.graph_all()

if __name__ == "__main__":
    utils.set_results_directory("/home/ml/hhuang63/results/final")
    while True:
        experiments.run2(exp3, n=5, m=100, proc=30)
        experiments.plot_best(exp3)
    graph.graph_all()
