import cartpole
import cartpole.features
import cartpole.utils

from cartpole import exp1
from cartpole import exp3
from cartpole import exp4
from cartpole import exp5
from cartpole import graph
import experiments
import utils

def run_all(proc=20):
    # SGD
    experiments.run1(exp1, n=1, proc=proc)
    for _ in tqdm(range(10)):
        experiments.run2(exp1, n=10, m=100, proc=proc)
    experiments.run3(exp1, n=15, proc=proc)
    experiments.plot_best(exp1)
    experiments.plot_final_rewards(exp1)

    ## LSTD
    #experiments.run1(exp3, n=1, proc=proc)
    #for _ in range(100):
    #    experiments.run2(exp3, n=1, m=100, proc=proc)
    #experiments.run3(exp3, n=100, proc=proc)
    #experiments.plot_best(exp3)
    #experiments.plot_final_rewards(exp3)

    #experiments.run3(exp4, n=100, proc=proc)
    #experiments.plot_best(exp4)

    #experiments.run3(exp5, n=100, proc=proc)
    #experiments.plot_best(exp5)

    #graph.graph_sgd()
    #graph.graph_lstd()
    #graph.graph_sarsa_tb()
    graph.graph_all()

if __name__ == "__main__":
    utils.set_results_directory("/home/ml/hhuang63/results/final")
    run_all(20)
