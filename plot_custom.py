import frozenlake
import frozenlake.exp2
import frozenlake.exp3
import frozenlake2.exp2
import frozenlake2.exp3
import experiments
import utils

if __name__=='__main__':
    utils.set_results_directory("/home/ml/hhuang63/results/final")
    experiments.plot_custom_best_mean(
            [frozenlake.exp2, frozenlake2.exp2, frozenlake.exp3, frozenlake2.exp3],
            ['SGD accumulating', 'SGD replacing', 'LSTD accumulating', 'LSTD replacing'])
