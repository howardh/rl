from collections import defaultdict
import os
import re

import numpy as np
import dill
from simple_slurm import Slurm
from matplotlib import pyplot as plt

import rl.utils
from rl.experiments.hrl2.disjoint import make_app as make_app_disjoint
from rl.experiments.hrl2.dropout import make_app as make_app_dropout

PROJECT_ROOT = './rl'
EXPERIMENT_GROUP_NAME = 'disjoint-dqn-sac-hopper'
RESULTS_ROOT_DIRECTORY = rl.utils.get_results_root_directory()
EXPERIMENT_GROUP_DIRECTORY = os.path.join(RESULTS_ROOT_DIRECTORY, EXPERIMENT_GROUP_NAME)
TRAIN_ID_RANGE = range(1,3)

def run_slurm(debug=False):
    """
    Exp 1: hrl-001
    Exp 2.1: distil
    Exp 2.2: dropout
    """
    print('Initializing experiments')

    ##################################################
    # Train

    slurm = Slurm(
            array=TRAIN_ID_RANGE,
            cpus_per_task=1,
            output='/miniscratch/huanghow/slurm/%A_%a.out',
            time='24:00:00',
    )
    #command = './sbatch37.sh {script} run hrl-001 --experiment-group {group}'
    command = './sbatch37.sh {script} run hrl-001 --results-directory {results_directory} {debug}'
    command = command.format(
            script = os.path.join(PROJECT_ROOT, 'rl/experiments/hrl2/disjoint.py'),
            group = EXPERIMENT_GROUP_NAME,
            results_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY, 'train-$SLURM_ARRAY_TASK_ID'), # Note: The environment variable substitution is done by bash when this command is run
            debug = '--debug' if debug else ''
    )
    print(command)
    train_job_id = slurm.sbatch(command)

    ##################################################
    # Dropout

    # For each of these training experiments, we want a dropout experiment 
    train_dir = os.path.join(EXPERIMENT_GROUP_DIRECTORY, 'train-$SLURM_ARRAY_TASK_ID')
    command = './sbatch37.sh {script} run {filename} --results-directory {results_directory}'
    command = command.format(
            filename = os.path.join(train_dir,'output','deploy_state-best.pkl'),
            script = os.path.join(PROJECT_ROOT, 'rl/experiments/hrl2/dropout.py'),
            results_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY, 'dropout-$SLURM_ARRAY_TASK_ID'),
    )
    slurm = Slurm(
            array=TRAIN_ID_RANGE,
            cpus_per_task=1,
            output='/miniscratch/huanghow/slurm/%A_%a.out',
            time='1:00:00',
            dependency=dict(aftercorr=train_job_id), # 1-to-1 mapping of dropout job to train job
    )
    dropout_job_id = slurm.sbatch(command)
    # Note: Check that this works with `scontrol show jobid -dd <job_id>`

    ##################################################
    # Plot

    command = './sbatch37.sh {script} plot'
    command = command.format(
            script = os.path.join(PROJECT_ROOT, 'rl/experiments/hrl2/__main__.py'),
    )
    slurm = Slurm(
            cpus_per_task=1,
            output='/miniscratch/huanghow/slurm/%A_%a.out',
            time='1:00:00',
            dependency=dict(afterok=dropout_job_id),
    )
    slurm.sbatch(command)

def run_local(debug=False):
    # Run training with an experiment group
    train(n=5, debug=debug)
    # Run a dropout experiment on completed training runs
    dropout(debug=debug)
    # Plot the results from all dropout experiments in this group
    plot_dropout()

def train(n=1, debug=False):
    _,commands = make_app_disjoint()
    for i in range(n):
        commands['run'](
                'hrl-001',
                #trial_id = 'train-%d' % i,
                results_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY, 'train-%d'%i),
                debug=debug,
        )

def find_all_train_directories(root_directory):
    dirs = os.listdir(root_directory)
    for d in dirs:
        if d.startswith('train-'):
            yield os.path.join(root_directory,d)

def dropout(debug=False):
    _,commands = make_app_dropout()
    # Run a dropout experiment on all found training runs
    pattern = r'/train-(\d+)'
    for train_dir in find_all_train_directories(EXPERIMENT_GROUP_DIRECTORY):
        match = re.search(pattern,train_dir)
        train_id = match.group(1)
        commands['run'](
                os.path.join(train_dir,'output','deploy_state.pkl'),
                results_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY, 'dropout-%s'%train_id),
                debug=debug,
        )

def find_all_dropout_directories(root_directory):
    dirs = os.listdir(root_directory)
    for d in dirs:
        if d.startswith('dropout-'):
            yield os.path.join(root_directory,d)

def plot_train():
    pass # TODO: Plot performance over time with smoothing

def plot_dropout():
    rewards = defaultdict(lambda: [])
    for dropout_dir in find_all_dropout_directories(EXPERIMENT_GROUP_DIRECTORY):
        checkpoint_path = os.path.join(dropout_dir,'checkpoint.pkl')
        with open(checkpoint_path, 'rb') as f:
            checkpoint = dill.load(f)
        results = checkpoint['exp']['results']
        for k,v in results.items():
            total_reward_mean = [x['total_reward'] for x in v]
            rewards[k].append(np.mean(total_reward_mean))
    labels = sorted(rewards.keys())
    values = [rewards[l] for l in labels]
    plt.violinplot(
            values,
            vert = False,
            showmeans = True,
    )
    plt.yticks(ticks=list(range(1,len(labels)+1)),labels=labels)
    plt.ylabel('Dropout Probability')
    plt.xlabel('Total Reward')
    plt.title('Dropout on Hopper-v3')
    plt.grid()

    plot_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY, 'GROUP')
    os.makedirs(plot_directory, exist_ok=True)
    plot_filename = os.path.join(plot_directory, 'dropout.png')
    plt.savefig(plot_filename)
    print('Plot saved at %s' % plot_filename)

def make_app():
    import typer
    app = typer.Typer()

    @app.command()
    def slurm(debug : bool = typer.Option(False, '--debug')):
        run_slurm(debug=debug)

    @app.command()
    def local(debug : bool = typer.Option(False, '--debug')):
        run_local(debug=debug)

    @app.command()
    def plot():
        plot_dropout()

    commands = {
            'slurm': slurm,
            'local': local,
            'plot': plot,
    }

    return app,commands

if __name__=='__main__':
    app,_ = make_app()
    app()
