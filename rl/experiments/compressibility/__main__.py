import os
import re

from simple_slurm import Slurm

import rl.utils
from rl.experiments.compressibility.train import make_app as make_app_train
from rl.experiments.compressibility.distill import make_app as make_app_distill
from rl.experiments.compressibility.transfer import make_app as make_app_transfer

EXPERIMENT_SET = 0

if EXPERIMENT_SET == 0:
    # TODO: Description of experiment
    PROJECT_ROOT = './rl'
    EXPERIMENT_GROUP_NAME = 'compressibility-oc-Pong-v5'
    RESULTS_ROOT_DIRECTORY = rl.utils.get_results_root_directory()
    EXPERIMENT_GROUP_DIRECTORY = os.path.join(RESULTS_ROOT_DIRECTORY, EXPERIMENT_GROUP_NAME)
    TRAIN_EXP_NAME = 'oc-001'

##################################################
# Training
##################################################

def train(debug=False):
    _,commands = make_app_train()
    commands['run'](
            TRAIN_EXP_NAME,
            results_directory = os.path.join(
                EXPERIMENT_GROUP_DIRECTORY, 'train'),
            debug=debug,
    )

def find_all_trained_models():
    models_directory = os.path.join(
            EXPERIMENT_GROUP_DIRECTORY, 'train', 'output', 'models')
    files = os.listdir(models_directory)
    pattern = r'(\d+)\.pkl'
    for fn in files:
        match = re.search(pattern,fn)
        if match is None:
            continue
        yield os.path.join(models_directory,fn)

##################################################
# Transfer
##################################################

def transfer(debug=False):
    _,commands = make_app_transfer()
    # Run a transfer experiment on the model at various points during training
    for model_filename in find_all_trained_models():
        steps = os.path.split(model_filename)[-1][:-4]
        commands['run'](
                TRAIN_EXP_NAME,
                model_filename,
                results_directory = os.path.join(
                    EXPERIMENT_GROUP_DIRECTORY, f'transfer-{steps}'),
                debug=debug,
        )

##################################################
# Distillation
##################################################

def distillation(debug=False):
    _,commands = make_app_distill()
    # Run a distillation experiment on the model at various points during training
    for model_filename in find_all_trained_models():
        steps = os.path.split(model_filename)[-1][:-4]
        commands['run'](
                model_filename,
                results_directory = os.path.join(
                    EXPERIMENT_GROUP_DIRECTORY, f'distill-{steps}'),
                debug=debug,
        )

##################################################
# App
##################################################

def slurm_queue_train(debug=False):
    if debug:
        slurm = Slurm(
                #array=range(3),
                array='1-3%1',
                cpus_per_task=1,
                output='/network/scratch/h/huanghow/slurm/%A_%a.out',
                time='0:10:00',
        )
    else:
        slurm = Slurm(
                #array=range(10),
                array='1-10%1',
                cpus_per_task=1,
                output='/network/scratch/h/huanghow/slurm/%A_%a.out',
                time='24:00:00',
        )
    command = './sbatch.sh {script} run {exp_name} --results-directory {results_directory} {debug} --slurm'
    command = command.format(
            script = os.path.join(PROJECT_ROOT, 'rl/experiments/compressibility/train.py'),
            results_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY, 'train'),
            debug = '--debug' if debug else '',
            exp_name = TRAIN_EXP_NAME,
    )
    print(command)
    job_id = slurm.sbatch(command)
    return job_id

def slurm_queue_distill(debug=False):
    if debug:
        slurm = Slurm(
                #array=range(3),
                array='1-3%1',
                cpus_per_task=1,
                output='/network/scratch/h/huanghow/slurm/%A_%a.out',
                time='0:10:00',
        )
    else:
        slurm = Slurm(
                #array=range(10),
                array='1-10%1',
                cpus_per_task=1,
                output='/network/scratch/h/huanghow/slurm/%A_%a.out',
                time='24:00:00',
        )
    command = './sbatch.sh {script} run {filename} --results-directory {results_directory} {debug} --slurm'
    job_ids = []
    for model_filename in find_all_trained_models():
        steps = os.path.split(model_filename)[-1][:-4]
        formatted_command = command.format(
                script = os.path.join(PROJECT_ROOT, 'rl/experiments/compressibility/distill.py'),
                filename = model_filename,
                results_directory = os.path.join(
                    EXPERIMENT_GROUP_DIRECTORY, f'distill-{steps}'),
                debug = '--debug' if debug else '',
        )
        print(formatted_command)
        job_id = slurm.sbatch(formatted_command)
        job_ids.append(job_id)
        break
    return job_ids

def slurm_queue_transfer(debug=False):
    if debug:
        slurm = Slurm(
                #array=range(3),
                array='1-3%1',
                cpus_per_task=1,
                output='/network/scratch/h/huanghow/slurm/%A_%a.out',
                time='0:10:00',
        )
    else:
        slurm = Slurm(
                #array=range(10),
                array='1-10%1',
                cpus_per_task=1,
                output='/network/scratch/h/huanghow/slurm/%A_%a.out',
                time='24:00:00',
        )
    command = './sbatch.sh {script} run {exp_name} {filename} --results-directory {results_directory} {debug} --slurm'
    job_ids = []
    for model_filename in find_all_trained_models():
        steps = os.path.split(model_filename)[-1][:-4]
        formatted_command = command.format(
                script = os.path.join(PROJECT_ROOT, 'rl/experiments/compressibility/transfer.py'),
                exp_name = 'oc-001',
                filename = model_filename,
                results_directory = os.path.join(
                    EXPERIMENT_GROUP_DIRECTORY, f'transfer-{steps}'),
                debug = '--debug' if debug else '',
        )
        print(formatted_command)
        job_id = slurm.sbatch(formatted_command)
        job_ids.append(job_id)
        break
    return job_ids

def run_slurm(debug=False):
    """
    Exp 1: train TRAIN_EXP_NAME
    Exp 2.1: distill
    Exp 2.2: transfer
    """

    print('Initializing experiments')
    # distill and transfer need to be manually queued up after training is done.

    slurm_queue_train(debug)
    #slurm_queue_distill(debug)
    #slurm_queue_transfer(debug)

def run_local(debug=False):
    print('-'*50)
    print(' Train')
    print('-'*50)
    train(debug=debug)

    print('-'*50)
    print(' Transfer')
    print('-'*50)
    transfer(debug=debug)

    print('-'*50)
    print(' Distillation')
    print('-'*50)
    distillation(debug=debug)

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
        pass

    commands = {
            'slurm': slurm,
            'local': local,
            'plot': plot,
    }

    return app,commands

if __name__=='__main__':
    app,_ = make_app()
    app()
