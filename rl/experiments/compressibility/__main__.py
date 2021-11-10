import os
import re

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
    TRAIN_ID_RANGE = range(1,3)
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
