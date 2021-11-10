import os
from typing import Optional

import dill
from experiment import load_checkpoint, make_experiment_runner

from rl.experiments.training.basic import TrainExperiment
from rl.experiments.compressibility.train import get_params

def make_app():
    import typer
    app = typer.Typer()

    @app.command()
    def run(exp_name : str,
            model_filename : str,
            trial_id : Optional[str] = None,
            results_directory : Optional[str] = None,
            debug : bool = typer.Option(False, '--debug')):

        if trial_id is None:
            slurm_job_id = os.environ.get('SLURM_JOB_ID')
            #slurm_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
            slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
            trial_id = slurm_job_id
            if slurm_array_task_id is not None:
                trial_id = '%s_%s' % (slurm_job_id, slurm_array_task_id)

        config = get_params()[exp_name]
        if debug:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config={
                        **config,
                        'test_frequency': 50,
                        'num_test_episodes': 2,
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=None,
                    max_iterations=100,
                    verbose=True,
            )
            exp_runner.exp.logger.init_wandb({
                'project': f'Compressibility-transfer-{exp_name} (debug)',
            })
        else:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config={
                        **config,
                        'test_frequency': 50_000,
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=250_000,
                    max_iterations=1_000_000,
                    verbose=True,
            )
            exp_runner.exp.logger.init_wandb({
                'project': f'Compressibility-transfer-{exp_name}',
            })
        with open(model_filename,'rb') as f:
            exp_runner.exp.agent.load_state_dict_deploy(dill.load(f))
        exp_runner.run()
        exp_runner.exp.logger.finish_wandb()

    @app.command()
    def checkpoint(filename):
        exp = load_checkpoint(TrainExperiment, filename)
        exp.run()

    @app.command()
    def video(state_filename : str,
            output : str = 'output.avi'):
        state_filename = state_filename
        output = output
        raise NotImplementedError('Not implemented')

    commands = {
            'run': run,
            'checkpoint': checkpoint,
            'video': video
    }

    return app, commands

def run():
    app,_ = make_app()
    app()

if __name__ == "__main__":
    run()

