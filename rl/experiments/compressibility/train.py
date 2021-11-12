import os
from typing import Optional

from experiment import load_checkpoint, make_experiment_runner

from rl.agent.option_critic import OptionCriticAgent
from rl.experiments.training.basic import TrainExperiment

def get_params():
    params = {}

    base_agent_params = {
        'discount_factor': 0.99,
        'behaviour_eps': 0.02,
        'learning_rate': 1e-4,
        'update_frequency': 1,
        'target_update_frequency': 200,
        'polyak_rate': 1,
        'num_options': 1,
        'entropy_reg': 0.01,
    }
    base_env_config = {
        'atari': True,
        'config': {
            'frameskip': 1,
            'mode': 1,
            'difficulty': 0,
            'repeat_action_probability': 0.25,
            #'render_mode': 'human',
        }
    }

    env_name = 'ALE/Pong-v5'
    num_actors = 16
    train_env_keys = list(range(num_actors))

    params['oc-001'] = {
        'agent': {
            'type': OptionCriticAgent,
            'parameters': base_agent_params,
        },
        'env_test': {'env_name': env_name, **base_env_config},
        'env_train': {'env_name': env_name, **base_env_config},
        'train_env_keys': train_env_keys,
        'save_model_frequency': 250_000,
        'verbose': True,
        'test_frequency': None,
    }

    return params

def make_app():
    import typer
    app = typer.Typer()

    @app.command()
    def run(exp_name : str,
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
                        'save_model_frequency': 5,
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=None,
                    max_iterations=15,
                    verbose=True,
            )
            exp_runner.exp.logger.init_wandb({
                'project': f'Compressibility-train-{exp_name} (debug)',
            })
        else:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config=config,
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=250_000,
                    max_iterations=50_000_000,
                    verbose=True,
            )
            exp_runner.exp.logger.init_wandb({
                'project': f'Compressibility-train-{exp_name}',
            })
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
