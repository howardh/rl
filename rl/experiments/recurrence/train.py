import os
from pathlib import Path
from typing import Optional

import dill
import experiment.logger
from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger

from rl.agent.smdp.a2c import A2CAgentRecurrentVec
from rl.experiments.training.vectorized import TrainExperiment

def merge(source, destination):
    """
    (Source: https://stackoverflow.com/a/20666342/382388)

    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination

class ExperimentConfigs(dict):
    def __init__(self):
        self._last_key = None
    def add(self, key, config, inherit=None):
        if key in self:
            raise Exception(f'Key {key} already exists.')
        if inherit is None:
            self[key] = config
        else:
            self[key] = merge(self[inherit],config)
        self._last_key = key
    def add_change(self, key, config):
        self.add(key, config, inherit=self._last_key)

def get_params():
    params = ExperimentConfigs()

    num_envs = 16
    env_name = 'Pong-v5'
    #env_config = {
    #    'frameskip': 1,
    #    'mode': 0,
    #    'difficulty': 0,
    #    'repeat_action_probability': 0.25,
    #}
    env_config = {
            'env_name': env_name,
            'atari': True,
            'atari_config': {
                'num_envs': num_envs
            }
    }

    params.add('exp-001', {
        'agent': {
            'type': A2CAgentRecurrentVec,
            'parameters': {
                'target_update_frequency': 32_000,
                'num_train_envs': num_envs,
                'num_test_envs': 1,
                'max_rollout_length': 128,
                'hidden_reset_min_prob': 0,
                'hidden_reset_max_prob': 0,
            },
        },
        'env_test': env_config,
        'env_train': env_config,
        'test_frequency': None,
        'save_model_frequency': None,
        'verbose': True,
    })

    params.add_change('exp-002', {
        'agent': {
            'parameters': {
                #'max_rollout_length': 32
            },
        },
    })

    return params

def make_app():
    import typer
    app = typer.Typer()

    @app.command()
    def run(exp_name : str,
            trial_id : Optional[str] = None,
            results_directory : Optional[str] = None,
            slurm : bool = typer.Option(False, '--slurm'),
            wandb : bool = typer.Option(False, '--wandb'),
            debug : bool = typer.Option(False, '--debug')):
        config = get_params()[exp_name]
        if debug:
            raise NotImplementedError()
        else:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config=config,
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=250_000,
                    max_iterations=5_000_000,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
            if wandb:
                exp_runner.exp.logger.init_wandb({
                    'project': f'A2C-recurrent-{exp_name}'
                })
        exp_runner.run()
        exp_runner.exp.logger.finish_wandb()

    @app.command()
    def checkpoint(filename):
        exp = load_checkpoint(TrainExperiment, filename)
        exp.run()

    @app.command()
    def plot(result_directory : Path):
        import experiment.plotter as eplt
        from experiment.plotter import EMASmoothing

        checkpoint_filename = os.path.join(result_directory,'checkpoint.pkl')
        with open(checkpoint_filename,'rb') as f:
            x = dill.load(f)
        logger = Logger()
        logger.load_state_dict(x['exp']['logger'])
        if isinstance(logger.data, experiment.logger.FileBackedList):
            logger.data.iterate_past_end = True
        output_directory = x['exp']['output_directory']
        plot_directory = os.path.join(output_directory,'plots')
        os.makedirs(plot_directory,exist_ok=True)

        for k in ['agent_train_state_value_target_net', 'agent_train_state_value', 'train_reward', 'reward']:
            try:
                filename = os.path.abspath(os.path.join(plot_directory,f'plot-{k}.png'))
                eplt.plot(logger,
                        filename=filename,
                        curves=[{
                            'key': k,
                            'smooth_fn': EMASmoothing(0.9),
                        }],
                        min_points=2,
                        xlabel='Steps',
                        ylabel=k,
                        aggregate='mean',
                        show_unaggregated=False,
                )
                print(f'Plot saved to {filename}')
            except KeyError:
                print(f'Could not plot {k}. Key not found.')

    commands = {
            'run': run,
            'checkpoint': checkpoint,
            'plot': plot,
    }

    return app, commands

def run():
    app,_ = make_app()
    app()

if __name__ == "__main__":
    run()
