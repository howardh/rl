import os
from typing import Optional
from pathlib import Path

import dill

from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger

from rl.agent.smdp.a2c import A2CAgentRecurrent
from rl.experiments.training.basic import TrainExperiment

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

    num_actors = 16
    train_env_keys = list(range(num_actors))
    env_name = 'ALE/Pong-v5'
    env_config = {
        'frameskip': 1,
        'mode': 0,
        'difficulty': 0,
        'repeat_action_probability': 0.25,
    }

    params.add('exp-001', {
        'agent': {
            'type': A2CAgentRecurrent,
            'parameters': {
                'training_env_keys': train_env_keys,
                'max_rollout_length': 5,
                'hidden_reset_min_prob': 0,
                'hidden_reset_max_prob': 0,
            },
        },
        'env_test': {'env_name': env_name, 'atari': True, 'config': env_config, 'frame_stack': 1},
        'env_train': {'env_name': env_name, 'atari': True, 'config': env_config, 'frame_stack': 1},
        'train_env_keys': train_env_keys,
        'test_frequency': None,
        'save_model_frequency': 250_000,
        'verbose': True,
    })

    params.add_change('exp-002', {
        'agent': {
            'parameters': {
                'max_rollout_length': 32
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
                    max_iterations=50_000_000,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
            #exp_runner.exp.logger.init_wandb({
            #    'project': 'A2C-recurrent-%s' % env_name.replace('/','_')
            #})
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
        output_directory = x['exp']['output_directory']
        plot_directory = os.path.join(output_directory,'plots')
        os.makedirs(plot_directory,exist_ok=True)

        filename = os.path.abspath(os.path.join(plot_directory,'plot-so-val.png'))
        eplt.plot(logger,
                filename=filename,
                curves=[{
                    'key': 'agent_state_option_value',
                    'smooth_fn': EMASmoothing(0.9),
                }],
                min_points=2,
                xlabel='Steps',
                ylabel='State Option Value',
                #aggregate='mean',
        )
        print(f'Plot saved to {filename}')

        filename = os.path.abspath(os.path.join(plot_directory,'plot-rewards.png'))
        eplt.plot(logger,
                filename=filename,
                curves=[{
                    'key': 'train_reward_by_episode',
                    'smooth_fn': EMASmoothing(0.9),
                }],
                min_points=2,
                xlabel='Steps',
                ylabel='Rewards',
                aggregate='mean',
                show_unaggregated=False,
        )
        print(f'Plot saved to {filename}')

        filename = os.path.abspath(os.path.join(plot_directory,'plot-option-choice.png'))
        eplt.stacked_area_plot(logger,
                filename=filename,
                key='agent_option_choice_count',
                title='Option choice relative frequencies',
        )
        print(f'Plot saved to {filename}')

        for k in ['agent_total_loss','agent_termination_loss','agent_policy_loss','agent_entropy_loss','agent_critic_loss','agent_training_option_duration']:
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
