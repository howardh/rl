import os
from pathlib import Path
from typing import Optional
from pprint import pprint

import gym.spaces
import dill
import experiment.logger
from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger

from rl.agent.smdp.a2c import A2CAgentRecurrentVec, PolicyValueNetworkRecurrent
from rl.experiments.training.vectorized import TrainExperiment
from rl.experiments.recurrence.train import ExperimentConfigs
from rl.experiments.pathways.models import ConvPolicy


class AttnRecAgent(A2CAgentRecurrentVec):
    def __init__(self, recurrence_type='RecurrentAttention', num_recurrence_blocks=1, **kwargs):
        self._recurrence_type = recurrence_type
        self._num_recurrence_blocks = num_recurrence_blocks
        super().__init__(**kwargs)
    def _init_default_net(self, observation_space, action_space, device) -> PolicyValueNetworkRecurrent:
        if isinstance(observation_space, gym.spaces.Box):
            assert observation_space.shape is not None
            if len(observation_space.shape) == 1: # Mujoco
                raise Exception('Unsupported observation space or action space.')
            if len(observation_space.shape) == 3: # Atari
                return ConvPolicy(
                        num_actions=action_space.n,
                        input_size=512,
                        key_size=512,
                        value_size=512,
                        num_heads=8,
                        ff_size = 1024,
                        in_channels=observation_space.shape[0],
                        recurrence_type=self._recurrence_type,
                        num_blocks=self._num_recurrence_blocks,
                ).to(device)
        raise Exception('Unsupported observation space or action space.')


def get_params():
    from rl.experiments.pathways.train import AttnRecAgent as Agent # Need to import for pickling purposes

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
                'num_envs': num_envs,
                'stack_num': 1,
                'repeat_action_probability': 0.25,
            }
    }

    params.add('exp-001', {
        'agent': {
            'type': Agent,
            'parameters': {
                'target_update_frequency': 32_000,
                'num_train_envs': num_envs,
                'num_test_envs': 1,
                'max_rollout_length': 128,
                'hidden_reset_min_prob': 0,
                'hidden_reset_max_prob': 0,
                'recurrence_type': 'RecurrentAttention',
                'num_recurrence_blocks': 1,
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
                'recurrence_type': 'RecurrentAttention2',
            },
        },
    })

    params.add_change('exp-003', {
        'agent': {
            'parameters': {
                'recurrence_type': 'RecurrentAttention3',
            },
        },
    })

    params.add_change('exp-004', {
        'agent': {
            'parameters': {
                'recurrence_type': 'RecurrentAttention4',
            },
        },
    })

    params.add_change('exp-005', {
        'agent': {
            'parameters': {
                'recurrence_type': 'RecurrentAttention5',
            },
        },
    })

    params.add_change('exp-006', {
        'agent': {
            'parameters': {
                'recurrence_type': 'RecurrentAttention6',
            },
        },
    })

    params.add_change('exp-007', {
        'agent': {
            'parameters': {
                'recurrence_type': 'RecurrentAttention7',
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
            max_iterations : int = 5_000_000,
            slurm : bool = typer.Option(False, '--slurm'),
            wandb : bool = typer.Option(False, '--wandb'),
            debug : bool = typer.Option(False, '--debug')):
        config = get_params()[exp_name]
        pprint(config)
        if debug:
            raise NotImplementedError()
        else:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config=config,
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=250_000,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
            if wandb:
                exp_runner.exp.logger.init_wandb({
                    'project': f'A2C-pathways-{exp_name}'
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
