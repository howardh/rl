import os
from pathlib import Path
from typing import Optional
from pprint import pprint

import dill
import experiment.logger
from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger
import gym_minigrid.minigrid
import gym_minigrid.register

from rl.experiments.training.vectorized import TrainExperiment
from rl.experiments.training._utils import ExperimentConfigs


class GoalDeterministic(gym_minigrid.minigrid.Goal):
    def __init__(self, reward):
        super().__init__()
        self.reward = reward


def get_params():
    #from rl.agent.smdp.a2c import PPOAgentRecurrentVec as AgentPPO
    from rl.experiments.pathways.train import AttnRecAgentPPO as AgentPPO

    params = ExperimentConfigs()

    num_envs = 16
    env_name = 'MiniGrid-NRoomBanditsSmall-v0'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'episode_stack': 100,
            'dict_obs': True,
            'action_shuffle': False,
            'config': {}
        }] * num_envs
    }

    params.add('exp-001', {
        'agent': {
            'type': AgentPPO,
            'parameters': {
                'num_train_envs': num_envs,
                'num_test_envs': 1,
                'obs_scale': {
                    'obs (image)': 1.0 / 255.0,
                },
                'max_rollout_length': 128,
                'model_type': 'ModularPolicy5',
                'recurrence_type': 'RecurrentAttention10',
                'architecture': [3, 3]
            },
        },
        'env_test': env_config,
        'env_train': env_config,
        'test_frequency': None,
        'save_model_frequency': None,
        'verbose': True,
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
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config={
                        **config,
                        #'save_model_frequency': 100_000, # This is number of iterations, not the number of transitions experienced
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    #checkpoint_frequency=250_000,
                    checkpoint_frequency=None,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
        else:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config={
                        **config,
                        #'save_model_frequency': 100_000, # This is number of iterations, not the number of transitions experienced
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=250_000,
                    #checkpoint_frequency=None,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
        if wandb:
            exp_runner.exp.logger.init_wandb({
                'project': f'PPO-bandits-{exp_name}'
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
        logger.load_to_memory(verbose=True)
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