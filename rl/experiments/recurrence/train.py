import os
from pathlib import Path
from typing import Optional

import dill
import experiment.logger
from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger

from rl.agent.smdp.a2c import A2CAgentRecurrentVec
from rl.experiments.training.vectorized import TrainExperiment
from rl.experiments.training._utils import ExperimentConfigs


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

    # Decrease framestack from default of 4 to 3 (working)
    params.add_change('exp-002', {
        'env_test':  {'atari_config': {'stack_num': 3}},
        'env_train': {'atari_config': {'stack_num': 3}},
    })

    # Decrease framestack to 2
    params.add_change('exp-003', {
        'env_test':  {'atari_config': {'stack_num': 2}},
        'env_train': {'atari_config': {'stack_num': 2}},
    })

    # Decrease framestack to 1
    params.add_change('exp-004', {
        'env_test':  {'atari_config': {'stack_num': 1}},
        'env_train': {'atari_config': {'stack_num': 1}},
    })

    # All of the above experiments encountered catastrophic forgetting
    # Trying a random thing to see if it helps
    params.add_change('exp-005', {
        'agent': {
            'parameters': {
                'hidden_reset_min_prob': 0,
                'hidden_reset_max_prob': 0.5,
            },
        },
    }) # Haven't tried this since fixing the hidden state bug. Not sure if it's useful or not.

    # It could be due to not having enough randomness in the environment?
    params.add('exp-006', {
        'env_test':  {'atari_config': {'repeat_action_probability': 0.25}},
        'env_train': {'atari_config': {'repeat_action_probability': 0.25}},
    }, inherit='exp-004') # I have no idea if this worked. The last experiment I ran was bugged and didn't use this config.

    params.add_change('exp-007', {
        'agent': {
            'parameters': {
                'max_rollout_length': 256,
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
        logger.load_to_memory(verbose=True)
        output_directory = x['exp']['output_directory']
        plot_directory = os.path.join(output_directory,'plots')
        os.makedirs(plot_directory,exist_ok=True)

        for k in ['agent_train_state_value_target_net', 'agent_train_state_value', 'reward']:
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
