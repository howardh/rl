import os
from typing import Optional
from pathlib import Path

import dill

from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger

from rl.agent.option_critic import OptionCriticAgent
from rl.experiments.training.basic import TrainExperiment

def get_agent_params():
    base_agent_params = {
        'discount_factor': 0.99,
        'behaviour_eps': 0.02,
        'learning_rate': 1e-4,
        'update_frequency': 1,
        'target_update_frequency': 200,
        'polyak_rate': 1,
        'num_options': 1,
        'termination_reg': 0.01,
        'entropy_reg': 0.01,
        'deliberation_cost': 0,
    }

    params = {}
    params['oc-001'] = {
        'type': OptionCriticAgent,
        'parameters': base_agent_params,
    }

    # Use 8 options like in the option-critic paper
    params['oc-002'] = {
        'type': OptionCriticAgent,
        'parameters': {
            **base_agent_params,
            'num_options': 8,
        },
    }

    # Option-Critic with deliberation cost params
    params['oc-003'] = {
        'type': OptionCriticAgent,
        'parameters': {
            **base_agent_params,
            'num_options': 8,
            'behaviour_eps': 0.1,
            'target_eps': 0.1,
            'learning_rate': 7e-4,
            'termination_reg': 0,
            'deliberation_cost': 0.01,
            'target_update_frequency': 1, # Jean's code doesn't use a target network, and there's no mention of it in the paper.
            'optimizer': 'rmsprop',
        },
    }

    return params

def get_env_params():
    pong = [{
        'env_name': 'ALE/Pong-v5',
        'atari': True,
        'config': {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 0,
            'repeat_action_probability': 0.25,
        }
    },{
        'env_name': 'ALE/Pong-v5',
        'atari': True,
        'config': {
            'frameskip': 1,
            'mode': 1,
            'difficulty': 0,
            'repeat_action_probability': 0.25,
        }
    },{
        'env_name': 'ALE/Pong-v5',
        'atari': True,
        'config': {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 1,
            'repeat_action_probability': 0.25,
        }
    },{
        'env_name': 'ALE/Pong-v5',
        'atari': True,
        'config': {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 3,
            'repeat_action_probability': 0.25,
        }
    },]

    seaquest = [{
        'env_name': 'ALE/Seaquest-v5',
        'atari': True,
        'config': {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 0,
            'repeat_action_probability': 0.25,
        }
    },{
        'env_name': 'ALE/Seaquest-v5',
        'atari': True,
        'config': {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 1,
            'repeat_action_probability': 0.25,
        }
    },{
        'env_name': 'ALE/Seaquest-v5',
        'atari': True,
        'atari_config': {
            'terminal_on_life_loss': True
        },
        'config': {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 0,
            'repeat_action_probability': 0.25,
        }
    }]

    return {
        'pong': pong,
        'seaquest': seaquest,
    }

def get_params():
    agent_params = get_agent_params()
    env_params = get_env_params()
    params = {}

    num_actors = 16
    train_env_keys = list(range(num_actors))
    base_exp_params = {
        'train_env_keys': train_env_keys,
        'save_model_frequency': 250_000,
        'verbose': True,
        'test_frequency': None,
    }

    params['exp-001'] = {
        'agent': agent_params['oc-001'],
        'env_test': env_params['pong'][0],
        'env_train': env_params['pong'][0],
        **base_exp_params,
    }

    params['exp-002'] = {
        'agent': agent_params['oc-002'],
        'env_test': env_params['pong'][0],
        'env_train': env_params['pong'][0],
        **base_exp_params,
    }

    params['exp-003'] = {
        'agent': agent_params['oc-002'],
        'env_test': env_params['seaquest'][0],
        'env_train': env_params['seaquest'][0],
        **base_exp_params,
    }

    params['exp-004'] = {
        'agent': agent_params['oc-001'],
        'env_test': env_params['pong'][3],
        'env_train': env_params['pong'][3],
        **base_exp_params,
    }

    params['exp-005'] = {
        'agent': agent_params['oc-003'],
        'env_test': env_params['seaquest'][0],
        'env_train': env_params['seaquest'][0],
        **base_exp_params,
    }

    params['exp-006'] = { # Terminate on life loss
        'agent': agent_params['oc-003'],
        'env_test': env_params['seaquest'][2],
        'env_train': env_params['seaquest'][2],
        **base_exp_params,
    }

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
            #exp_runner = make_experiment_runner(
            #        TrainExperiment,
            #        config={
            #            **config,
            #            #'save_model_frequency': 5,
            #        },
            #        results_directory=results_directory,
            #        trial_id=trial_id,
            #        checkpoint_frequency=2000,
            #        max_iterations=2000,
            #        #checkpoint_frequency=100,
            #        #max_iterations=10_000,
            #        slurm_split=slurm,
            #        verbose=True,
            #)
            exp_runner = make_experiment_runner( # Debug checkpointing
                    TrainExperiment,
                    config={
                        **config,
                        #'save_model_frequency': 5,
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=10_000,
                    max_iterations=500_000,
                    slurm_split=slurm,
                    verbose=True,
            )
            #exp_runner.exp.logger.init_wandb({
            #    'project': f'Compressibility-train-{exp_name} (debug)',
            #})
        else:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config=config,
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=250_000,
                    max_iterations=100_000_000,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
            #exp_runner.exp.logger.init_wandb({
            #    'project': f'Compressibility-train-{exp_name}',
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
