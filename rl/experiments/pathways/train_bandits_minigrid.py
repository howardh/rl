import os
import itertools
from pathlib import Path
from typing import Optional, Tuple
from pprint import pprint

from tqdm import tqdm
import numpy as np
import dill
import experiment.logger
from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger
import gym_minigrid.minigrid
import gym_minigrid.register

from rl.experiments.training.vectorized import TrainExperiment, make_vec_env
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

    # Use a different wrapper for all the meta-learning stuff. Previous version was bugged.
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 100,
                'dict_obs': True,
                'randomize': True,
            },
            'config': {
                'rewards': [1, -1],
                'shuffle_goals_on_reset': False,
            }
        }] * num_envs
    }
    params.add('exp-002', {
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
                'recurrence_type': 'RecurrentAttention11',
                'architecture': [3, 3]
            },
        },
        'env_test': env_config,
        'env_train': env_config,
        'test_frequency': None,
        'save_model_frequency': None,
        'verbose': True,
    }) 

    # Add the reward permutation as part of the observation
    env_config = {
        #'env_type': 'gym_sync',
        'env_configs': [{
            'config': {
                'include_reward_permutation': True,
            }
        }] * num_envs
    }
    params.add_change('exp-003', {
        'env_test': env_config,
        'env_train': env_config,
        #'save_model_frequency': 1_000_000,
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
                    checkpoint_frequency=25_000,
                    #checkpoint_frequency=None,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
        if wandb:
            exp_runner.exp.logger.init_wandb({
                'project': f'PPO-minigrid-bandits-{exp_name}'
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

    @app.command()
    def test(
            checkpoint_filename: Path,
            output: Path = None,
            num_trials: int = 10,
            reward_config: Tuple[float,float] = (1, -1)):
        import matplotlib
        if output is not None:
            matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        exp = load_checkpoint(TrainExperiment, checkpoint_filename)
        num_steps = exp._steps * exp.exp.agent.num_training_envs
        env = make_vec_env(
            env_type = 'gym_sync',
            env_configs = [{
                'env_name': 'MiniGrid-NRoomBanditsSmall-v0',
                'minigrid': True,
                'minigrid_config': {},
                'meta_config': {
                    'episode_stack': 100,
                    'dict_obs': True,
                    'randomize': False,
                },
                'config': {
                    'rewards': reward_config,
                    'shuffle_goals_on_reset': False,
                }
            }]
        )

        agent = exp.exp.agent
        results = {}

        results['total_reward'] = []
        results['reward'] = []
        agent = exp.exp.agent

        agent.reset()
        obs = env.reset()
        done = np.array([False] * env.num_envs)
        agent.observe(obs, testing=True)
        for i in range(num_trials):
            print(f'Trial {i}')
            results['total_reward'].append([])
            results['reward'].append([])
            total_reward = 0
            for _ in tqdm(itertools.count()):
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                agent.observe(obs, reward, done, testing=True)

                total_reward += reward[0]
                results['total_reward'][-1].append(total_reward)
                results['reward'][-1].append(reward[0])
                #print(f'{len(results["total_reward"][-1])} {action} {reward} {total_reward}')
                if obs['done'].any():
                    tqdm.write(str([g.reward for g in env.envs[0].goals])) # type: ignore
                    tqdm.write('ep done')
                if done.any():
                    tqdm.write('-'*80)
                    break

        data = [[],[]]
        for i in range(num_trials):
            r = np.array(results['reward'][i])
            nzr = r[r.nonzero()]
            halfway_index = len(nzr)//2
            mean_r_1 = np.mean(nzr[:halfway_index])
            mean_r_2 = np.mean(nzr[halfway_index:])
            if not np.isfinite(mean_r_1) or not np.isfinite(mean_r_2):
                breakpoint()
                continue
            data[0].append(mean_r_1)
            data[1].append(mean_r_2)
        plt.boxplot(data, labels=['1st half','2nd half'])
        plt.ylabel('Average Rewards')
        plt.title(f'{num_trials} Trials on {str(reward_config)} after {num_steps:,} steps')
        if output is not None:
            plt.savefig(output)
            print(f'Saved to {os.path.abspath(output)}')
        else:
            plt.show()
        breakpoint()

    @app.command()
    def video(checkpoint_filename : Path, reward_config: Tuple[float,float] = (1, -1)):
        import cv2
        import PIL.Image, PIL.ImageDraw, PIL.ImageFont
        from fonts.ttf import Roboto # type: ignore

        num_trials = 1
        exp = load_checkpoint(TrainExperiment, checkpoint_filename)
        env = make_vec_env(
            env_type = 'gym_sync',
            env_configs = [{
                'env_name': 'MiniGrid-NRoomBanditsSmall-v0',
                'minigrid': True,
                'minigrid_config': {},
                'meta_config': {
                    'episode_stack': 100,
                    'dict_obs': True,
                    'randomize': False,
                },
                'config': {
                    'rewards': reward_config,
                    'shuffle_goals_on_reset': False,
                    'include_reward_permutation': False,
                }
            }]
        )

        def concat_images(images, padding=0, direction='h', align=0):
            if direction == 'h':
                width = sum([i.size[0] for i in images]) + padding * (len(images) + 1)
                height = max([i.size[1] for i in images]) + padding*2
                new_image = PIL.Image.new('RGB', (width, height), color=(255,255,255))
                x = 0
                for i in images:
                    new_image.paste(i, (x+padding, (height - 2*padding - i.size[1]) // 2 * (align + 1) + padding))
                    x += i.size[0] + padding
                return new_image
            elif direction == 'v':
                width = max([i.size[0] for i in images]) + padding*2
                height = sum([i.size[1] for i in images]) + padding * (len(images) + 1)
                new_image = PIL.Image.new('RGB', (width, height), color=(255,255,255))
                y = 0
                for i in images:
                    new_image.paste(i, ((width - 2*padding - i.size[0]) // 2 * (align + 1) + padding, y + padding))
                    y += i.size[1] + padding
                return new_image
            else:
                raise ValueError('direction must be "h" or "v"')

        def draw_attention(core_attention, query_gating, output_attention):
            block_size = 24
            padding = 2

            core_images = []
            for layer in core_attention:
                num_blocks, _, num_inputs = layer.shape
                width = num_inputs*block_size + (num_inputs+1)*padding
                height = num_blocks*block_size + (num_blocks+1)*padding
                img = PIL.Image.new('RGB', (width, height), color=(255,255,255))
                for i in range(num_blocks):
                    for j in range(num_inputs):
                        weight = layer[i,0,j].item()
                        c = int(255*(1-weight))
                        x = j*(block_size+padding) + padding
                        y = i*(block_size+padding) + padding
                        PIL.ImageDraw.Draw(img).rectangle(
                                (x,y,x+block_size,y+block_size),
                                fill=(c,c,c),
                        )
                core_images.append(img)
            core_imags_concat = concat_images(core_images, padding=padding, direction='v', align=1)

            num_layers = len(query_gating)
            max_layer_size = max(layer.shape[0] for layer in query_gating)
            width = num_layers*block_size + (num_layers+1)*padding
            height = max_layer_size*block_size + (max_layer_size+1)*padding
            query_image = PIL.Image.new('RGB', (width, height), color=(255,255,255))
            for i, layer in enumerate(query_gating):
                num_blocks = layer.shape[0]
                for j in range(num_blocks):
                    weight = layer[j,0].item()
                    c = int(255*(1-weight))
                    x = i*(block_size+padding) + padding
                    y = j*(block_size+padding) + padding
                    PIL.ImageDraw.Draw(query_image).rectangle(
                            (x,y,x+block_size,y+block_size),
                            fill=(c,c,c)
                    )

            output_images = {}
            for k, layer in output_attention.items():
                layer = layer.squeeze()
                num_inputs = len(layer)
                width = num_inputs*block_size + (num_inputs+1)*padding
                height = block_size + 2*padding
                img = PIL.Image.new('RGB', (width, height), color=(255,255,255))
                for i in range(num_inputs):
                    weight = layer[i].item()
                    c = int(255*(1-weight))
                    x = i*(block_size+padding) + padding
                    y = padding
                    PIL.ImageDraw.Draw(img).rectangle(
                            (x,y,x+block_size,y+block_size),
                            fill=(c,c,c)
                    )
                output_images[k] = img

            font_family = Roboto
            font_size = 18
            font = PIL.ImageFont.truetype(font_family, font_size)
            text_images = {}
            for k in output_attention.keys():
                text_width, text_height = font.getsize(k)
                img = PIL.Image.new('RGB',
                        (text_width+2*padding, text_height+2*padding),
                        color=(255,255,255))
                draw = PIL.ImageDraw.Draw(img)
                draw.fontmode = 'L' # type: ignore
                draw.text(
                        (padding, padding),
                        k,
                        font=font,
                        fill=(0,0,0)
                )
                text_images[k] = img

            output_images_concat = concat_images(
                    [
                        concat_images(
                            [
                                text_images[k],
                                output_images[k],
                            ],
                            padding = padding,
                            direction='v',
                            align=-1,
                        )
                        for k in output_images.keys()],
                    padding=padding, direction='v'
            )

            all_images_concat = concat_images(
                    [
                        core_imags_concat,
                        query_image,
                        output_images_concat,
                    ],
                    padding=padding, direction='h'
            )

            return all_images_concat

        agent = exp.exp.agent
        results = {}
        fps = 25

        results['agent'] = []
        agent = exp.exp.agent
        for i in range(num_trials):
            video_writer = cv2.VideoWriter( # type: ignore
                    f'video-{i}.webm',
                    cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                    fps,
                    (env.envs[0].unwrapped.width*32, env.envs[0].unwrapped.height*32), # type: ignore
            )
            video_writer2 = cv2.VideoWriter( # type: ignore
                    f'video2-{i}.webm',
                    cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                    fps,
                    env.envs[0].observation_space['obs (image)'].shape[1:], # type: ignore
            )
            video_writer3 = None
            num_frames = 0

            results['agent'].append([])
            agent.reset()
            obs = env.reset()
            done = np.array([False] * env.num_envs)
            agent.observe(obs, testing=True)

            frame = env.envs[0].render(mode=None) # type: ignore
            video_writer.write(frame[:,:,::-1])
            video_writer2.write(np.moveaxis(obs['obs (image)'].squeeze(), 0, 2)[:,:,::-1])
            while not done[0]:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                agent.observe(obs, reward, done, testing=True)
                num_frames += 1
                print(f'{num_frames} {action} {reward}')
                frame = env.envs[0].render(mode=None) # type: ignore
                video_writer.write(frame[:,:,::-1])
                video_writer2.write(np.moveaxis(obs['obs (image)'].squeeze(), 0, 2)[:,:,::-1])
                attn_img = draw_attention(
                        core_attention = agent.net.last_attention,
                        query_gating = agent.net.last_ff_gating,
                        output_attention = agent.net.last_output_attention)
                if video_writer3 is None:
                    video_writer3 = cv2.VideoWriter( # type: ignore
                            f'video3-{i}.webm',
                            cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                            fps,
                            attn_img.size,
                    )
                video_writer3.write(np.array(attn_img)[:,:,::-1])
                #if num_frames > 100:
                #    break
            video_writer.release()
            video_writer2.release()
            if video_writer3 is not None:
                video_writer3.release()

    commands = {
            'run': run,
            'checkpoint': checkpoint,
            'plot': plot,
            'test': test,
            'video': video,
    }

    return app, commands


def run():
    app,_ = make_app()
    app()


if __name__ == "__main__":
    run()
