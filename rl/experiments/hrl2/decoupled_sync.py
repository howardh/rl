import os
import itertools
import pprint
from typing import Optional

from PIL import Image
from copy import copy
import dill
import gym.spaces
from gym.wrappers import TimeLimit
from tqdm import tqdm
import torch
import torch.cuda
import numpy as np
import gym
import gym.envs
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from experiment import Experiment, load_checkpoint, make_experiment_runner
from experiment.logger import Logger
import rl.agent.smdp.sac as sac
import rl.agent.smdp.dqn as dqn
#import rl.agent.smdp.rand as rand
import rl.agent.smdp.hrl as hrl
#import rl.agent.smdp.constant as constant

def make_env(env_name, time_limit=100_000):
    env = gym.make(env_name)
    if isinstance(env,gym.wrappers.TimeLimit):
        env = env.env
    env = TimeLimit(env,time_limit)
    return env

def test(env, agent, render=False, video_file_name='output.avi'):
    total_reward = 0
    total_steps = 0

    obs = env.reset()
    agent.observe(obs, testing=True)
    if render:
        frame = env.render(mode='rgb_array')
        video = cv2.VideoWriter(video_file_name, 0, 60, (frame.shape[0],frame.shape[1])) # type: ignore
        video.write(frame)
    for total_steps in tqdm(itertools.count(), desc='test episode'):
        obs, reward, done, _ = env.step(agent.act(testing=True))
        if render:
            frame = env.render(mode='rgb_array')
            video.write(frame) # type: ignore
        total_reward += reward
        agent.observe(obs, reward, done, testing=True)
        if done:
            break
    env.close()
    if render:
        video.release() # type: ignore

    def compute_action_counts_test():
        parent_action = agent.logger[-1]['parent_action_testing']
        count = [0] * agent.num_children
        for a in parent_action:
            count[a] += 1
        return count

    def compute_action_counts_train():
        count = [0] * agent.num_children
        for x,_ in zip(reversed(agent.logger),range(1000)):
            if 'parent_action_training' not in x:
                continue
            a = x['parent_action_training']
            count[a] += 1
        return count

    return {
        'total_steps': total_steps,
        'total_reward': total_reward,
        'parent_actions_count_test': compute_action_counts_test(),
        'parent_actions_count_train': compute_action_counts_train(),
    }

def render_test_episode(env, agent : hrl.HRLAgent, video_file_name : str ='output.avi'):
    import cv2
    total_reward = 0

    obs = env.reset()
    agent.observe(obs, testing=True)

    frame = env.render(mode='rgb_array')
    width = frame.shape[0] # XXX: Is this the width or height?
    video = cv2.VideoWriter('temp.avi', 0, 60, (frame.shape[0],frame.shape[1])) # type: ignore
    video.write(frame)

    for _ in tqdm(itertools.count(), desc='test episode'):
        obs, reward, done, _ = env.step(agent.act(testing=True))

        frame = env.render(mode='rgb_array')
        video.write(frame)

        total_reward += reward
        agent.observe(obs, reward, done, testing=True)
        if done:
            break
    env.close()
    video.release()

    actions = agent.logger[-1]['parent_action_testing']
    colours = [(255,0,0),(0,255,0),(0,0,255)]
    timeline = np.array([colours[a] for a in actions], dtype=np.uint8)
    timeline = np.expand_dims(timeline,0)
    timeline_img = Image.fromarray(timeline)
    timeline_img = timeline_img.resize([frame.shape[0],1], Image.ANTIALIAS)
    timeline_marker = np.array([
            [0,1,0],
            [1,1,1],
    ], dtype=np.uint8)
    timeline_marker_img = Image.fromarray(timeline_marker)

    video = cv2.VideoCapture('temp.avi') # type: ignore
    video2 = cv2.VideoWriter(video_file_name, 0, 60, (frame.shape[0],frame.shape[1])) # type: ignore

    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT) # type: ignore

    for i in itertools.count():
        ret, frame = video.read()

        if not ret:
            break

        frame = Image.fromarray(frame)
        frame.paste(timeline_img, (0,0))
        frame.paste(timeline_marker_img, (int(width/(frame_count-1)*i),1))
        frame = np.array(frame)
        video2.write(frame)

    video2.release()

def _get_mujoco_state(env):
    return {
            'qpos': env.sim.data.qpos,
            'qvel': env.sim.data.qvel,
    }

class HRLExperiment(Experiment): # DQN parent, SAC children
    def setup(self, config, output_directory):
        self.output_directory = output_directory
        self.device = self._init_device()

        self._test_iterations = config.get('test_iterations',5)
        self._test_frequency = config.get('test_frequency',1000)
        self._deploy_state_checkpoint_frequency = 250_000

        self.env = self._init_envs()
        self.agent = self._init_agents(
                delay=config.get('delay'),
                parent_params=config.get('parent_params'),
                children_params=config.get('children_params'),
                env=self.env[0],
                device=self.device
        )

        self.done = True
        self._best_score = float('-inf')
        self.logger = Logger(key_name='step')
    def _init_device(self):
        if torch.cuda.is_available():
            print('GPU found')
            return torch.device('cuda')
        else:
            print('No GPU found. Running on CPU.')
            return torch.device('cpu')
    def _init_envs(self):
        available_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
        if 'Hopper-v1' in available_envs:
            env_name = 'Hopper-v1'
        else:
            env_name = 'Hopper-v2'
        env = [make_env(env_name,1_000), make_env(env_name,1_000)]
        return env
    def _init_agents(self, delay, parent_params, children_params, env, device):
        # Children
        children = []
        for child_param in children_params:
            child_param = copy(child_param)
            pi_net  = sac.PolicyNetwork(
                    env.observation_space.shape[0],
                    env.action_space.shape[0],
                    structure = child_param.pop('pi_net_structure')
            ).to(device)
            sac_agent = sac.SACAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                reward_scale=5,
                **child_param,
                q_net_1 = sac.QNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device),
                q_net_2 = sac.QNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device),
                v_net   = sac.VNetwork(env.observation_space.shape[0]).to(device),
                pi_net  = pi_net,
                device=device,
            )
            children.append(sac_agent)
        # Parent
        parent_params = copy(parent_params)
        q_net_structure = parent_params.pop('q_net_structure')
        q_net = dqn.QNetworkFCNN([env.observation_space.shape[0],*q_net_structure,len(children)]).to(device)
        dqn_agent = dqn.DQNAgent(
            action_space=gym.spaces.Discrete(len(children)),
            observation_space=env.observation_space,
            **parent_params,
            q_net=q_net,
            device=device,
        )
        # Hierarchy
        return hrl.HRLAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            agent = dqn_agent,
            children = children,
            children_discount = [p['discount_factor'] for p in children_params],
            delay=delay,
        )
    def run_step(self, iteration):
        # Save agent specs that can be deployed
        if iteration % self._deploy_state_checkpoint_frequency == 0:
            filename = os.path.join(self.output_directory,'deploy_state-%d.pkl'%iteration)
            self._save_deploy_state(filename)
            tqdm.write('Saved deploy state at %s' % os.path.abspath(filename))
        # Test and plot results
        if iteration % self._test_frequency == 0:
            self._test(iteration)
            self._plot()
            # Save best model
            score = np.mean(self.logger[-1]['total_reward'])
            if score > self._best_score:
                self._best_score = score
                filename = os.path.join(self.output_directory,'deploy_state-best.pkl')
                self._save_deploy_state(filename)
                tqdm.write('Saved deploy state at %s with score %f' % (os.path.abspath(filename), score))
        # Train agent
        self._train()
    def _save_deploy_state(self, filename):
        data = self.agent.state_dict_deploy()
        with open(filename, 'wb') as f:
            dill.dump(data,f)
    def _test(self,iteration):
        for _ in range(self._test_iterations):
            results = test(self.env[1], self.agent, render=False)
            self.logger.append(step=iteration, total_steps=results['total_steps'], total_reward=results['total_reward'])

        def compute_action_counts_test():
            parent_action = self.agent.logger[-1]['parent_action_testing']
            count = [0] * self.agent.num_children
            for a in parent_action:
                count[a] += 1
            return count

        def compute_action_counts_train():
            count = [0] * self.agent.num_children
            for x,_ in zip(reversed(self.agent.logger),range(1000)):
                if 'parent_action_training' not in x:
                    continue
                a = x['parent_action_training']
                count[a] += 1
            return count

        self.logger.log(
                step=iteration,
                parent_actions_count_train=compute_action_counts_train(),
                parent_actions_count_test=compute_action_counts_test())
        tqdm.write(pprint.pformat(self.logger[-1], indent=4))

    def _train(self):
        env = self.env[0]
        done = self.done
        if done:
            obs = env.reset()
            self.agent.observe(obs, testing=False)
        obs, reward, done, _ = env.step(self.agent.act(testing=False))
        self.agent.observe(obs, reward, done, testing=False)
        self.done = done
    def _plot(self):
        plot_directory = os.path.join(self.output_directory,'plots')
        if not os.path.isdir(plot_directory):
            os.makedirs(plot_directory)

        test_indices = [i for i,x in enumerate(self.logger) if 'total_reward' in x]
        keys = [self.logger[i]['step'] for i in test_indices]

        ##################################################
        # Reward over time
        y = [np.mean(self.logger[i]['total_reward']) for i in test_indices]
        plt.plot(keys,y)
        plt.ylabel('Total Reward')
        plt.xlabel('Steps')
        plt.grid()

        filename = os.path.join(plot_directory,'plot-reward.png')
        plt.savefig(filename)
        print('Saved plot to %s' % os.path.abspath(filename))

        plt.yscale('log')
        filename = os.path.join(plot_directory,'plot-reward-log.png')
        plt.savefig(filename)
        print('Saved plot to %s' % os.path.abspath(filename))

        plt.close()

        ##################################################
        # Action Choice
        for traintest in ['train','test']:
            y = np.array(
                [self.logger[i]['parent_actions_count_%s' % traintest] for i in test_indices]
            )
            y = y/y.sum(1, keepdims=True)
            if y.shape[0] < 3:
                continue
            y = y.transpose()
            plt.stackplot(keys, *y, labels=range(y.shape[0]))
            plt.ylabel('Training Subpolicy Choice Distribution')
            plt.xlabel('Steps')
            plt.legend(loc='best')
            plt.grid()

            filename = os.path.join(plot_directory,'plot-actions-%s.png' % traintest)
            plt.savefig(filename)
            print('Saved plot to %s' % os.path.abspath(filename))
            plt.close()

    def state_dict(self):
        return {
            'agent': self.agent.state_dict(),
            'env_state': [_get_mujoco_state(env.unwrapped) for env in self.env],
            'done': self.done,
            'logger': self.logger.state_dict(),
        }
    def load_state_dict(self, state):
        self.agent.load_state_dict(state['agent'])
        for env,env_state in zip(self.env, state['env_state']):
            env.reset() # Without this line, we get the error "Cannot call env.step() before calling reset()"
            env.unwrapped.set_state(**env_state)
        self.done = state['done']

def make_app():
    import typer
    app = typer.Typer()

    def get_params():
        params = {}

        base_dqn_params = {
            'discount_factor': 0.99,
            'q_net_structure': [256,256],
            'learning_rate': 1e-3,
            'batch_size': 32,
            'warmup_steps': 1000,
            'replay_buffer_size': 50_000,
            'target_update_frequency': 1000,
            'polyak_rate': 1,
            'behaviour_eps': 0.1,
            'target_eps': 0,
        }
        base_sac_params = {
            'discount_factor': 0.99,
            'pi_net_structure': [256,256],
            'learning_rate': 3e-4,
            'batch_size': 256,
            'warmup_steps': 1000,
            'replay_buffer_size': 50_000,
            'target_update_frequency': 1,
            'polyak_rate': 0.005,
        }
        params['hrl-001'] = {
            'test_iterations': 5,
            'test_frequency': 1000,
            'parent_params': base_dqn_params,
            'children_params': [
                base_sac_params,
                base_sac_params,
            ],
        }
        params['hrl-002'] = {
            'test_iterations': 5,
            'test_frequency': 1000,
            'delay': 2,
            'parent_params': base_dqn_params,
            'children_params': [
                base_sac_params,
                base_sac_params,
            ],
        }

        return params

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
            exp = make_experiment_runner(
                    HRLExperiment,
                    experiment_name=exp_name,
                    trial_id=trial_id,
                    verbose=True,
                    checkpoint_frequency=10,
                    max_iterations=100,
                    results_directory=results_directory,
                    config={
                        **config,
                        'test_iterations': 2,
                        'test_frequency': 20,
                    })
        else:
            exp = make_experiment_runner(
                    HRLExperiment,
                    experiment_name=exp_name,
                    trial_id=trial_id,
                    verbose=True,
                    checkpoint_frequency=10_000,
                    max_iterations=4_000_000,
                    results_directory=results_directory,
                    config=config)
        exp.run()

    @app.command()
    def checkpoint(filename):
        exp = load_checkpoint(HRLExperiment, filename)
        exp.run()

    @app.command()
    def video(state_filename : str,
            output : str = 'output.avi'):
        with open(state_filename,'rb') as f:
            state = dill.load(f)

        available_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
        if 'Hopper-v1' in available_envs:
            env_name = 'Hopper-v1'
        else:
            env_name = 'Hopper-v2'
        env = make_env(env_name,1000)

        device = torch.device('cpu') # TODO
        agent = hrl.HRLAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                agent = dqn.DQNAgent(
                    action_space=gym.spaces.Discrete(len(state['children'])),
                    observation_space=env.observation_space,
                    q_net = dqn.QNetworkFCNN([env.observation_space.shape[0],256,256,len(state['children'])]).to(device)
                ),
                children = [
                    sac.SACAgent(
                        action_space=env.action_space,
                        observation_space=env.observation_space,
                        q_net_1 = sac.QNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device),
                        q_net_2 = sac.QNetwork(env.observation_space.shape[0], env.action_space.shape[0]).to(device),
                        v_net   = sac.VNetwork(env.observation_space.shape[0]).to(device),
                        pi_net  = sac.PolicyNetwork(
                                env.observation_space.shape[0],
                                env.action_space.shape[0],
                                structure = [256,256]
                        ).to(device)
                    )
                    for _ in state['children']
                ],
                children_discount=0.99,
                delay=state['delay']
        )
        agent.load_state_dict_deploy(state)
        render_test_episode(env, agent, video_file_name=output)
        print('Saved video to %s' % os.path.abspath(output))

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
