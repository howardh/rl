import os
import itertools
#from typing import Mapping

import gym
from gym.wrappers import FrameStack#, AtariPreprocessing
from gym.spaces import Box
import torch
import dill
import numpy as np
from tqdm import tqdm
import cv2

from experiment import Experiment
from experiment.logger import Logger
from experiment.plotter import plot

from rl.utils import default_state_dict, default_load_state_dict
import rl.debug_tools.frozenlake

def _run_test(env, agent, verbose=False, env_key='test'):
    steps = itertools.count()
    if verbose:
        steps = tqdm(steps, desc='test episode')

    total_reward = 0
    total_steps = 0

    obs = env.reset()
    agent.observe(obs, testing=True, env_key=env_key)
    for total_steps in steps:
        obs, reward, done, _ = env.step(agent.act(testing=True, env_key=env_key))
        total_reward += reward
        agent.observe(obs, reward, done, testing=True, env_key=env_key)
        if done:
            break
    env.close()

    return {
        'total_steps': total_steps,
        'total_reward': total_reward,
    }

def make_env(env_name, config={}, atari=False, one_hot_obs=False, atari_config={}, frame_stack=4):
    env = gym.make(env_name, **config)
    if atari:
        env = AtariPreprocessing(env, **atari_config)
        env = FrameStack(env, frame_stack)
    if one_hot_obs:
        env = rl.debug_tools.frozenlake.OnehotObs(env)
    return env

class TrainExperiment(Experiment):
    """ An experiment for training a standard RL algorithm. """
    def setup(self, config, output_directory=None):
        self.logger = Logger(key_name='step',allow_implicit_key=True)
        self.logger.log(step=0)

        self.device = self._init_device()

        self._train_env_keys = config.get('train_env_keys',[0])
        self._test_env_keys = config.get('test_env_keys',['test'])
        self.env_test = make_env(**config['env_test'])
        self.env_train = {
                k:make_env(**config['env_train'])
                for k in self._train_env_keys
        }

        self.agent = config.get('agent')
        self.agent = self._init_agent(
                agent_config=config.get('agent'),
                env = self.env_test,
                device = self.device,
        )

        self.output_directory = output_directory
        self.test_frequency = config.get('test_frequency')
        self.num_test_episodes = config.get('num_test_episodes', 5)
        self.save_model_frequency = config.get('save_model_frequency')
        self.verbose = config.get('verbose',False)

        self.done = {k:True for k in self._train_env_keys}
        self._ep_len = {k:0 for k in self._train_env_keys}
        self._ep_rewards = {k:[] for k in self._train_env_keys}
    def _init_device(self):
        if torch.cuda.is_available():
            print('GPU found')
            return torch.device('cuda')
        else:
            print('No GPU found. Running on CPU.')
            return torch.device('cpu')
    def _init_agent(self, agent_config, env, device):
        cls = agent_config['type']
        parameters = agent_config['parameters']
        return cls(
            action_space=env.action_space,
            observation_space=env.observation_space,
            logger=self.logger.make_sublogger('agent_'),
            **parameters,
            device=device,
        )
    def run_step(self, i):
        self.logger.log(step=i)
        if self.test_frequency is not None and i % self.test_frequency == 0:
            self._test(i)
        if self.save_model_frequency is not None and i % self.save_model_frequency == 0:
            self._save_model(i)
        self._train(i)
    def _test(self,i):
        test_results = [_run_test(self.env_test, self.agent, verbose=self.verbose) for _ in tqdm(range(self.num_test_episodes), desc='testing')]
        test_rewards = [x['total_reward'] for x in test_results]
        #action_values = np.mean(self.agent.logger[-1]['testing_action_value'])
        #tqdm.write(f'Iteration {i}\t Average reward: {avg}\t Action values: {action_values}')
        #tqdm.write(pprint.pformat(test_results[i], indent=4))
        #tqdm.write('Mean weights:')
        #tqdm.write(pprint.pformat([x.abs().mean().item() for x in agent.qu_net.parameters()], indent=4))
        self.logger.log(test_rewards=test_rewards, test_rewards_mean=np.mean(test_rewards))
        # Plot
        plot_filename = os.path.join(self.output_directory,'plot.png')
        plot(logger=self.logger, curves=['test_rewards'], filename=plot_filename, xlabel='steps', ylabel='test rewards')
        # Terminal output
        if self.verbose:
            mean_reward = np.mean(test_rewards)
            tqdm.write(f'Iteration {i}\t Average testing reward: {mean_reward}')
            tqdm.write('Plot saved to %s' % os.path.abspath(plot_filename))
    def _train(self,i):
        env_key = self._train_env_keys[i%len(self._train_env_keys)]
        env = self.env_train[env_key]
        self._train_one_env(i,env,env_key)
    def _train_one_env(self,i,env,env_key):
        agent = self.agent

        if self.done[env_key]:
            # TODO: Episode end callback?
            if self.verbose and self._ep_len[env_key] > 0:
                total_reward = sum(self._ep_rewards[env_key])
                self.logger.append(train_reward_by_episode=total_reward)
                #running_avg = np.mean([np.mean(x) for x in self.logger['train_reward_by_episode'][1][-100:]])
                #tqdm.write(f'Iteration {i}\t Training reward: {total_reward}\t avg: {running_avg}')
                tqdm.write(f'Iteration {i}\t Training reward: {total_reward}')
            # Reset
            self.done[env_key] = False
            obs = env.reset()
            agent.observe(obs, testing=False, env_key=env_key)
            self._ep_len[env_key] = 0
            self._ep_rewards[env_key].clear()
        else:
            action = agent.act(testing=False,env_key=env_key)
            obs, reward, self.done[env_key], _ = env.step(action)
            agent.observe(obs, reward, self.done[env_key], testing=False, env_key=env_key)
            # Logging
            self.logger.append(train_reward=reward)
            self._ep_len[env_key] += 1
            self._ep_rewards[env_key].append(reward)
    def _save_model(self,i):
        """ Save the model parameters """
        models_path = os.path.join(self.output_directory, 'models')
        os.makedirs(models_path, exist_ok=True)
        model_file_name = os.path.join(models_path, '%d.pkl' % i)
        if hasattr(self.agent,'state_dict_deploy'):
            model = self.agent.state_dict_deploy()
        else:
            model = self.agent.state_dict()
        with open(model_file_name,'wb') as f:
            dill.dump(model, f)
        # Verbose
        if self.verbose:
            tqdm.write(f'Trained model saved to {model_file_name}')
    def state_dict(self):
        """ Return the experiment state as a dictionary. """
        return default_state_dict(self, [
            'agent',
            'env_train','env_test',
            'output_directory',
            'test_frequency',
            'save_model_frequency',
            'verbose',
            'logger',
            'done',
            'env_train',
            '_ep_len',
            '_ep_rewards',
        ])
    def load_state_dict(self, state):
        default_load_state_dict(self, state)

# See https://github.com/openai/gym/pull/2454
class AtariPreprocessing(gym.Wrapper):
    r"""Atari 2600 preprocessings.
    This class follows the guidelines in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    Specifically:
    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
    * Scale observation: optional
    Args:
        env (Env): environment
        noop_max (int): max number of no-ops
        frame_skip (int): the frequency at which the agent experiences the game.
        screen_size (int): resize Atari frame
        terminal_on_life_loss (bool): if True, then step() returns done=True whenever a
            life is lost.
        grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
            is returned.
        grayscale_newaxis (bool): if True and grayscale_obs=True, then a channel axis is added to
            grayscale observations to make them 3-dimensional.
        scale_obs (bool): if True, then observation normalized in range [0,1] is returned. It also limits memory
            optimization benefits of FrameStack Wrapper.
    """

    def __init__(
        self,
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    ):
        super().__init__(env)
        assert (
            cv2 is not None
        ), "opencv-python package not installed! Try running pip install gym[atari] to get dependencies  for atari"
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            if (
                "NoFrameskip" not in env.spec.id
                and getattr(env.unwrapped, "_frameskip", None) != 1
            ):
                raise ValueError(
                    "disable frame-skipping in the original env. Otherwise, more than one"
                    " frame-skip will happen as through this wrapper"
                )
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs

        # buffer of most recent two observations for max pooling
        if grayscale_obs:
            self.obs_buffer = [
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            ]
        else:
            self.obs_buffer = [
                np.empty(env.observation_space.shape, dtype=np.uint8),
                np.empty(env.observation_space.shape, dtype=np.uint8),
            ]

        self.ale = env.unwrapped.ale
        self.lives = 0
        self.game_over = False

        _low, _high, _obs_dtype = (
            (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        )
        _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
        if grayscale_obs and not grayscale_newaxis:
            _shape = _shape[:-1]  # Remove channel axis
        self.observation_space = Box(
            low=_low, high=_high, shape=_shape, dtype=_obs_dtype
        )

    def step(self, action):
        R = 0.0

        done = False
        info = None
        for t in range(self.frame_skip):
            _, reward, done, info = self.env.step(action)
            R += reward
            self.game_over = done

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.lives = new_lives

            if done:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[1])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[1])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[0])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[0])
        return self._get_obs(), R, done, info

    def reset(self, **kwargs):
        # NoopReset
        self.env.reset(**kwargs)
        noops = (
            self.env.unwrapped.np_random.randint(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset(**kwargs)

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)
        return self._get_obs()

    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = cv2.resize( # type: ignore
            self.obs_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA, # type: ignore
        )

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs

if __name__=='__main__':
    from experiment import make_experiment_runner

    from rl.agent.smdp.dqn import DQNAgent

    env_name = 'PongNoFrameskip-v4'
    #env_name = 'SeaquestNoFrameskip-v4'
    #env_name = 'MsPacmanNoFrameskip-v4'

    #agent = DQNAgent(
    #    action_space=env_train.action_space,
    #    observation_space=env_train.observation_space,
    #    #discount_factor=0.99,
    #    #behaviour_eps=0.02,
    #    #learning_rate=1e-4,
    #    #update_frequency=1,
    #    #target_update_frequency=1_000,
    #    #polyak_rate=1,
    #    #warmup_steps=10_000,
    #    #replay_buffer_size=100_000,
    #    device=device,
    #)

    exp_runner = make_experiment_runner(
            TrainExperiment,
            config={
                'agent': {
                    'type': DQNAgent,
                    'parameters': {
                        'discount_factor': 0.99,
                        'behaviour_eps': 0.02,
                        'learning_rate': 1e-4,
                        'update_frequency': 1,
                        'target_update_frequency': 1_000,
                        'polyak_rate': 1,
                        'warmup_steps': 10_000,
                        'replay_buffer_size': 100_000,
                    }
                },
                'env_test': {'env_name': env_name, 'atari': True},
                'env_train': {'env_name': env_name, 'atari': True},
                'test_frequency': 10_000,
                #'test_frequency': 50_000,
                #'save_model_frequency': 100_000,
                #'save_model_frequency': 250_000,
                'save_model_frequency': 50_000,
                'verbose': True,
            },
            #trial_id='checkpointtest',
            checkpoint_frequency=50_000,
            #checkpoint_frequency=250_000,
            #checkpoint_frequency=1_000_000,
            max_iterations=5_000_000,
            #max_iterations=50_000_000,
            verbose=True,
    )
    exp_runner.exp.logger.init_wandb({
        'project': 'DQN-%s' % env_name
    })
    try:
        exp_runner.run()
    except KeyboardInterrupt:
        breakpoint()
