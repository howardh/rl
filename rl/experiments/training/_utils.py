import os
import copy
from typing import Sequence, Iterable, Tuple, Mapping, Union
import threading
import time

import torch
import numpy as np
import gym
import gym.spaces
from gym.wrappers import FrameStack#, AtariPreprocessing
from gym.spaces import Box
import cv2
from permutation import Permutation
import gym_minigrid.wrappers
import gym_minigrid.minigrid
import gym_minigrid.register

import rl.debug_tools.frozenlake
from rl.utils import get_env_state, set_env_state


def make_env(env_name: str,
        config={},
        atari=False,
        atari_config={},
        minigrid=False,
        minigrid_config={},
        meta_config=None,
        one_hot_obs=False,
        frame_stack=4,
        episode_stack=None,
        dict_obs=False,
        action_shuffle=False) -> gym.Env:
    env = gym.make(env_name, **config)
    if atari:
        env = AtariPreprocessing(env, **atari_config)
        env = FrameStack(env, frame_stack)
    if minigrid:
        env = MinigridPreprocessing(env, **minigrid_config)
    if one_hot_obs:
        env = rl.debug_tools.frozenlake.OnehotObs(env)
    if episode_stack is not None:
        env = EpisodeStack(env, episode_stack, dict_obs=dict_obs)
    elif dict_obs:
        raise Exception('dict_obs requires episode_stack')
    if meta_config is not None:
        env = MetaWrapper(env, **meta_config)
    if action_shuffle:
        env = ActionShuffle(env)
    return env


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


class MinigridPreprocessing(gym.Wrapper):
    def __init__(
        self,
        env,
        rgb = True,
        screen_size = None,
    ):
        super().__init__(env)
        assert ( cv2 is not None ), 'opencv-python package not installed!'

        if rgb:
            self.env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
            self.observation_space['image'] = Box(
                    low=0, high=255,
                    shape=(
                        self.env.observation_space['image'].shape[2],
                        self.env.observation_space['image'].shape[0],
                        self.env.observation_space['image'].shape[1]),
                    dtype=np.uint8)

        self.screen_size = screen_size
        self.rgb = rgb
    
    def _resize_obs(self, obs):
        if not self.rgb:
            return obs

        # Resize
        if self.screen_size is not None:
            obs['image'] = cv2.resize( # type: ignore
                obs['image'],
                (self.screen_size, self.screen_size),
                interpolation=cv2.INTER_AREA, # type: ignore
            )

        # Move channel dimension to start
        obs['image'] = np.moveaxis(obs['image'], 2, 0)

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._resize_obs(obs)
        ## XXX: Check for reward permutation (until bug is mixed in minigrid)
        #if self.env.unwrapped.include_reward_permutation:
        #    obs['reward_permutation'] = self.env.unwrapped.reward_permutation
        return obs, reward, done, info

    def reset(self, **kwargs):
        # NoopReset
        obs = self.env.reset(**kwargs)
        obs = self._resize_obs(obs)
        ## XXX: Check for reward permutation (until bug is mixed in minigrid)
        #if self.env.unwrapped.include_reward_permutation:
        #    obs['reward_permutation'] = self.env.unwrapped.reward_permutation
        return obs


class EpisodeStack(gym.Wrapper):
    def __init__(self, env, num_episodes : int, dict_obs: bool = False):
        super().__init__(env)
        self.num_episodes = num_episodes
        self.episode_count = 0
        self.dict_obs = dict_obs
        self._done = True

        if dict_obs:
            obs_space = [('obs', self.env.observation_space)]
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                obs_space = [(f'obs ({k})',v) for k,v in self.env.observation_space.items()]
            self.observation_space = gym.spaces.Dict([
                ('reward', gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)),
                ('done', gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)),
                #('obs', self.env.observation_space),
                *obs_space,
                ('action', self.env.action_space),
            ])

    def step(self, action):
        if self._done:
            if isinstance(self.env.unwrapped, NRoomBanditsSmall):
                self.env.unwrapped.shuffle_goals_on_reset = False
            self.episode_count += 1
            self._done = False
            obs, reward, done, info = self.env.reset(), 0, False, {}
        else:
            obs, reward, done, info = self.env.step(action)
        self._done = done

        if self.dict_obs:
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                obs = {
                    'reward': np.array([reward], dtype=np.float32),
                    'done': np.array([done], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': action,
                }
            else:
                obs = {
                    'reward': np.array([reward], dtype=np.float32),
                    'done': np.array([done], dtype=np.float32),
                    'obs': obs,
                    'action': action,
                }

        if done:
            if self.episode_count >= self.num_episodes:
                self.episode_count = 0
                return obs, reward, done, info
            else:
                return obs, reward, False, info
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.episode_count = 0
        if isinstance(self.env.unwrapped, NRoomBanditsSmall):
            self.env.unwrapped.shuffle_goals_on_reset = True
        obs = self.env.reset(**kwargs)
        if self.dict_obs:
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                return {
                    'reward': np.array([0], dtype=np.float32),
                    'done': np.array([False], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': self.env.action_space.sample(),
                }
            else:
                return {
                    'reward': np.array([0], dtype=np.float32),
                    'done': np.array([False], dtype=np.float32),
                    'obs': obs,
                    'action': self.env.action_space.sample(),
                }
        else:
            return obs

    def state_dict(self):
        return {
            'episode_count': self.episode_count,
            'env': get_env_state(self.env),
            'dict_obs': self.dict_obs,
            '_done': self._done,
        }

    def load_state_dict(self, state_dict):
        self.episode_count = state_dict['episode_count']
        set_env_state(self.env, state_dict['env'])
        self.dict_obs = state_dict['dict_obs']
        self._done = state_dict['_done']


class MetaWrapper(gym.Wrapper):
    """
    Wrapper for meta-RL.

    Features:
    - Converting observations to dict
    - Adding reward, termination signal, and previous action to observations
    - Stacking episodes
    - Randomizing the environment between trials (requires the environment to have a `randomize()` method)
    """
    def __init__(self,
            env,
            episode_stack: int,
            dict_obs: bool = False,
            randomize: bool = True,
            action_shuffle: bool = False):
        super().__init__(env)
        self.episode_stack = episode_stack
        self.randomize = randomize
        self.dict_obs = dict_obs
        self.action_shuffle = action_shuffle

        self.episode_count = 0
        self._done = True

        if action_shuffle:
            raise NotImplementedError('Action shuffle not implemented')

        if dict_obs:
            obs_space = [('obs', self.env.observation_space)]
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                obs_space = [(f'obs ({k})',v) for k,v in self.env.observation_space.items()]
            self.observation_space = gym.spaces.Dict([
                ('reward', gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)),
                ('done', gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)),
                *obs_space,
                ('action', self.env.action_space),
            ])

    def step(self, action):
        if self._done:
            if self.episode_count == 0 and self.randomize:
                self.env.randomize()
            self.episode_count += 1
            self._done = False
            obs, reward, done, info = self.env.reset(), 0, False, {}
        else:
            obs, reward, done, info = self.env.step(action)
        self._done = done

        if self.dict_obs:
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                obs = {
                    'reward': np.array([reward], dtype=np.float32),
                    'done': np.array([done], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': action,
                }
            else:
                obs = {
                    'reward': np.array([reward], dtype=np.float32),
                    'done': np.array([done], dtype=np.float32),
                    'obs': obs,
                    'action': action,
                }

        if done:
            if 'expected_return' in info and 'max_return' in info:
                regret = info['max_return'] - info['expected_return']
                self._regret.append(regret)
                info['regret'] = self._regret

            if self.episode_count >= self.episode_stack: # Episode count starts at 1
                self.episode_count = 0
                return obs, reward, done, info
            else:
                return obs, reward, False, info
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        self.episode_count = 0
        if self.randomize:
            self.env.randomize()

        self._regret = []

        obs = self.env.reset(*args, **kwargs)
        if self.dict_obs:
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                return {
                    'reward': np.array([0], dtype=np.float32),
                    'done': np.array([False], dtype=np.float32),
                    **{f'obs ({k})': v for k,v in obs.items()},
                    'action': self.env.action_space.sample(),
                }
            else:
                return {
                    'reward': np.array([0], dtype=np.float32),
                    'done': np.array([False], dtype=np.float32),
                    'obs': obs,
                    'action': self.env.action_space.sample(),
                }
        else:
            return obs

    def state_dict(self):
        return {
            'episode_count': self.episode_count,
            'env': get_env_state(self.env),
            'dict_obs': self.dict_obs,
            '_done': self._done,
        }

    def load_state_dict(self, state_dict):
        self.episode_count = state_dict['episode_count']
        set_env_state(self.env, state_dict['env'])
        self.dict_obs = state_dict['dict_obs']
        self._done = state_dict['_done']


class ActionShuffle(gym.Wrapper):
    def __init__(self, env, actions=None, permutation=None):
        """
        Args:
            env: gym.Env
            actions: list of ints, indices of actions to shuffle. Alternatively, if set to True, then all actions are shuffled.
            permutation: list of ints, indices of the permutation to use. If not set, then a new permutation is randomly generated at the start of each episode.
        """
        super().__init__(env)
        if actions is None:
            self._actions = list(range(self.env.action_space.n))
        else:
            self._actions = actions

        if isinstance(permutation, list):
            self.permutation = permutation
        elif isinstance(permutation, int):
            self.permutation = Permutation.from_lehmer(permutation, len(self._actions)).to_image()
        else:
            self.permutation = None

        self._init_mapping()

    def _init_mapping(self):
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.mapping = np.arange(self.env.action_space.n)
            for i,j in enumerate(np.random.permutation(len(self._actions))):
                self.mapping[self._actions[i]] = self._actions[j]
        else:
            raise ValueError("ActionShuffle only supports Discrete action spaces")

    def reset(self, **kwargs):
        self._init_mapping()
        return self.env.reset(**kwargs)

    def step(self, action):
        action = self.mapping[action]
        return self.env.step(action)

    def state_dict(self):
        return {
            'mapping': self.mapping,
            'env': get_env_state(self.env),
        }

    def load_state_dict(self, state_dict):
        self.mapping = state_dict['mapping']
        set_env_state(self.env, state_dict['env'])


def merge(source, destination):
    """
    (Source: https://stackoverflow.com/a/20666342/382388)

    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    destination = copy.deepcopy(destination)
    if isinstance(source, Mapping):
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                destination[key] = merge(value, node)
            elif isinstance(value, list):
                if isinstance(destination[key],list):
                    destination[key] = [merge(s,d) for s,d in zip(source[key],destination[key])]
                else:
                    destination[key] = value
            else:
                destination[key] = value

        return destination
    else:
        return source


def zip2(*args) -> Iterable[Union[Tuple,Mapping]]:
    """
    Zip objects together. If dictionaries are provided, the lists within the dictionary are zipped together.

    >>> list(zip2([1,2,3], [4,5,6]))
    [(1, 4), (2, 5), (3, 6)]

    >>> list(zip2({'a': [4,5,6], 'b': [7,8,9]}))
    [{'a': 4, 'b': 7}, {'a': 5, 'b': 8}, {'a': 6, 'b': 9}]

    >>> list(zip2([1,2,3], {'a': [4,5,6], 'b': [7,8,9]}))
    [(1, {'a': 4, 'b': 7}), (2, {'a': 5, 'b': 8}), (3, {'a': 6, 'b': 9})]

    >>> import torch
    >>> list(zip2(torch.tensor([1,2,3]), torch.tensor([4,5,6])))
    [(tensor(1), tensor(4)), (tensor(2), tensor(5)), (tensor(3), tensor(6))]
    """
    if len(args) == 1:
        if isinstance(args[0],(Sequence)):
            return args[0]
        if isinstance(args[0],torch.Tensor):
            return (x for x in args[0])
        if isinstance(args[0], dict):
            keys = args[0].keys()
            return (dict(zip(keys, vals)) for vals in zip(*(args[0][k] for k in keys)))
    return zip(*[zip2(a) for a in args])


class ExperimentConfigs(dict):
    def __init__(self):
        self._last_key = None
    def add(self, key, config, inherit=None):
        if key in self:
            raise Exception(f'Key {key} already exists.')
        if inherit is None:
            self[key] = config
        else:
            self[key] = merge(config,self[inherit])
        self._last_key = key
    def add_change(self, key, config):
        self.add(key, config, inherit=self._last_key)


class GoalDeterministic(gym_minigrid.minigrid.Goal):
    def __init__(self, reward):
        super().__init__()
        self.reward = reward


class GoalMultinomial(gym_minigrid.minigrid.Goal):
    def __init__(self, rewards, probs):
        super().__init__()
        self.rewards = rewards
        self.probs = probs

    def sample_reward(self):
        return self.rewards[np.random.choice(len(self.rewards), p=self.probs)]

    @property
    def expected_value(self):
        return (np.array(self.rewards, dtype=np.float32) * np.array(self.probs, dtype=np.float32)).sum() # type: ignore


class NRoomBanditsSmall(gym_minigrid.minigrid.MiniGridEnv):
    def __init__(self, rewards=[-1,1], shuffle_goals_on_reset=True, include_reward_permutation=False, seed=None):
        self.mission = 'Reach the goal with the highest reward.'
        self.rewards = rewards
        self.goals = [
            GoalDeterministic(reward=r) for r in self.rewards
        ]
        self.shuffle_goals_on_reset = shuffle_goals_on_reset
        self.include_reward_permutation = include_reward_permutation

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)

        super().__init__(width=5, height=5, seed=seed)

        if include_reward_permutation:
            self.observation_space = gym.spaces.Dict({
                **self.observation_space.spaces,
                'reward_permutation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.rewards),), dtype=np.float32),
            })

    @property
    def reward_permutation(self):
        return [g.reward for g in self.goals]

    def randomize(self):
        self._shuffle_goals()

    def _shuffle_goals(self):
        reward_indices = self.np_random.permutation(len(self.rewards))
        for g,i in zip(self.goals,reward_indices):
            g.reward = self.rewards[i]

    def _gen_grid(self, width, height):
        self.grid = gym_minigrid.minigrid.Grid(width, height)

        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.agent_pos = (2, height-2)
        self.agent_dir = self._rand_int(0, 4)

        if self.shuffle_goals_on_reset:
            self._shuffle_goals()

        for i,g in enumerate(self.goals):
            self.put_obj(g, 1+i*2, 1)

    def _reward(self):
        curr_cell = self.grid.get(*self.agent_pos) # type: ignore (where is self.grid assigned?)
        if curr_cell != None and hasattr(curr_cell,'reward'):
            return curr_cell.reward
        breakpoint()
        return 0

    def reset(self):
        obs = super().reset()
        if self.include_reward_permutation:
            assert isinstance(obs, dict)
            obs['reward_permutation'] = self.reward_permutation
        if self.shuffle_goals_on_reset:
            self._shuffle_goals()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['reward_permutation'] = self.reward_permutation
        if self.include_reward_permutation:
            obs['reward_permutation'] = info['reward_permutation']
        return obs, reward, done, info


gym_minigrid.register.register(
    id='MiniGrid-NRoomBanditsSmall-v0',
    entry_point=NRoomBanditsSmall
)


class NRoomBanditsSmallBernoulli(gym_minigrid.minigrid.MiniGridEnv):
    def __init__(self, reward_scale=1, prob=0.9, shuffle_goals_on_reset=True, include_reward_permutation=False, seed=None):
        self.mission = 'Reach the goal with the highest reward.'
        self.reward_scale = reward_scale
        self.prob = prob
        self.goals = [
            GoalMultinomial(rewards=[reward_scale,-reward_scale], probs=[prob,1-prob]),
            GoalMultinomial(rewards=[reward_scale,-reward_scale], probs=[1-prob,prob]),
        ]
        self.shuffle_goals_on_reset = shuffle_goals_on_reset
        self.include_reward_permutation = include_reward_permutation

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)

        super().__init__(width=5, height=5, seed=seed)

        if include_reward_permutation:
            self.observation_space = gym.spaces.Dict({
                **self.observation_space.spaces,
                'reward_permutation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            })

        # Info
        self._expected_return = 0

    @property
    def reward_permutation(self):
        return [g.expected_value for g in self.goals]

    def randomize(self):
        self._shuffle_goals()

    def _shuffle_goals(self):
        permutation = self.np_random.permutation(2)
        probs = [
                [self.prob, 1-self.prob],
                [1-self.prob, self.prob],
        ]
        for g,i in zip(self.goals,permutation):
            g.probs = probs[i]

    def _gen_grid(self, width, height):
        self.grid = gym_minigrid.minigrid.Grid(width, height)

        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.agent_pos = (2, height-2)
        self.agent_dir = self._rand_int(0, 4)

        if self.shuffle_goals_on_reset:
            self._shuffle_goals()

        for i,g in enumerate(self.goals):
            self.put_obj(g, 1+i*2, 1)

    def _reward(self):
        curr_cell = self.grid.get(*self.agent_pos) # type: ignore
        if curr_cell != None and hasattr(curr_cell,'rewards') and hasattr(curr_cell,'probs'):
            return self.np_random.choice(curr_cell.rewards, p=curr_cell.probs)
        breakpoint()
        return 0

    def _expected_reward(self):
        """ Expected reward of the current state """
        curr_cell = self.grid.get(*self.agent_pos) # type: ignore (where is self.grid assigned?)
        if curr_cell is None:
            return 0
        if hasattr(curr_cell,'expected_value'):
            return curr_cell.expected_value
        return 0
    
    @property
    def max_return(self):
        """ Expected return (undiscounted sum) of the optimal policy """
        return max(self.reward_permutation)

    def reset(self):
        obs = super().reset()
        if self.include_reward_permutation:
            assert isinstance(obs, dict)
            obs['reward_permutation'] = self.reward_permutation
        if self.shuffle_goals_on_reset:
            self._shuffle_goals()

        self._expected_return = 0

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info['reward_permutation'] = self.reward_permutation
        if self.include_reward_permutation:
            obs['reward_permutation'] = info['reward_permutation']
        self._expected_return += self._expected_reward()
        info['expected_return'] = self._expected_return
        info['max_return'] = self.max_return
        return obs, reward, done, info


gym_minigrid.register.register(
    id='MiniGrid-NRoomBanditsSmallBernoulli-v0',
    entry_point=NRoomBanditsSmallBernoulli
)


class BanditsFetch(gym_minigrid.minigrid.MiniGridEnv):
    """
    Environment in which the agent has to fetch a random object
    named using English text strings
    """

    def __init__(
        self,
        size=8,
        num_objs=3,
        num_trials=1,
        reward_correct=1,
        reward_incorrect=-1,
        num_obj_types=2,
        num_obj_colors=6,
        unique_objs=False,
        include_reward_permutation=False,
        seed=None,
    ):
        """
        Args:
            size (int): Size of the grid
            num_objs (int): Number of objects in the environment
            num_trials (int): Number of trials to run with the same set of objects and goal
            reward_correct (int): Reward for picking up the correct object
            reward_incorrect (int): Reward for picking up the incorrect object
            num_obj_types (int): Number of possible object types
            num_obj_colors (int): Number of possible object colors
            unique_objs (bool): If True, each object is unique
            include_reward_permutation (bool): If True, include the reward permutation in the observation
        """
        self.numObjs = num_objs

        self.num_trials = num_trials
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.num_obj_types = num_obj_types
        self.num_obj_colors = num_obj_colors
        self.unique_objs = unique_objs
        self.include_reward_permutation = include_reward_permutation

        self.types = ['key', 'ball']
        self.colors = gym_minigrid.minigrid.COLOR_NAMES

        self.types = self.types[:self.num_obj_types]
        self.colors = self.colors[:self.num_obj_colors]

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)

        super().__init__(
            grid_size=size,
            max_steps=5*size**2*num_trials,
            # Set this to True for maximum speed
            see_through_walls=True,
            seed=seed,
        )

        self.trial_count = 0
        self.objects = []

        if include_reward_permutation:
            self.observation_space = gym.spaces.Dict({
                **self.observation_space.spaces,
                'reward_permutation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.types)*len(self.colors),), dtype=np.float32),
            })

    def _gen_grid(self, width, height):
        self.grid = gym_minigrid.minigrid.Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width-1, 0)

        types = self.types
        colors = self.colors

        type_color_pairs = [(t,c) for t in types for c in colors]

        objs = []

        # For each object to be generated
        while len(objs) < self.numObjs:
            objType, objColor = self._rand_elem(type_color_pairs)
            if self.unique_objs:
                type_color_pairs.remove((objType, objColor))

            if objType == 'key':
                obj = gym_minigrid.minigrid.Key(objColor)
            elif objType == 'ball':
                obj = gym_minigrid.minigrid.Ball(objColor)
            else:
                raise ValueError(f'Unknown object type: {objType}')

            self.place_obj(obj)
            objs.append(obj)

        self.objects = objs

        # Randomize the player start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

        descStr = '%s %s' % (self.targetColor, self.targetType)

        # Generate the mission string
        idx = self._rand_int(0, 5)
        if idx == 0:
            self.mission = 'get a %s' % descStr
        elif idx == 1:
            self.mission = 'go get a %s' % descStr
        elif idx == 2:
            self.mission = 'fetch a %s' % descStr
        elif idx == 3:
            self.mission = 'go fetch a %s' % descStr
        elif idx == 4:
            self.mission = 'you must fetch a %s' % descStr
        assert hasattr(self, 'mission')

    def reset(self):
        obs = super().reset()
        if self.include_reward_permutation:
            assert isinstance(obs, dict)
            obs['reward_permutation'] = self.reward_permutation
        return obs

    def step(self, action):
        obs, reward, done, info = gym_minigrid.minigrid.MiniGridEnv.step(self, action)

        if self.carrying:
            if self.carrying.color == self.targetColor and \
               self.carrying.type == self.targetType:
                reward = self.reward_correct
            else:
                reward = self.reward_incorrect
            self.place_obj(self.carrying)
            self.carrying = None
            self.trial_count += 1
            if self.trial_count >= self.num_trials:
                done = True
                self.trial_count = 0

        if self.include_reward_permutation:
            assert isinstance(obs, dict)
            obs['reward_permutation'] = self.reward_permutation

        return obs, reward, done, info

    @property
    def reward_permutation(self):
        r = [self.reward_incorrect, self.reward_correct]
        return [
            r[t == self.targetType and c == self.targetColor]
            for t in self.types for c in self.colors
        ]


gym_minigrid.register.register(
    id='MiniGrid-BanditsFetch-v0',
    entry_point=BanditsFetch,
)


class MultiRoomEnv(gym_minigrid.minigrid.MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        min_num_rooms,
        max_num_rooms,
        max_room_size=10,
        num_trials=100,
        seed = None,
    ):
        assert min_num_rooms > 0
        assert max_num_rooms >= min_num_rooms
        assert max_room_size >= 4

        self.minNumRooms = min_num_rooms
        self.maxNumRooms = max_num_rooms
        self.maxRoomSize = max_room_size

        self.num_trials = num_trials

        self.rooms = []

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)

        super(MultiRoomEnv, self).__init__(
            grid_size=25,
            max_steps=self.maxNumRooms * 20 * self.num_trials,
            seed = seed,
        )

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = gym_minigrid.minigrid.Grid(width, height)
        wall = gym_minigrid.minigrid.Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(gym_minigrid.minigrid.COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = gym_minigrid.minigrid.Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor) # type: ignore
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal = gym_minigrid.minigrid.Goal()
        self.goal_pos = self.place_obj(self.goal, roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height: # type: ignore
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(gym_minigrid.envs.multiroom.Room( # type: ignore
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for _ in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    def reset(self):
        obs = super().reset()
        self.trial_count = 0
        return obs

    def step(self, action):
        obs, reward, done, info = gym_minigrid.minigrid.MiniGridEnv.step(self, action)

        if done:
            self.trial_count += 1
            if reward > 0:
                reward = 1
            self.grid.set(self.goal_pos[0], self.goal_pos[1], None)
            r = self._rand_int(0, len(self.rooms) - 1)
            self.goal_pos = self.place_obj(self.goal, self.rooms[r].top, self.rooms[r].size)
            if self.trial_count >= self.num_trials:
                done = True
                self.trial_count = 0
            else:
                done = False
            if self.step_count >= self.max_steps:
                done = True

        return obs, reward, done, info


gym_minigrid.register.register(
    id='MiniGrid-MultiRoom-v0',
    entry_point=MultiRoomEnv,
)


class Room:
    __slots__ = ['top', 'bottom', 'left', 'right']

    def __init__(self,
            top: int,
            bottom: int,
            left: int,
            right: int,
    ):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def __repr__(self):
        return 'Room(top={}, bottom={}, left={}, right={})'.format(
            self.top,
            self.bottom,
            self.left,
            self.right,
        )

def room_is_valid(rooms, room, width, height):
    if room.left < 0 or room.right >= width or room.top < 0 or room.bottom >= height:
        return False
    for r in rooms:
        if room.top >= r.bottom:
            continue
        if room.bottom <= r.top:
            continue
        if room.left >= r.right:
            continue
        if room.right <= r.left:
            continue
        return False
    return True



class MultiRoomEnv_v1(gym_minigrid.minigrid.MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        min_num_rooms,
        max_num_rooms,
        min_room_size=5,
        max_room_size=10,
        num_trials=100,
        fetch_config: dict = None,
        bandits_config: dict = None,
        seed = None,
    ):
        assert min_num_rooms > 0
        assert max_num_rooms >= min_num_rooms
        assert max_room_size >= 4

        self.min_num_rooms = min_num_rooms
        self.max_num_rooms = max_num_rooms
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size

        self.num_trials = num_trials
        self.fetch_config = fetch_config
        self.bandits_config = bandits_config

        self.rooms = []

        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
            seed = seed % (2**32 - 1)

        super(MultiRoomEnv_v1, self).__init__(
            grid_size=25,
            max_steps=self.max_num_rooms * 20 * self.num_trials,
            see_through_walls = False,
            seed = seed,
        )

    def _gen_grid(self, width, height):
        room_list = []
        self.rooms = room_list

        # Choose a random number of rooms to generate
        num_rooms = self._rand_int(self.min_num_rooms, self.max_num_rooms+1)

        # Create first room
        room_height = self._rand_int(self.min_room_size, self.max_room_size+1)
        room_width = self._rand_int(self.min_room_size, self.max_room_size+1)
        top = self._rand_int(1, height - room_height - 1)
        left = self._rand_int(1, width - room_width - 1)
        room_list.append(Room(top, top + room_height - 1, left, left + room_width - 1))

        new_room_openings = [ (0, 'left'), (0, 'right'), (0, 'top'), (0, 'bottom') ]
        while len(room_list) < num_rooms:
            if len(new_room_openings) == 0:
                break

            # Choose a random place to connect the new room to
            r = self._rand_int(0, len(new_room_openings))
            starting_room_index, wall = new_room_openings[r]

            temp_room = self._generate_room(
                    room_list,
                    idx = starting_room_index,
                    wall = wall,
                    min_size = self.min_room_size,
                    max_size = self.max_room_size,
                    width = width,
                    height = height,
            )
            if temp_room is not None:
                room_list.append(temp_room)
                new_room_openings.append((len(room_list)-1, 'left'))
                new_room_openings.append((len(room_list)-1, 'right'))
                new_room_openings.append((len(room_list)-1, 'top'))
                new_room_openings.append((len(room_list)-1, 'bottom'))
            else:
                new_room_openings.remove(new_room_openings[r])

        self.grid = gym_minigrid.minigrid.Grid(width, height)
        self.doors = []
        wall = gym_minigrid.minigrid.Wall()
        self.wall = wall

        for room in room_list:
            # Look for overlapping walls
            overlapping_walls = {
                'top': [],
                'bottom': [],
                'left': [],
                'right': [],
            }
            for i in range(room.left + 1, room.right):
                if self.grid.get(i,room.top) == wall and self.grid.get(i,room.top+1) is None and self.grid.get(i,room.top-1) is None:
                    overlapping_walls['top'].append((room.top, i))
                if self.grid.get(i,room.bottom) == wall and self.grid.get(i,room.bottom+1) is None and self.grid.get(i,room.bottom-1) is None:
                    overlapping_walls['bottom'].append((room.bottom, i))
            for j in range(room.top + 1, room.bottom):
                if self.grid.get(room.left,j) == wall and self.grid.get(room.left+1,j) is None and self.grid.get(room.left-1,j) is None:
                    overlapping_walls['left'].append((j, room.left))
                if self.grid.get(room.right,j) == wall and self.grid.get(room.right+1,j) is None and self.grid.get(room.right-1,j) is None:
                    overlapping_walls['right'].append((j, room.right))

            # Create room
            # Top wall
            for i in range(room.left, room.right + 1):
                self.grid.set(i, room.top, wall)
            # Bottom wall
            for i in range(room.left, room.right + 1):
                self.grid.set(i, room.bottom, wall)
            # Left wall
            for i in range(room.top, room.bottom + 1):
                self.grid.set(room.left, i, wall)
            # Right wall
            for i in range(room.top, room.bottom + 1):
                self.grid.set(room.right, i, wall)

            # Create doorways between rooms
            for ow in overlapping_walls.values():
                if len(ow) == 0:
                    continue
                opening = self._rand_elem(ow)
                if self.np_random.rand() < 0.5:
                    self.grid.set(opening[1], opening[0], None)
                else:
                    door = gym_minigrid.minigrid.Door(
                        color = self._rand_elem(gym_minigrid.minigrid.COLOR_NAMES)
                    )
                    self.grid.set(opening[1], opening[0], door)
                    self.doors.append(door)

        self._init_agent()
        self.mission = 'Do whatever'

    def _init_fetch(self, num_objs, num_obj_types=2, num_obj_colors=6, unique_objs=True):
        types = ['key', 'ball'][:num_obj_types]
        colors = gym_minigrid.minigrid.COLOR_NAMES[:num_obj_colors]

        type_color_pairs = [(t,c) for t in types for c in colors]

        objs = []

        # For each object to be generated
        while len(objs) < num_objs:
            obj_type, obj_color = self._rand_elem(type_color_pairs)
            if unique_objs:
                type_color_pairs.remove((obj_type, obj_color))

            if obj_type == 'key':
                obj = gym_minigrid.minigrid.Key(obj_color)
            elif obj_type == 'ball':
                obj = gym_minigrid.minigrid.Ball(obj_color)
            else:
                raise ValueError(f'Unknown object type: {obj_type}')

            self.place_obj(obj)
            objs.append(obj)

        self.objects = objs

        # Choose a random object to be picked up
        target = objs[self._rand_int(0, len(objs))]
        self.targetType = target.type
        self.targetColor = target.color

    def _init_bandits(self, probs=[1,0]):
        reward_scale = 1
        self.goals = [
            GoalMultinomial(rewards=[reward_scale,-reward_scale], probs=[p,1-p])
            for p in probs
        ]

        for g in self.goals:
            self.place_obj(g, unobstructive=True)

    def _init_agent(self):
        # Randomize the player start position and orientation
        self.agent_pos = self._rand_space()
        self.agent_dir = self._rand_int(0, 4)

    def _rand_space(self):
        """ Find and return the coordinates of a random empty space in the grid """

        room_indices = list(range(len(self.rooms)))
        self.np_random.shuffle(room_indices) # type: ignore

        for r in room_indices:
            # Choose a random room
            room = self.rooms[r]

            # List all spaces in the room
            spaces = [
                (x, y)
                for x in range(room.left+1, room.right)
                for y in range(room.top+1, room.bottom)
            ]
            self.np_random.shuffle(spaces) # type: ignore

            # Choose a random location in the room
            for x, y in spaces:
                # Check if the location is empty
                if self.grid.get(x, y) is not None:
                    continue
                # Check if the agent is here
                if np.array_equal((x,y), self.agent_pos):
                    continue
                return x, y

        raise Exception('Could not find a random empty space')

    def _rand_space_unobstructive(self):
        """ Find and return the coordinates of a random empty space in the grid.
        This space is chosen from a set of spaces that would not obstruct access to other parts of the environment if an object were to be placed there.
        """

        room_indices = list(range(len(self.rooms)))
        self.np_random.shuffle(room_indices) # type: ignore

        for r in room_indices:
            # Choose a random room
            room = self.rooms[r]

            # List all spaces in the room
            spaces = [
                (x, y)
                for x in range(room.left+1, room.right)
                for y in range(room.top+1, room.bottom)
            ]
            self.np_random.shuffle(spaces) # type: ignore

            # Choose a random location in the room
            for x, y in spaces:
                # Check if the location is empty
                if self.grid.get(x, y) is not None:
                    continue
                # Check if the agent is here
                if np.array_equal((x,y), self.agent_pos):
                    continue
                # Check if it blocks a doorway
                obstructive = False
                for d in [[0,1],[1,0],[0,-1],[-1,0]]:
                    if self.grid.get(x+d[0], y+d[1]) is self.wall:
                        continue
                    c1 = [d[1]+d[0],d[1]+d[0]]
                    c2 = [-d[1]+d[0],d[1]-d[0]]
                    if self.grid.get(x+c1[0], y+c1[1]) is self.wall and self.grid.get(x+c2[0], y+c2[1]) is self.wall:
                        obstructive = True
                        break

                if obstructive:
                    continue

                return x, y

        raise Exception('Could not find a random empty space')

    def place_obj(self, obj, unobstructive=False):
        if unobstructive:
            pos = self._rand_space_unobstructive()
        else:
            pos = self._rand_space()

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def _generate_room(self, rooms, idx, wall, min_size, max_size, width, height):
        starting_room = rooms[idx]
        new_room = Room(0,0,0,0)
        if wall == 'left' or wall == 'right':
            min_top = max(starting_room.top - max_size + 3, 0)
            max_top = starting_room.bottom - 2
            min_bottom = starting_room.top + 2
            max_bottom = starting_room.bottom + max_size - 3
            if wall == 'left':
                #new_room.right = starting_room.left
                min_right = starting_room.left
                max_right = starting_room.left
                min_left = max(starting_room.left - max_size + 1, 0)
                max_left = starting_room.left - min_size + 1
            else:
                #new_room.left = starting_room.right
                min_left = starting_room.right
                max_left = starting_room.right
                min_right = starting_room.right + min_size - 1
                max_right = starting_room.right + max_size - 1
        else:
            min_left = max(starting_room.left - max_size + 3, 0)
            max_left = starting_room.right - 2
            min_right = starting_room.left + 2
            max_right = starting_room.right + max_size - 3
            if wall == 'top':
                #new_room.bottom = starting_room.top
                min_bottom = starting_room.top
                max_bottom = starting_room.top
                min_top = max(starting_room.top - max_size + 1, 0)
                max_top = starting_room.top - min_size + 1
            else:
                #new_room.top = starting_room.bottom
                min_top = starting_room.bottom
                max_top = starting_room.bottom
                min_bottom = starting_room.bottom + min_size - 1
                max_bottom = starting_room.bottom + max_size - 1
        possible_rooms = [
            (t,b,l,r)
            for t in range(min_top, max_top + 1)
            for b in range(max(min_bottom,t+min_size-1), min(max_bottom + 1, t+max_size))
            for l in range(min_left, max_left + 1)
            for r in range(max(min_right,l+min_size-1), min(max_right + 1, l+max_size))
        ]
        self.np_random.shuffle(possible_rooms)
        for room in possible_rooms:
            new_room.top = room[0]
            new_room.bottom = room[1]
            new_room.left = room[2]
            new_room.right = room[3]
            if room_is_valid(rooms, new_room, width, height):
                return new_room
        return None

    def reset(self):
        obs = super().reset()
        self.trial_count = 0
        if self.fetch_config is not None:
            self._init_fetch(**self.fetch_config)
        if self.bandits_config is not None:
            self._init_bandits(**self.bandits_config)
        return obs

    def step(self, action):
        obs, reward, done, info = gym_minigrid.minigrid.MiniGridEnv.step(self, action)

        if self.fetch_config is not None:
            if self.carrying:
                if self.carrying.color == self.targetColor and \
                   self.carrying.type == self.targetType:
                    reward = self.fetch_config.get('reward_correct', 1)
                else:
                    reward = self.fetch_config.get('reward_incorrect', -1)
                self.place_obj(self.carrying)
                self.carrying = None
                self.trial_count += 1

        if self.bandits_config is not None:
            curr_cell = self.grid.get(*self.agent_pos) # type: ignore
            if curr_cell != None and hasattr(curr_cell,'rewards') and hasattr(curr_cell,'probs'):
                # Give a reward
                reward = self.np_random.choice(curr_cell.rewards, p=curr_cell.probs)
                done = False
                self.trial_count += 1
                # Teleport the agent to a random location
                self._init_agent()

        if self.trial_count >= self.num_trials:
            done = True
            self.trial_count = 0

        return obs, reward, done, info


gym_minigrid.register.register(
    id='MiniGrid-MultiRoom-v1',
    entry_point=MultiRoomEnv_v1,
)

class MultiRoomBanditsLarge(MultiRoomEnv_v1):
    def __init__(self):
        super().__init__(min_num_rooms=5, max_num_rooms=5, max_room_size=16, fetch_config={'num_objs': 5}, bandits_config={})

gym_minigrid.register.register(
    id='MiniGrid-MultiRoom-Large-v1',
    entry_point=MultiRoomBanditsLarge,
)

if __name__ == '__main__':
    env = gym.make('MiniGrid-MultiRoom-v1',
            min_num_rooms=2,
            max_num_rooms=2,
            min_room_size=4,
            max_room_size=6,
            #fetch_config={'num_objs': 5},
            bandits_config={
                'probs': [0.9, 0.1]
            },
            #seed=2349918951,
    )
    env.reset()
    env.render()
    breakpoint()
