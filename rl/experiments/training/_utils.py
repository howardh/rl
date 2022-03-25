import copy
from typing import Sequence, Iterable, Tuple, Mapping, Union

import torch
import numpy as np
import gym
import gym.spaces
from gym.wrappers import FrameStack#, AtariPreprocessing
from gym.spaces import Box
import cv2
from permutation import Permutation

import rl.debug_tools.frozenlake
from rl.utils import get_env_state, set_env_state

def make_env(env_name: str,
        config={},
        atari=False,
        one_hot_obs=False,
        atari_config={},
        frame_stack=4,
        episode_stack=None,
        dict_obs=False,
        action_shuffle=False):
    env = gym.make(env_name, **config)
    if atari:
        env = AtariPreprocessing(env, **atari_config)
        env = FrameStack(env, frame_stack)
    if one_hot_obs:
        env = rl.debug_tools.frozenlake.OnehotObs(env)
    if episode_stack is not None:
        env = EpisodeStack(env, episode_stack, dict_obs=dict_obs)
    elif dict_obs:
        raise Exception('dict_obs requires episode_stack')
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


class EpisodeStack(gym.Wrapper):
    def __init__(self, env, num_episodes : int, dict_obs: bool = False):
        super().__init__(env)
        self.num_episodes = num_episodes
        self.episode_count = 0
        self.dict_obs = dict_obs
        self._done = True

        if dict_obs:
            self.observation_space = gym.spaces.Dict([
                ('reward', gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)),
                ('done', gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)),
                ('obs', self.env.observation_space),
                ('action', self.env.action_space),
            ])

    def step(self, action):
        if self._done:
            self.episode_count += 1
            self._done = False
            obs, reward, done, info = self.env.reset(), 0, False, {}
        else:
            obs, reward, done, info = self.env.step(action)
        self._done = done

        if self.dict_obs:
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
        if self.dict_obs:
            return {
                'reward': np.array([0], dtype=np.float32),
                'done': np.array([False], dtype=np.float32),
                'obs': self.env.reset(**kwargs),
                'action': self.env.action_space.sample(),
            }
        else:
            return self.env.reset(**kwargs)

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
