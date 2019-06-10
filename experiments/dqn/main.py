from collections import deque
import numpy as np
import gym
from gym.spaces import Box
from gym.wrappers import TimeLimit
import torch
from tqdm import tqdm

from agent.dqn_agent import DQNAgent
from agent.dqn_agent import get_greedy_epsilon_policy

class AtariPreprocessing(gym.Wrapper):
    r"""Atari 2600 preprocessings. 

    This class follows the guidelines in 
    Machado et al. (2018), "Revisiting the Arcade Learning Environment: 
    Evaluation Protocols and Open Problems for General Agents".

    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on reset. 
    * FireReset: take action on reset for environments that are fixed until firing. 
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional

    Args:
        env (Env): environment
        noop_max (int): max number of no-ops
        frame_skip (int): the frequency at which the agent experiences the game. 
        screen_size (int): resize Atari frame
        terminal_on_life_loss (bool): if True, then step() returns done=True whenever a
            life is lost. 
        grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
            is returned.

    """
    def __init__(self, env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True):
        super().__init__(env)
        assert frame_skip > 0
        assert screen_size > 0

        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs

        # buffer of most recent two observations for max pooling
        if grayscale_obs:
            self.obs_buffer = [np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                               np.empty(env.observation_space.shape[:2], dtype=np.uint8)]
        else:
            self.obs_buffer = [np.empty(env.observation_space.shape, dtype=np.uint8),
                               np.empty(env.observation_space.shape, dtype=np.uint8)]

        self.ale = env.unwrapped.ale
        self.lives = 0
        self.game_over = False

        if grayscale_obs:
            self.observation_space = Box(low=0, high=255, shape=(screen_size, screen_size), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=255, shape=(screen_size, screen_size, 3), dtype=np.uint8)

    def step(self, action):
        R = 0.0

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
                    self.ale.getScreenGrayscale(self.obs_buffer[0])
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[0])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[1])    
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[1])    
        return self._get_obs(), R, done, info

    def reset(self, **kwargs):
        # NoopReset
        self.env.reset(**kwargs)
        noops = self.env.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset(**kwargs)

        # FireReset
        action_meanings = self.env.unwrapped.get_action_meanings()
        if action_meanings[1] == 'FIRE' and len(action_meanings) >= 3:
            self.env.step(1)
            self.env.step(2)

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB2(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)
        return self._get_obs()

    def _get_obs(self):
        import cv2
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = cv2.resize(self.obs_buffer[0], (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        obs = np.asarray(obs, dtype=np.uint8)
        return obs

class LazyFrames(object):
    r"""Ensures common frames are only stored once to optimize memory use. 
    To further reduce the memory use, it is optionally to turn on lz4 to 
    compress the observations.
    .. note::
        This object should only be converted to numpy array just before forward pass. 
    """
    def __init__(self, frames, lz4_compress=False):
        if lz4_compress:
            self.shape = frames[0].shape
            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        if self.lz4_compress:
            frames = [np.frombuffer(decompress(frame), dtype=np.uint8).reshape(self.shape) for frame in self._frames]
        else:
            frames = self._frames
        out = np.stack(frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.__array__())

    def __getitem__(self, i):
        return self.__array__()[i]

class FrameStack(gym.ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner. 
    For example, if the number os stacks is 4, then returned observation constains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [3, 4]. 
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first. 
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
    """
    def __init__(self, env, num_stack, lz4_compress=False):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def _get_observation(self):
        assert len(self.frames) == self.num_stack
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation()

# TODO: Wrappers above are taken from pending pull requests on the OpenAI Gym library. Remove and use the gym library implementation when the pull request goes through.

def run_trial(gamma, alpha, eps_b, eps_t, directory=None,
        max_iters=5000, epoch=50, test_iters=1):
    args = locals()
    env_name = 'Breakout-v0'
    env = gym.make(env_name)
    env = AtariPreprocessing(env)
    env = FrameStack(env,4)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=alpha,
            discount_factor=gamma,
            device=device
    )
    #agent.set_behaviour_policy("%.3f-epsilon"%eps_b)
    #agent.set_target_policy("%.3f-epsilon"%eps_t)

    rewards = []
    try:
        for iters in range(0,max_iters+1):
            # Run tests
            if iters % epoch == 0:
                r = agent.test(env, test_iters, render=False, processors=1)
                rewards.append(r)
                print('iter %d \t Reward: %f' % (iters, np.mean(r)))
            # Run an episode
            obs = env.reset()
            action = agent.act(obs)
            obs2 = None
            done = False
            step_count = 0
            reward_sum = 0
            while not done:
                step_count += 1

                obs2, reward2, done, _ = env.step(action)
                action2 = agent.act(obs2)

                agent.observe_step(obs, action, reward2, obs2, terminal=done)
                agent.train(batch_size=32,iterations=1)

                # Next time step
                obs = obs2
                action = action2
    except ValueError as e:
        tqdm.write(str(e))
        tqdm.write("Diverged")

    while len(rewards) < (max_iters/epoch)+1: # Means it diverged at some point
        rewards.append([0]*test_iters)

    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", directory)
    with open(file_name, "wb") as f:
        dill.dump(data, f)

def run_trial_steps(gamma, alpha, eps_b, eps_t, directory=None,
        max_steps=5000, epoch=50, test_iters=1, verbose=False):
    args = locals()
    env_name = 'Breakout-v0'
    env = gym.make(env_name)
    env = AtariPreprocessing(env)
    env = FrameStack(env,4)
    test_env = gym.make(env_name)
    test_env = AtariPreprocessing(test_env)
    test_env = FrameStack(test_env,4)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=alpha,
            discount_factor=gamma,
            device=device,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t)
    )

    rewards = []
    try:
        done = True
        step_range = range(0,max_steps+1)
        if verbose:
            step_range = tqdm(step_range)
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                r = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))

            # Run step
            if done:
                obs = env.reset()
            action = agent.act(obs)

            obs2, reward2, done, _ = env.step(action)
            agent.observe_step(obs, action, reward2, obs2, terminal=done)

            # Update weights
            agent.train(batch_size=32,iterations=1)

            # Next time step
            obs = obs2
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")

    while len(rewards) < (max_steps/epoch)+1: # Means it diverged at some point
        rewards.append([0]*test_iters)

    data = (args, rewards, steps_to_learn)
    file_name, file_num = utils.find_next_free_file("results", "pkl", directory)
    with open(file_name, "wb") as f:
        dill.dump(data, f)
