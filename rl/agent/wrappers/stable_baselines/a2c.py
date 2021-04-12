import time
from tqdm import tqdm
import gym
import copy
from collections import defaultdict
import numpy as np
import torch
import torch as th

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common import logger
from stable_baselines3.common.utils import safe_mean

import rl
from rl.agent.agent import Agent

class A2CAgent(Agent):
    def __init__(self, *args, **kwargs):
        self.n_rollout_steps = 5
        self.num_envs = 4 # env.num_envs
        self.done_collecting_rollouts = True
        self.iteration = 0

        self.model = A2C(*args, **kwargs)

        total_timesteps = -1
        eval_env = None
        callback = None
        eval_freq = -1
        n_eval_episodes = 5
        eval_log_path = None
        reset_num_timesteps = True
        tb_log_name = "A2C"
        self.total_timesteps, self.callback = self.model._setup_learn(
                total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        self._last_action = [None, None]
        self._last_values = [None, None]
        self._last_log_probs = [None, None]

        pass

    def observe(self, new_obs, reward, terminal=False, info=None, testing=False, worker_id=None):
        # renaming stuff
        rewards = reward
        dones = terminal
        infos = info
        callback = self.callback
        n_rollout_steps = self.model.n_steps
        total_timesteps = self.total_timesteps
        log_interval = 100
        iteration = self.iteration
        actions = self._last_action[testing]
        values = self._last_values[testing]
        log_probs = self._last_log_probs[testing]
        # n_steps is equivalent to self.model.rollout_buffer.size()

        if not testing:
            self.iteration += 1

        if reward is None: # After a reset
            self.model._last_obs = new_obs
            ### start of rollout
            if self.done_collecting_rollouts:
                self.done_collecting_rollouts = False
                assert self.model._last_obs is not None, "No previous observation was provided"
                #n_steps = 0
                self.model.rollout_buffer.reset()
                # Sample new weights for the state dependent exploration
                if self.model.use_sde:
                    self.model.policy.reset_noise(self.num_envs)

                callback.on_rollout_start()
        else: # collect_rollouts() starts here
            ### start of rollout
            if self.done_collecting_rollouts:
                self.done_collecting_rollouts = False
                assert self.model._last_obs is not None, "No previous observation was provided"
                #n_steps = 0
                self.model.rollout_buffer.reset()
                # Sample new weights for the state dependent exploration
                if self.model.use_sde:
                    self.model.policy.reset_noise(self.num_envs)

                callback.on_rollout_start()

            ### START - observe(new_obs=new_obs, reward=rewards, terminal=dones) - Part 1 ###
            self.model.num_timesteps += self.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                self.done_collecting_rollouts = True
                #return False

            if not self.done_collecting_rollouts:
                self.model._update_info_buffer(infos)
                #n_steps += 1

                if isinstance(self.model.action_space, gym.spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)
                self.model.rollout_buffer.add(self.model._last_obs, actions, rewards, self.model._last_dones, values, log_probs)
                self.model._last_obs = new_obs
                self.model._last_dones = dones
            ### END - observe(new_obs=new_obs, reward=rewards, terminal=dones) - Part 1 ###

            ### START - observe(new_obs=new_obs, reward=rewards, terminal=dones) - Part 2 ###
            if self.model.use_sde and self.model.sde_sample_freq > 0 and self.model.rollout_buffer.size() % self.model.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.model.policy.reset_noise(env.num_envs)
            ### END - observe(new_obs=new_obs, reward=rewards, terminal=dones) - Part 2 ###

        if self.model.rollout_buffer.size() >= n_rollout_steps:
            self.done_collecting_rollouts = True

            ### START - end of rollout ###
            with th.no_grad():
                # Compute value for the last timestep
                obs_tensor = th.as_tensor(new_obs).to(self.model.device)
                _, values, _ = self.model.policy.forward(obs_tensor)

            self.model.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

            callback.on_rollout_end()
            ### END - end of rollout ###

        if self.done_collecting_rollouts and not testing:
            self.model._update_current_progress_remaining(self.model.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.model.num_timesteps / (time.time() - self.model.start_time))
                logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer]))
                logger.record("time/fps", fps)
                logger.record("time/time_elapsed", int(time.time() - self.model.start_time), exclude="tensorboard")
                logger.record("time/total_timesteps", self.model.num_timesteps, exclude="tensorboard")
                logger.dump(step=self.model.num_timesteps)

            self.model.train()

    def act(self, observation, testing=False):
        single_env = False # False = vectorized env = observation is a vector of observations and returned action is a vector of actions
        if testing and len(observation.shape) == 1:
            observation = np.expand_dims(observation,0)
            single_env = True

        with th.no_grad():
            # Convert to pytorch tensor
            obs_tensor = torch.as_tensor(observation).to(self.model.device)
            actions, values, log_probs = self.model.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)

        if single_env:
            clipped_actions = clipped_actions.item()

        self._last_action[testing] = clipped_actions
        self._last_values[testing] = values
        self._last_log_probs[testing] = log_probs

        return clipped_actions

def collect_rollouts(self, env , callback , rollout_buffer , n_rollout_steps):
    ### START - start of rollout ###
    assert self._last_obs is not None, "No previous observation was provided"
    n_steps = 0
    rollout_buffer.reset()
    # Sample new weights for the state dependent exploration
    if self.use_sde:
        self.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()
    ### END - start of rollout ###

    while n_steps < n_rollout_steps:
        ### START - observe(new_obs=new_obs, reward=rewards, terminal=dones) - Part 2 ###
        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            # Sample a new noise matrix
            self.policy.reset_noise(env.num_envs)
        ### END - observe(new_obs=new_obs, reward=rewards, terminal=dones) - Part 2 ###

        ### START - act() ###
        with th.no_grad():
            # Convert to pytorch tensor
            obs_tensor = th.as_tensor(self._last_obs).to(self.device)
            actions, values, log_probs = self.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
        ### END - act() ###

        new_obs, rewards, dones, infos = env.step(clipped_actions)

        ### START - observe(new_obs=new_obs, reward=rewards, terminal=dones) - Part 1 ###
        self.num_timesteps += env.num_envs

        # Give access to local variables
        callback.update_locals(locals())
        if callback.on_step() is False:
            return False

        self._update_info_buffer(infos)
        n_steps += 1

        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)
        rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
        self._last_obs = new_obs
        self._last_dones = dones
        ### END - observe(new_obs=new_obs, reward=rewards, terminal=dones) - Part 1###

    ### START - end of rollout ###
    with th.no_grad():
        # Compute value for the last timestep
        obs_tensor = th.as_tensor(new_obs).to(self.device)
        _, values, _ = self.policy.forward(obs_tensor)

    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    callback.on_rollout_end()
    ### END - end of rollout ###

    return True

def run():
    #env = gym.make('CartPole-v1')
    env = make_vec_env('CartPole-v1', n_envs=4)
    model = A2C("MlpPolicy", env, verbose=1)

    #model.learn(total_timesteps=25000)

    log_interval = 100

    model._last_obs = obs = env.reset()
    total_timesteps = 25000
    eval_env = None
    callback = None
    eval_freq = -1
    n_eval_episodes = 5
    eval_log_path = None
    reset_num_timesteps = True
    tb_log_name = "A2C"

    iteration = 0

    total_timesteps, callback = model._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
    )

    while model.num_timesteps < total_timesteps:
        continue_training = collect_rollouts(model, model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

        if continue_training is False:
            break
        iteration += 1

        model._update_current_progress_remaining(model.num_timesteps, total_timesteps)

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            fps = int(model.num_timesteps / (time.time() - model.start_time))
            logger.record("time/iterations", iteration, exclude="tensorboard")
            if len(model.ep_info_buffer) > 0 and len(model.ep_info_buffer[0]) > 0:
                logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in model.ep_info_buffer]))
                logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in model.ep_info_buffer]))
            logger.record("time/fps", fps)
            logger.record("time/time_elapsed", int(time.time() - model.start_time), exclude="tensorboard")
            logger.record("time/total_timesteps", model.num_timesteps, exclude="tensorboard")
            logger.dump(step=model.num_timesteps)

        model.train()

    # Testing?
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

def run2():
    env = make_vec_env('CartPole-v1', n_envs=4)
    model = A2CAgent("MlpPolicy", env, verbose=1)

    # Training
    obs = env.reset()
    model.observe(obs, None)
    for i in tqdm(range(25000//4)):
        action = model.act(obs)
        obs, rewards, dones, info = env.step(action)
        model.observe(obs, rewards, dones, info, testing=False)

    # Testing
    env = gym.make('CartPole-v1')
    done = True
    while True:
        if done:
            obs = env.reset()
            model.observe(obs, None, testing=True)
        action = model.act(obs, testing=True)
        obs, reward, done, info = env.step(action)
        env.render()

if __name__=='__main__':
    run2()
