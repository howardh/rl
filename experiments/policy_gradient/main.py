import numpy as np
import gym
import torch
from tqdm import tqdm

from agent.ddpg_agent import DDPGAgent
from agent.policy import get_greedy_epsilon_policy

import utils
import dill

try:
    import roboschool
except:
    print('Roboschool unavailable')

class BestModelSaver():
    def __init__(self):
        self.best_model = None
        self.best_performance = None
    def record_performance(self, model, performance):
        if self.best_performance is None or self.best_performance < performance:
            self.best_model = model.state_dict()
            self.best_performance = performance
    def save_model(self, file_name):
        torch.save(self.best_model, file_name)

def run_trial(gamma, actor_lr, critic_lr, polyak_rate=1e-3, noise_std=0.1,
        results_directory=None, model_directory=None,
        env_name='MountainCarContinuous-v0',
        max_iters=5000, epoch=50, test_iters=1,verbose=False):
    args = locals()
    env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = DDPGAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            discount_factor=gamma,
            device=device,
            training_noise=torch.distributions.Normal(0,noise_std),
            testing_noise=torch.distributions.Normal(0,0.01)
    )
    best_model_saver = BestModelSaver()

    rewards = []
    try:
        for iters in range(0,max_iters+1):
            # Run tests
            if iters % epoch == 0:
                r = agent.test(env, test_iters, render=False, processors=1)
                rewards.append(r)
                if verbose:
                    tqdm.write('iter %d \t Reward: %f' % (iters, np.mean(r)))
                best_model_saver.record_performance(agent.actor, np.mean(r))
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
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")

    # Save best model
    if model_directory is not None:
        model_file_name, file_num = utils.find_next_free_file(
                "model", "pt", model_directory)
        best_model_saver.save_model(model_file_name)
    # Save rewards
    if results_directory is not None:
        data = {
                'args': args,
                'rewards': rewards,
                'model_file_name': model_file_name
        }
        file_name, file_num = utils.find_next_free_file(
                "results", "pkl", results_directory)
        with open(file_name, "wb") as f:
            dill.dump(data, f)

def run_trial_steps(gamma, actor_lr, critic_lr, noise_std=0.1,
        polyak_rate=1e-3, results_directory=None, model_directory=None,
        env_name='MountainCarContinuous-v0',
        batch_size=32, min_replay_buffer_size=10000,
        max_steps=5000, epoch=50, test_iters=1, verbose=False):
    args = locals()
    env = gym.make(env_name)
    test_env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = DDPGAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            discount_factor=gamma,
            polyak_rate=polyak_rate,
            device=device,
            training_noise=torch.distributions.Normal(0,noise_std),
            testing_noise=torch.distributions.Normal(0,0.01)
    )
    best_model_saver = BestModelSaver()

    rewards = []
    done = True
    step_range = range(0,max_steps+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                r = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))
                best_model_saver.record_performance(agent.actor, np.mean(r))

            # Run step
            if done:
                obs = env.reset()
            action = agent.act(obs)

            obs2, reward2, done, _ = env.step(action)
            agent.observe_step(obs, action, reward2, obs2, terminal=done)

            # Update weights
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)

            # Next time step
            obs = obs2
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")

    # Save best model
    if model_directory is not None:
        model_file_name, file_num = utils.find_next_free_file(
                "model", "pt", model_directory)
        best_model_saver.save_model(model_file_name)
    # Save rewards
    if results_directory is not None:
        data = {
                'args': args,
                'rewards': rewards,
                'model_file_name': model_file_name
        }
        file_name, file_num = utils.find_next_free_file(
                "results", "pkl", results_directory)
        with open(file_name, "wb") as f:
            dill.dump(data, f)
