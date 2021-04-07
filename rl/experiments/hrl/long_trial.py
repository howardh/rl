import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools

from agent.dqn_agent import DQNAgent
from agent.policy import get_greedy_epsilon_policy

from environment.wrappers import DiscreteObservationToBox

from .model import QFunction

import utils

def run_trial(gamma, alpha, eps_b, eps_t, tau, directory=None,
        net_structure=[2,3,4],
        env_name='FrozenLake-v0', batch_size=32, min_replay_buffer_size=1000,
        epoch=50, test_iters=1, verbose=False):
    args = locals()
    env = gym.make(env_name)
    env = DiscreteObservationToBox(env)
    test_env = gym.make(env_name)
    test_env = DiscreteObservationToBox(test_env)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=alpha,
            discount_factor=gamma,
            polyak_rate=tau,
            device=device,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t),
            q_net=QFunction(layer_sizes=net_structure)
    )

    # Create file to save results
    results_file_path = utils.save_results(args,
            {'rewards': [], 'state_action_values': []},
            directory=directory)

    rewards = []
    state_action_values = []
    done = True
    step_range = itertools.count()
    if verbose:
        step_range = tqdm(step_range)
    try:
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                r,sa_vals = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                state_action_values.append(sa_vals)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))
                utils.save_results(args,
                        {'rewards': rewards, 'state_action_values': state_action_values},
                        file_path=results_file_path)

            # Linearly Anneal epsilon
            agent.behaviour_policy = get_greedy_epsilon_policy((1-min(steps/1000000,1))*(1-eps_b)+eps_b)

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
            raise e
    except KeyboardInterrupt:
        tqdm.write('Keyboard Interrupt')

    return (args, rewards, state_action_values)

def plot(results_directory, plot_directory):
    results = utils.get_all_results(results_directory)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches(10,4)
    for k,v in results:
        params = dict(k)
        mean_rewards = [np.mean(epoch) for epoch in v['rewards']]
        mean_sa_vals = [np.mean(epoch) for epoch in v['state_action_values']]
        assert len(mean_rewards) == len(mean_sa_vals)

        x = list(range(0,len(mean_rewards)*params['epoch'],params['epoch']))
        ax1.set_title('Testing Reward')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Average Reward')
        ax1.grid(True,which='both',axis='both',color='grey')
        ax1.plot(x,mean_rewards)
        ax2.set_title('Predicted Action Values')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Expected Return')
        ax2.grid(True,which='both',axis='both',color='grey')
        ax2.plot(x,mean_sa_vals)
        ax2.set_ylim([0,1])
    file_name = os.path.join(plot_directory,'plot.png')
    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)
    fig.savefig(file_name)
    plt.close(fig)
    print('Saved file', file_name)

def run():
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)

    run_trial(gamma=1,alpha=0.001,eps_b=0.1,eps_t=0,tau=0.01,net_structure=(10,10),batch_size=256,epoch=1000,test_iters=10,verbose=True,directory=directory)
    plot(results_directory=directory,plot_directory=plot_directory)
