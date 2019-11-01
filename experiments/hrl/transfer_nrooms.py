import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools
from collections import defaultdict

from agent.hdqn_agent import HDQNAgentWithDelayAC
from agent.policy import get_greedy_epsilon_policy

from .model import QFunction, PolicyFunction
from .long_trial import plot

import utils

rooms_map_1 = """
xxxxxxxxxxxxxxx
x       x     x
x       x     x
x       x     x
x       x     x
x    xx x     x
x    x  x     x
x    x xx     x
x    x        x
x    x        x
x    x        x
x    x        x
xxxxxxxxxxxxxxx"""
rooms_map_2 = """
xxxxxxxxxxxxxxx
x             x
x             x
x             x
x             x
xxxxxxxxx     x
x       x     x
x     x       x
x     xxxxxxxxx
x             x
x             x
x             x
xxxxxxxxxxxxxxx"""

def run_trial(gamma, directory=None, steps_per_task=100,
        epoch=50, test_iters=1, verbose=False,
        agent_name='HDQNAgentWithDelayAC', **agent_params):
    args = locals()
    env_name='gym_fourrooms:fourrooms-v0'

    #env_name = 'FrozenLake-v0' # wtf? Why is e2 getting wrapped in a TimeLimit?
    #e1 = gym.make(env_name)
    #e1 = gym.wrappers.TimeLimit(e1,20)
    #e2 = gym.make(env_name)
    #breakpoint()

    env1 = gym.make(env_name,env_map=rooms_map_1)
    env1 = gym.wrappers.TimeLimit(env1,20)
    test_env1 = gym.make(env_name,env_map=rooms_map_1).env
    test_env1 = gym.wrappers.TimeLimit(test_env1,150)
    env2 = gym.make(env_name,env_map=rooms_map_2).env
    env2 = gym.wrappers.TimeLimit(env2,20)
    test_env2 = gym.make(env_name,env_map=rooms_map_2).env
    test_env2 = gym.wrappers.TimeLimit(test_env2,150)

    print(env1, test_env1)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    before_step = lambda s: None
    after_step = lambda s: None

    if agent_name == 'HDQNAgentWithDelayAC' or agent_name == 'ActorCritic':
        if agent_name == 'HDQNAgentWithDelayAC':
            delay = agent_params['delay']
        else:
            delay = 0
        eps_b = 0.01
        num_options = 3
        min_replay_buffer_size = 1000
        batch_size = 256
        agent = HDQNAgentWithDelayAC(
                action_space=env1.action_space,
                observation_space=env1.observation_space,
                learning_rate=0.01,
                discount_factor=gamma,
                polyak_rate=0.001,
                device=device,
                behaviour_epsilon=eps_b,
                replay_buffer_size=1000,
                delay_steps=delay,
                controller_net=PolicyFunction(
                    layer_sizes=[3,3],input_size=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(layer_sizes=[2],input_size=4) for _ in range(num_options)],
                q_net=QFunction(layer_sizes=(15,15),input_size=4,output_size=4),
        )
        def before_step(steps):
            agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'OptionCritic':
        raise NotImplementedError('Option Critic not implemented yet.')

    # Create file to save results
    results_file_path = utils.save_results(args,
            {'rewards': [], 'state_action_values': []},
            directory=directory)

    if verbose:
        step_range1 = tqdm(range(steps_per_task))
        step_range2 = tqdm(range(steps_per_task))
    else:
        step_range1 = range(steps_per_task)
        step_range2 = range(steps_per_task)

    rewards = []
    state_action_values = []
    steps_to_reward = []
    for step_range, env, test_env in [(step_range1, env1, test_env1),(step_range2, env2, test_env2)]:
        done = True
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                test_results = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(np.mean([r['total_rewards'] for r in test_results]))
                state_action_values.append(np.mean([r['state_action_values'] for r in test_results]))
                steps_to_reward.append(np.mean([r['steps'] for r in test_results]))
                if verbose:
                    tqdm.write('steps %d \t Reward: %f \t Steps: %f' % (steps, rewards[-1], steps_to_reward[-1]))
                utils.save_results(args,
                        {'rewards': rewards,
                        'state_action_values': state_action_values,
                        'steps_to_reward': steps_to_reward},
                        file_path=results_file_path)

            before_step(steps)

            # Run step
            if done:
                obs = env.reset()
                agent.observe_change(obs, None)
            obs, reward, done, _ = env.step(agent.act())
            agent.observe_change(obs, reward, terminal=done)

            # Update weights
            after_step(steps)

    return (args, rewards, state_action_values)

def plot(results_directory,plot_directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    data_y = defaultdict(lambda: [])
    data_x = {}
    for args,result in utils.get_all_results(results_directory):
        data_y[args['agent_name']].append(result['steps_to_reward'])
        data_x[args['agent_name']] = range(0,args['steps_per_task']*2,args['epoch'])
    for k,v in data_y.items():
        max_len = max([len(y) for y in v])
        data_y[k] = np.nanmean(np.array([y+[np.nan]*(max_len-len(y)) for y in v]),0)
        plt.plot(data_x[k],data_y[k],label=k)
    plt.xlabel('Training Steps')
    plt.ylabel('Steps to Reward')
    plt.legend(loc='best')
    plot_path = os.path.join(plot_directory,'plot.png')
    plt.savefig(plot_path)
    plt.close()
    print('Saved plot %s' % plot_path)

def run():
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)

    #for _ in range(10):
    while True:
        run_trial(gamma=0.9,agent_name='ActorCritic',steps_per_task=10000,epoch=100,test_iters=10,verbose=True,directory=directory)
        run_trial(gamma=0.9,agent_name='HDQNAgentWithDelayAC',delay=1,steps_per_task=10000,epoch=100,test_iters=10,verbose=True,directory=directory)
        plot(results_directory=directory,plot_directory=plot_directory)
