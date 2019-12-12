import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools
from collections import defaultdict

from agent.hdqn_agent import HDQNAgentWithDelayAC, HDQNAgentWithDelayAC_v2, HDQNAgentWithDelayAC_v3
from agent.policy import get_greedy_epsilon_policy

from .model import QFunction, PolicyFunction, PolicyFunctionAugmentatedState
from .long_trial import plot

import utils

class TimeLimit(gym.Wrapper):
    """ Copied from gym.wrappers.TimeLimit with small modifications."""
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info
    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

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

def create_agent(agent_name, env, device, **agent_params):
    before_step = lambda s: None
    after_step = lambda s: None

    # FIXME: This won't warn me if I make a typo in the param name.
    gamma = agent_params.pop('gamma',0.9)
    eps_b = agent_params.pop('eps_b',0.05)
    num_options = agent_params.pop('num_options',3)
    min_replay_buffer_size = agent_params.pop('min_replay_buffer_size',1000)
    batch_size = agent_params.pop('batch_size',256)
    controller_learning_rate = agent_params.pop('controller_learning_rate',0.01)
    subpolicy_learning_rate = agent_params.pop('subpolicy_learning_rate',0.01)
    q_net_learning_rate = agent_params.pop('q_net_learning_rate',0.01)
    polyak_rate = agent_params.pop('polyak_rate',0.001)
    replay_buffer_size = agent_params.pop('replay_buffer_size',10000)
    delay = agent_params.pop('delay',1)
    if agent_name == 'HDQNAgentWithDelayAC':
        controller_net_structure = agent_params.pop(
                'controller_net_structure',[2,2])
        subpolicy_net_structure = agent_params.pop(
                'subpolicy_net_structure',[3])
        q_net_structure = agent_params.pop(
                'q_net_structure',[15,15])
        agent = HDQNAgentWithDelayAC(
                action_space=env.action_space,
                observation_space=env.observation_space,
                controller_learning_rate=controller_learning_rate,
                subpolicy_learning_rate=subpolicy_learning_rate,
                q_net_learning_rate=q_net_learning_rate,
                discount_factor=gamma,
                polyak_rate=polyak_rate,
                device=device,
                behaviour_epsilon=eps_b,
                replay_buffer_size=replay_buffer_size,
                delay_steps=1,
                controller_net=PolicyFunction(
                    layer_sizes=controller_net_structure,
                    input_size=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(
                    layer_sizes=subpolicy_net_structure,input_size=4)
                    for _ in range(num_options)],
                q_net=QFunction(layer_sizes=q_net_structure,
                    input_size=4,output_size=4),
        )
        def before_step(steps):
            agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'HDQNAgentWithDelayAC_v2':
        controller_net_structure = agent_params.pop(
                'controller_net_structure',[2,2])
        subpolicy_net_structure = agent_params.pop(
                'subpolicy_net_structure',[3])
        q_net_structure = agent_params.pop(
                'q_net_structure',[15,15])
        subpolicy_q_net_learning_rate = agent_params.pop(
                'subpolicy_q_net_learning_rate',1e-3)
        agent = HDQNAgentWithDelayAC_v2(
                action_space=env.action_space,
                observation_space=env.observation_space,
                controller_learning_rate=controller_learning_rate,
                subpolicy_learning_rate=subpolicy_learning_rate,
                q_net_learning_rate=q_net_learning_rate,
                subpolicy_q_net_learning_rate=subpolicy_q_net_learning_rate,
                discount_factor=gamma,
                polyak_rate=polyak_rate,
                device=device,
                behaviour_epsilon=eps_b,
                replay_buffer_size=replay_buffer_size,
                delay_steps = delay,
                controller_net=PolicyFunction(
                    layer_sizes=controller_net_structure,
                    input_size=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(
                    layer_sizes=subpolicy_net_structure,input_size=4)
                    for _ in range(num_options)],
                q_net=QFunction(layer_sizes=q_net_structure,
                    input_size=4,output_size=4),
        )
        def before_step(steps):
            agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'HDQNAgentWithDelayAC_v3':
        controller_net_structure = agent_params.pop(
                'controller_net_structure',[10,10])
        subpolicy_net_structure = agent_params.pop(
                'subpolicy_net_structure',[3])
        q_net_structure = agent_params.pop(
                'q_net_structure',[15,15])
        subpolicy_q_net_learning_rate = agent_params.pop(
                'subpolicy_q_net_learning_rate',1e-3)
        agent = HDQNAgentWithDelayAC_v3(
                action_space=env.action_space,
                observation_space=env.observation_space,
                controller_learning_rate=controller_learning_rate,
                subpolicy_learning_rate=subpolicy_learning_rate,
                q_net_learning_rate=q_net_learning_rate,
                subpolicy_q_net_learning_rate=subpolicy_q_net_learning_rate,
                discount_factor=gamma,
                polyak_rate=polyak_rate,
                device=device,
                behaviour_epsilon=eps_b,
                replay_buffer_size=replay_buffer_size,
                delay_steps = delay,
                controller_net=PolicyFunctionAugmentatedState(
                    layer_sizes=controller_net_structure,state_size=4,
                    num_actions=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(
                    layer_sizes=subpolicy_net_structure,input_size=4)
                    for _ in range(num_options)],
                q_net=QFunction(layer_sizes=q_net_structure,
                    input_size=4,output_size=4),
        )
        def before_step(steps):
            agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'ActorCritic':
        controller_net_structure = agent_params.pop(
                'controller_net_structure',[2,2])
        subpolicy_net_structure = agent_params.pop(
                'subpolicy_net_structure',[3])
        q_net_structure = agent_params.pop(
                'q_net_structure',[15,15])
        agent = HDQNAgentWithDelayAC(
                action_space=env.action_space,
                observation_space=env.observation_space,
                controller_learning_rate=controller_learning_rate,
                subpolicy_learning_rate=subpolicy_learning_rate,
                q_net_learning_rate=q_net_learning_rate,
                discount_factor=gamma,
                polyak_rate=polyak_rate,
                device=device,
                behaviour_epsilon=eps_b,
                replay_buffer_size=replay_buffer_size,
                delay_steps = 0,
                controller_net=PolicyFunction(
                    layer_sizes=controller_net_structure,
                    input_size=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(
                    layer_sizes=subpolicy_net_structure,input_size=4)
                    for _ in range(num_options)],
                q_net=QFunction(layer_sizes=q_net_structure,
                    input_size=4,output_size=4),
        )
        def before_step(steps):
            agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'OptionCritic':
        raise NotImplementedError('Option Critic not implemented yet.')

    if len(agent_params) > 0:
        raise Exception('Unused agent parameters: %s' % agent_params.keys())

    return agent, before_step, after_step

def run_trial(directory=None, steps_per_task=100, total_steps=1000,
        epoch=50, test_iters=1, verbose=False,
        agent_name='HDQNAgentWithDelayAC', **agent_params):
    args = locals()
    env_name='gym_fourrooms:fourrooms-v0'

    env = gym.make(env_name,goal_duration_steps=steps_per_task).unwrapped
    env = TimeLimit(env,36)
    test_env = gym.make(env_name,goal_duration_steps=float('inf')).unwrapped
    test_env = TimeLimit(test_env,500)

    print(env, test_env)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent, before_step, after_step = create_agent(
            agent_name, env, device, **agent_params)

    # Create file to save results
    results_file_path = utils.save_results(args,
            {'rewards': [], 'state_action_values': []},
            directory=directory,
            file_name_prefix=agent_name)

    if verbose:
        step_range = tqdm(range(total_steps))

    rewards = []
    state_action_values = []
    steps_to_reward = []
    done = True
    for steps in step_range:
        # Run tests
        if steps % epoch == 0:
            test_env.reset_goal(env.goal)
            test_results = agent.test(
                    test_env, test_iters, render=False, processors=1)
            rewards.append(np.mean(
                [r['total_rewards'] for r in test_results]))
            state_action_values.append(np.mean(
                [r['state_action_values'] for r in test_results]))
            steps_to_reward.append(np.mean(
                [r['steps'] for r in test_results]))
            if verbose:
                tqdm.write('steps %d \t Reward: %f \t Steps: %f' % (
                    steps, rewards[-1], steps_to_reward[-1]))
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

def run_gridsearch(proc=1):
    directory = os.path.join(utils.get_results_directory(),__name__)
    params_ac = {
            'agent_name': ['ActorCritic'],
            'gamma': [0.9],
            'controller_learning_rate': [0.01,0.001],
            'subpolicy_learning_rate': [0.01,0.001],
            'q_net_learning_rate': [0.01,0.001],
            'eps_b': [0.05],
            'polyak_rate': [0.001],
            'batch_size': [256],
            'min_replay_buffer_size': [1000],
            'steps_per_task': [100000],
            'epoch': [1000],
            'test_iters': [5],
            'verbose': [True],
            'controller_net_structure': [(10,10),(20,20)],
            'subpolicy_net_structure': [(5,5),(3,)],
            'q_net_structure': [(10,10),(20,20)],
            'directory': [directory]
    }
    params_v3 = {
            'agent_name': ['HDQNAgentWithDelayAC_v3'],
            'gamma': [0.9],
            'controller_learning_rate': [0.01,0.001],
            'subpolicy_learning_rate': [0.01,0.001],
            'q_net_learning_rate': [0.01,0.001],
            'subpolicy_q_net_learning_rate': [0.01,0.001],
            'eps_b': [0.05],
            'polyak_rate': [0.001],
            'batch_size': [256],
            'min_replay_buffer_size': [1000],
            'steps_per_task': [100000],
            'epoch': [1000],
            'test_iters': [5],
            'verbose': [True],
            'controller_net_structure': [(10,10),(20,20)],
            'subpolicy_net_structure': [(5,5),(3,)],
            'q_net_structure': [(10,10),(20,20)],
            'directory': [directory]
    }
    funcs = utils.gridsearch(params, run_trial)
    utils.cc(funcs,proc=proc)
    return utils.get_all_results(directory)

def run_refinement(proc=1,runs_per_agent=1,
        agent_name='HDQNAgentWithDelayAC_v3'):
    directory = os.path.join(utils.get_results_directory(),__name__)
    scores = compute_score(directory)
    scores_by_agent = defaultdict(lambda: [])
    for p,s in scores:
        p = dict(p)
        scores_by_agent[p['agent_name']].append((p,s))
    s = scores_by_agent[agent_name][0][1]
    p = dict(scores_by_agent[agent_name][0][0])
    p['directory'] = directory
    print(s)
    print(p)
    funcs = [lambda: run_trial(**p) for _ in range(runs_per_agent)]
    utils.cc(funcs,proc=proc)

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
        data_x[args['agent_name']] = range(0,args['total_steps'],args['epoch'])
    for k,v in data_y.items():
        max_len = max([len(y) for y in v])
        data_y[k] = np.nanmean(np.array([y+[np.nan]*(max_len-len(y)) for y in v]),0)
        plt.plot(data_x[k],data_y[k],label=k)
    plt.xlabel('Training Steps')
    plt.ylabel('Steps to Reward')
    plt.legend(loc='best')
    plt.grid()
    plot_path = os.path.join(plot_directory,'plot.png')
    plt.savefig(plot_path)
    plt.close()
    print('Saved plot %s' % plot_path)

def plot(data, plot_directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    for k,v in data.items():
        plt.plot(v['x'],v['y'],label=k)
    plt.xlabel('Training Steps')
    plt.ylabel('Steps to Reward')
    plt.legend(loc='best')
    plt.grid(which='both')
    plot_path = os.path.join(plot_directory,'plot.png')
    plt.savefig(plot_path)
    plt.close()
    print('Saved plot %s' % plot_path)

def map_func(params):
    p = params
    ap = p.pop('agent_params',{})
    if 'directory' in p:
        del p['directory']
    for k,v in ap.items():
        p[k] = v
    return frozenset(p.items())

def compute_score(directory,params={}):
    def reduce(r,acc):
        acc.append(np.mean(r['steps_to_reward'][50:]))
        return acc
    scores = utils.get_all_results_map_reduce(
            directory, map_func, reduce, lambda: [])
    for k,v in scores.items():
        scores[k] = np.mean(v)
    sorted_scores = sorted(scores.items(),key=lambda x: x[1])
    return sorted_scores

def compute_series(directory,params={}):
    def reduce(r,acc):
        acc.append(r['steps_to_reward'])
        return acc
    series = utils.get_all_results_map_reduce(
            directory, map_func, reduce, lambda: [])
    for k,v in series.items():
        max_len = max([len(x) for x in v])
        series[k] = np.nanmean(np.array(list(itertools.zip_longest(*v,fillvalue=np.nan))),axis=1)
    return series

def run():
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)

    #run_gridsearch()
    series = compute_series(directory)
    scores = compute_score(directory)
    scores_by_agent = defaultdict(lambda: [])
    for p,s in scores:
        p = dict(p)
        scores_by_agent[p['agent_name']].append((p,s))
    agent_names = scores_by_agent.keys()
    data = defaultdict(lambda: {'x': None, 'y': None})
    for an in agent_names:
        p = scores_by_agent[an][0][0]
        s = series[map_func(p)]
        data[an]['x'] = range(0,p['total_steps'],p['epoch'])
        data[an]['y'] = s
    plot(data,plot_directory)

    #for _ in range(10):
    while True:
        run_refinement(agent_name='ActorCritic')
        run_refinement(agent_name='HDQNAgentWithDelayAC_v3')
    #    plot(results_directory=directory,plot_directory=plot_directory)
    #    run_trial(gamma=0.9,agent_name='ActorCritic',steps_per_task=10000,total_steps=100000,epoch=1000,test_iters=10,verbose=True,directory=directory)
    #    #plot(results_directory=directory,plot_directory=plot_directory)
    #    #run_trial(gamma=0.9,agent_name='HDQNAgentWithDelayAC_v2',delay=1,steps_per_task=10000,total_steps=100000,epoch=1000,test_iters=10,verbose=True,directory=directory)
    #    plot(results_directory=directory,plot_directory=plot_directory)
    #    run_trial(gamma=0.9,agent_name='HDQNAgentWithDelayAC_v3',delay=1,steps_per_task=10000,total_steps=100000,epoch=1000,test_iters=10,verbose=True,directory=directory)
