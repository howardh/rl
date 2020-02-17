import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools
from collections import defaultdict
import pprint
from skopt import gp_minimize

from agent.hdqn_agent import HDQNAgentWithDelayAC, HDQNAgentWithDelayAC_v2, HDQNAgentWithDelayAC_v3
from agent.policy import get_greedy_epsilon_policy

from .model import QFunction, PolicyFunction, PolicyFunctionAugmentatedState
from .long_trial import plot

import utils
import hyperparams
import hyperparams.utils
from hyperparams.distributions import Uniform, LogUniform, CategoricalUniform, DiscreteUniform

def get_search_space():
    actor_critic_hyperparam_space = {
            'agent_name': 'ActorCritic',
            'gamma': 0.9,
            'controller_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_learning_rate': LogUniform(1e-4,1e-1),
            'q_net_learning_rate': LogUniform(1e-4,1e-1),
            'eps_b': Uniform(0,0.5),
            'polyak_rate': 0.001,
            'batch_size': 256,
            'min_replay_buffer_size': 1000,
            'steps_per_task': 10000,
            'total_steps': 100000,
            'epoch': 1000,
            'test_iters': 5,
            'verbose': True,
            'num_options': DiscreteUniform(2,10),
            'cnet_n_layers': DiscreteUniform(1,2),
            'cnet_layer_size': DiscreteUniform(1,20),
            'snet_n_layers': DiscreteUniform(1,2),
            'snet_layer_size': DiscreteUniform(1,5),
            'qnet_n_layers': DiscreteUniform(1,2),
            'qnet_layer_size': DiscreteUniform(10,20),
            'directory': None
    }
    hrl_delay_augmented_hyperparam_space = {
            'agent_name': 'HDQNAgentWithDelayAC_v3',
            'gamma': 0.9,
            'controller_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_learning_rate': LogUniform(1e-4,1e-1),
            'q_net_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_q_net_learning_rate': LogUniform(1e-4,1e-1),
            'eps_b': Uniform(0,0.5),
            'polyak_rate': 0.001,
            'batch_size': 256,
            'min_replay_buffer_size': 1000,
            'steps_per_task': 10000,
            'total_steps': 100000,
            'epoch': 1000,
            'test_iters': 5,
            'verbose': True,
            'num_options': DiscreteUniform(2,10),
            'cnet_n_layers': DiscreteUniform(1,2),
            'cnet_layer_size': DiscreteUniform(1,20),
            'snet_n_layers': DiscreteUniform(1,2),
            'snet_layer_size': DiscreteUniform(1,5),
            'qnet_n_layers': DiscreteUniform(1,2),
            'qnet_layer_size': DiscreteUniform(10,20),
            'directory': None
    }
    hrl_delay_memoryless_hyperparam_space = {
            'agent_name': 'HDQNAgentWithDelayAC_v2',
            'gamma': 0.9,
            'controller_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_learning_rate': LogUniform(1e-4,1e-1),
            'q_net_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_q_net_learning_rate': LogUniform(1e-4,1e-1),
            'eps_b': Uniform(0,0.5),
            'polyak_rate': 0.001,
            'batch_size': 256,
            'min_replay_buffer_size': 1000,
            'steps_per_task': 10000,
            'total_steps': 100000,
            'epoch': 1000,
            'test_iters': 5,
            'verbose': True,
            'num_options': DiscreteUniform(2,10),
            'cnet_n_layers': DiscreteUniform(1,2),
            'cnet_layer_size': DiscreteUniform(1,20),
            'snet_n_layers': DiscreteUniform(1,2),
            'snet_layer_size': DiscreteUniform(1,5),
            'qnet_n_layers': DiscreteUniform(1,2),
            'qnet_layer_size': DiscreteUniform(10,20),
            'directory': None
    }
    return {
            'ActorCritic': actor_critic_hyperparam_space,
            'HDQNAgentWithDelayAC_v2': hrl_delay_memoryless_hyperparam_space,
            'HDQNAgentWithDelayAC_v3': hrl_delay_augmented_hyperparam_space
    }

space = get_search_space()

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
        cnet_n_layers = agent_params.pop(
                'cnet_n_layers',None)
        cnet_layer_size = agent_params.pop(
                'cnet_layer_size',None)
        snet_n_layers = agent_params.pop(
                'snet_n_layers',None)
        snet_layer_size = agent_params.pop(
                'snet_layer_size',None)
        qnet_n_layers = agent_params.pop(
                'qnet_n_layers',None)
        qnet_layer_size = agent_params.pop(
                'qnet_layer_size',None)
        controller_net_structure = tuple([cnet_layer_size]*cnet_n_layers)
        subpolicy_net_structure = tuple([snet_layer_size]*snet_n_layers)
        q_net_structure = tuple([qnet_layer_size]*qnet_n_layers)
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
            #agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
            pass
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'HDQNAgentWithDelayAC_v2':
        cnet_n_layers = agent_params.pop(
                'cnet_n_layers',None)
        cnet_layer_size = agent_params.pop(
                'cnet_layer_size',None)
        snet_n_layers = agent_params.pop(
                'snet_n_layers',None)
        snet_layer_size = agent_params.pop(
                'snet_layer_size',None)
        qnet_n_layers = agent_params.pop(
                'qnet_n_layers',None)
        qnet_layer_size = agent_params.pop(
                'qnet_layer_size',None)
        controller_net_structure = tuple([cnet_layer_size]*cnet_n_layers)
        subpolicy_net_structure = tuple([snet_layer_size]*snet_n_layers)
        q_net_structure = tuple([qnet_layer_size]*qnet_n_layers)
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
            #agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
            pass
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'HDQNAgentWithDelayAC_v3':
        cnet_n_layers = agent_params.pop(
                'cnet_n_layers',None)
        cnet_layer_size = agent_params.pop(
                'cnet_layer_size',None)
        snet_n_layers = agent_params.pop(
                'snet_n_layers',None)
        snet_layer_size = agent_params.pop(
                'snet_layer_size',None)
        qnet_n_layers = agent_params.pop(
                'qnet_n_layers',None)
        qnet_layer_size = agent_params.pop(
                'qnet_layer_size',None)
        controller_net_structure = tuple([cnet_layer_size]*cnet_n_layers)
        subpolicy_net_structure = tuple([snet_layer_size]*snet_n_layers)
        q_net_structure = tuple([qnet_layer_size]*qnet_n_layers)
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
            #agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
            pass
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'ActorCritic':
        ['cnet_n_layers', 'cnet_layer_size', 'snet_n_layers',
                'snet_layer_size', 'qnet_n_layers', 'qnet_layer_size']
        cnet_n_layers = agent_params.pop(
                'cnet_n_layers',None)
        cnet_layer_size = agent_params.pop(
                'cnet_layer_size',None)
        snet_n_layers = agent_params.pop(
                'snet_n_layers',None)
        snet_layer_size = agent_params.pop(
                'snet_layer_size',None)
        qnet_n_layers = agent_params.pop(
                'qnet_n_layers',None)
        qnet_layer_size = agent_params.pop(
                'qnet_layer_size',None)
        controller_net_structure = tuple([cnet_layer_size]*cnet_n_layers)
        subpolicy_net_structure = tuple([snet_layer_size]*snet_n_layers)
        q_net_structure = tuple([qnet_layer_size]*qnet_n_layers)
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
            #agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
            pass
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
    pprint.pprint(args)

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

    return np.mean(steps_to_reward[50:])
    #return (args, rewards, state_action_values)

def run_hyperparam_search_extremes(space, proc=1):
    directory = os.path.join(utils.get_results_directory(),__name__)
    params = hyperparams.utils.list_extremes(space)
    params = utils.split_params(params)
    funcs = [lambda: run_trial(**p) for p in params]
    utils.cc(funcs,proc=proc)
    return utils.get_all_results(directory)

def run_hyperparam_search(space, proc=1):
    directory = os.path.join(utils.get_results_directory(),__name__)
    params = [hyperparams.utils.sample_hyperparam(space) for _ in range(proc)]
    funcs = [lambda: run_trial(**p) for p in params]
    utils.cc(funcs,proc=proc)
    return utils.get_all_results(directory)

def random_lin_comb(vecs):
    n_points = vecs.shape[0]
    weights = np.random.rand(n_points)
    weights /= np.sum(weights)
    return (weights.reshape(n_points,1)*vecs).sum(0)

def sample_convex_hull(results_directory, agent_name='ActorCritic', threshold=0.1, perturbance=0):
    """ Find the top {threshold}% of parameters, and sample a set of parameters
    within the convex hull formed by those points.
    """
    from scipy.spatial import ConvexHull
    scores = compute_score(results_directory,sortby='mean')[agent_name]
    params = []
    for p,s in scores[:n_points]:
        params.append(hyperparams.utils.param_to_vec(p,space[agent_name]))
    params = np.array(params)
    hull = ConvexHull(params)
    output = random_lin_comb(hull)
    output = output.tolist()
    if perturbance > 0:
        output = hyperparams.utils.perturb_vec(output,space[agent_name],perturbance)
    output = hyperparams.utils.vec_to_param(output,space[agent_name])
    return output

def bin_lsh(data, space, n_planes=4):
    random_planes = []
    for _ in range(n_planes):
        u = hyperparams.utils.sample_hyperparam(space)
        v = hyperparams.utils.sample_hyperparam(space)
        u = hyperparams.utils.param_to_vec(u,space)
        v = hyperparams.utils.param_to_vec(v,space)
        u = torch.tensor(u)
        v = torch.tensor(v)
        random_planes.append((u,v))

    bins = [[] for _ in range(1<<n_planes)]
    for p,v in data:
        # Convert params to vector
        x = hyperparams.utils.param_to_vec(p,space)
        x = torch.tensor(x)
        # Compute random projections
        bits = [torch.dot(v,x-u)>0 for u,v in random_planes]
        index = sum([b*(1<<i) for i,b in enumerate(bits)])
        # Place data in appropriate bin
        bins[index].append((x.tolist(),v))
    return bins

def hoeffding(vals,target,score_range):
    t = np.nanmean(vals) - target
    n = len(vals)
    d = score_range[1]-score_range[0]
    return np.exp(-2*n**2*t**2/d**2)

def sample_lsh(results_directory, agent_name='ActorCritic', n_planes=4, perturbance=0,
        scoring='mean', target_score=None, score_range=[0,500]):
    """ Split the data into a number of buckets through locality-sensitive
    hashing. Find the bucket with the best average score and sample parameters
    in the convex hull of parameters in that bucket.
    """
    scores = compute_score(results_directory)[agent_name]

    bins = bin_lsh(scores,space[agent_name], n_planes=n_planes)
    if scoring=='mean':
        score_bins = [np.nanmean([v['mean'] for k,v in b]) for b in bins]
        best_index = np.nanargmin(score_bins) # Lower score is better
    elif scoring=='improvement_prob':
        score_bins = [hoeffding([v['mean'] for k,v in b],target_score,score_range) for b in bins]
        best_index = np.nanargmax(score_bins) # Want highest probability of improvement
    param = random_lin_comb(np.array([k for k,v in bins[best_index]])).tolist()
    if perturbance > 0:
        param = hyperparams.utils.perturb_vec(param,space[agent_name],perturbance)
    param = hyperparams.utils.vec_to_param(param,space[agent_name])
    print('Performance:',np.nanmean([v['mean'] for k,v in bins[best_index]]))
    print('Number of points',len(bins[best_index]))
    if target_score is not None:
        print('Improvement Probability:',hoeffding([v['mean'] for k,v in bins[best_index]],target_score,score_range))
    return param

def run_bayes_opt(results_directory, agent_name='ActorCritic'):
    scores = compute_score(results_directory)[agent_name]
    x0 = []
    y0 = []
    count = 0
    for p,v in scores:
        if p['agent_name'] != agent_name:
            continue
        vec = hyperparams.utils.param_to_vec(p,space[agent_name])
        for a in vec:
            if np.abs(a) > 1.7320508075688772*100:
                count += 1
                break
        else:
            x0.append(vec)
            y0.append(v['mean'])
    print('Out of bounds:',count,len(x0))
    def func(vec):
        p = hyperparams.utils.vec_to_param(vec,space[agent_name])
        return run_trial(**p)
    res = gp_minimize(
            func,                  # the function to minimize
            hyperparams.utils.space_to_ranges(space[agent_name]),      # the bounds on each dimension of x
            acq_func="EI",      # the acquisition function
            n_calls=0,         # the number of evaluations of f
            n_random_starts=0,  # the number of random initialization points
            noise=25, # Variance of the function output
            x0=x0,y0=y0)
    vec = res.x
    val = res.fun
    print('Optimal parameters (score: %f):' % val)
    pprint.pprint(hyperparams.utils.vec_to_param(vec,space[agent_name]))

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

def plot_single_param(results_directory, plot_directory, agent_name, param,
        log=False):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    scores = compute_score(results_directory)[agent_name]

    x = []
    y = []
    for k,v in scores:
        p = dict(k)
        if log:
            x.append(np.log(p[param]))
        else:
            x.append(p[param])
        y.append(v['mean'])
    plt.scatter(x,y)
    plt.xlabel(param)
    plt.ylabel('Score')
    #plt.legend(loc='best')
    plt.grid(which='both')
    plot_path = os.path.join(plot_directory,'plot.png')
    plt.savefig(plot_path)
    plt.close()
    print('Saved plot %s' % plot_path)

def plot_tsne(results_directory, plot_directory, agent_name):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import sklearn
    from sklearn.manifold import TSNE

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    scores = compute_score(results_directory)[agent_name]

    log_params = ['controller_learning_rate','subpolicy_learning_rate','q_net_learning_rate']
    ignore_params = ['agent_name','gamma','test_iters','verbose','directory']

    params = []
    values = []
    for p,s in scores:
        params.append(hyperparams.utils.param_to_vec(p,space[agent_name]))
        values.append(s['mean'])
    x_embedded = TSNE(n_components=2).fit_transform(params)
    plt.scatter([x for x,y in x_embedded], [y for x,y in x_embedded],c=values,s=10)
    plt.colorbar()
    plot_path = os.path.join(plot_directory,'tsne.png')
    plt.savefig(plot_path)
    plt.close()
    print('Saved plot %s' % plot_path)

def plot_tsne_smooth(results_directory, plot_directory, agent_name, n_planes=4):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import sklearn
    from sklearn.manifold import TSNE

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    scores = smoothen_scores_lsh(results_directory,agent_name,n_planes=n_planes)

    params = []
    values = []
    for p,s in scores.items():
        params.append(p)
        #values.append(np.log(s)) # Log so we can better distinguish smaller values
        values.append(s) # Log so we can better distinguish smaller values
    x_embedded = TSNE(n_components=2).fit_transform(params)
    plt.scatter([x for x,y in x_embedded], [y for x,y in x_embedded],
            c=values, s=5)
    plt.colorbar()
    plot_path = os.path.join(plot_directory,'tsne_smooth.png')
    plt.savefig(plot_path,dpi=200)
    plt.close()
    print('Saved plot %s' % plot_path)

def flatten_params(params):
    p = params
    ap = p.pop('agent_params',{})
    if 'directory' in p:
        del p['directory']
    for k,v in ap.items():
        p[k] = v
    return dict(p.items())

def compute_score(directory,sortby='mean',
        keys=['mean']):
    results = utils.get_all_results(directory)
    scores = defaultdict(lambda: [])
    for param,values in results:
        try:
            param = flatten_params(param)
            d = values['steps_to_reward']
            if len(d) < 100:
                continue
            m = np.mean(d[50:])
            s = {
                    'data': d,
                    'mean': m
            }
            scores[param['agent_name']].append((param,s))
        except:
            pass # Invalid file
    sorted_scores = {}
    for an in scores.keys():
        sorted_scores[an] = sorted(scores[an],key=lambda x: x[1][sortby])
    return sorted_scores

def compute_series_all(directory):
    results = utils.get_all_results(directory)
    series = defaultdict(lambda: [])
    for k,v in results:
        s = v['steps_to_reward']
        if len(s) < 100:
            continue
        series[k['agent_name']].append(s)
    return series

def compute_series_euclidean(directory,agent_name='ActorCritic',params={},radius=0.1):
    params = hyperparams.utils.param_to_vec(
            params, space[agent_name])
    params = torch.tensor(params)
    results = utils.get_all_results(directory)
    series = []
    for k,v in results:
        if d['agent_name'] != agent_name:
            continue
        k = hyperparams.utils.param_to_vec(k, space[agent_name])
        k = torch.tensor(k)
        if ((k-param)**2).sum() > radius:
            continue
        series.append(v['steps_to_reward'])
    return series

def compute_series_lsh(directory, iterations=10, n_planes=4):
    scores = compute_score(directory)
    output = defaultdict(lambda: {'series': [], 'bin_size': []})
    for agent_name in scores.keys():
        for _ in tqdm(range(iterations),desc='Computing series (LSH)'):
            bins = bin_lsh(scores[agent_name], space[agent_name],
                    n_planes=n_planes)
            score_bins = [np.nanmean([v['mean'] for k,v in b]) for b in bins]
            min_index = np.nanargmin(score_bins)
            min_series = np.array([v['data'] for k,v in bins[min_index]]).mean(0)
            output[agent_name]['series'].append(min_series)
            output[agent_name]['bin_size'].append(len(bins[min_index]))
        output[agent_name]['series'] = np.array(output[agent_name]['series']).mean(0)
    return output

def smoothen_scores_lsh(results_directory, agent_name, iterations=10, n_planes=4):
    scores = compute_score(results_directory)[agent_name]

    output = defaultdict(lambda: [])
    for _ in range(iterations):
        bins = bin_lsh(scores,space[agent_name],n_planes=n_planes)
        for b in bins:
            if len(b) == 0:
                continue
            score = np.nanmean([v['mean'] for k,v in b])
            for k,_ in b:
                output[tuple(k)].append(score)
    for k,v in output.items():
        output[k] = np.mean(output[k])
    return output

def smoothen_series_lsh(results_directory, agent_name, iterations=10,n_planes=4):
    data = compute_series(results_directory)[agent_name]

    output = defaultdict(lambda: 0)
    for _ in range(iterations):
        bins = bin_lsh(scores,space[agent_name],n_planes=n_planes)
        for b in bins:
            if len(b) == 0:
                continue
            series = np.nanmean([v['mean'] for k,v in b])
            for k,_ in b:
                output[tuple(k)] += score
    for k,v in output.items():
        output[k] /= iterations
    return output

def fit_gaussian_process(directory, agent_name):
    results = utils.get_all_results(directory)
    scores = defaultdict(lambda: [])
    x = []
    y = []
    for param,values in results:
        try:
            if param['agent_name'] != agent_name:
                continue

            param = flatten_params(param)
            vec = hyperparams.utils.param_to_vec(param,space[agent_name])

            d = values['steps_to_reward']
            if len(d) < 100:
                continue
            m = np.mean(d[50:])

            x.append(vec)
            y.append(m)
        except:
            pass # Invalid file
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    kernel = RBF()
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x,y)
    breakpoint()
    return x,y

def run():
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl-2'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)
    for agent_name in space.keys():
        space[agent_name]['directory'] = directory

    #series = compute_series_lsh(directory,iterations=100,n_planes=8)
    #data = defaultdict(lambda: {})
    #for an,s in series.items():
    #    #data['all']['x'] = range(0,p['total_steps'],p['epoch'])
    #    data[an]['x'] = range(0,100000,1000)
    #    data[an]['y'] = s['series']
    #    print(an,np.mean(s['series'][50:]),np.mean(s['bin_size']))
    #plot(data,plot_directory)

    #plot_tsne(directory, plot_directory, 'ActorCritic')
    #plot_tsne_smooth(directory, plot_directory, 'ActorCritic',n_planes=6)
    #plot_tsne_smooth(directory, plot_directory, 'HDQNAgentWithDelayAC_v2',n_planes=6)
    #plot_tsne_smooth(directory, plot_directory, 'HDQNAgentWithDelayAC_v3',n_planes=6)

    #run_hyperparam_search(space['ActorCritic'])
    #run_hyperparam_search(space['HDQNAgentWithDelayAC_v2'])
    run_hyperparam_search_extremes(space['HDQNAgentWithDelayAC_v2'])
    #run_hyperparam_search(space['HDQNAgentWithDelayAC_v3'])

    #param = sample_convex_hull(directory)
    #param = sample_lsh(directory, 'HDQNAgentWithDelayAC_v2', n_planes=8, perturbance=0.0)
    #param = sample_lsh(directory, 'HDQNAgentWithDelayAC_v2', n_planes=6, perturbance=0.05, scoring='improvement_prob', target_score=182.58163722924036)
    #param = sample_lsh(directory, 'ActorCritic', n_planes=8, perturbance=0.01)
    #param = hyperparams.utils.sample_hyperparam(space['HDQNAgentWithDelayAC_v2'])
    #run_trial(**param)

    #run_bayes_opt(directory,'ActorCritic')
    #run_bayes_opt(directory,'HDQNAgentWithDelayAC_v2')

    #s = smoothen_scores_lsh(directory, 'HDQNAgentWithDelayAC_v2')
    #pprint.pprint(sorted(s.values()))

    #count = 0
    #for v,save in utils.modify_all_results(directory):
    #    if 'steps_to_reward' not in v[1] or len(v[1]['steps_to_reward']) < 100:
    #        #save(None)
    #        count += 1
    #print(count)

    #x,y = fit_gaussian_process(directory, 'ActorCritic')
