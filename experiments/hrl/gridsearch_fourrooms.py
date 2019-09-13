import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os

from agent.dqn_agent import DQNAgent
from agent.policy import get_greedy_epsilon_policy
from environment.wrappers import FrozenLakeToCoords
import utils

from .model import QFunction

def run_trial(gamma, alpha, eps_b, eps_t, tau, directory=None,
        net_structure=[2,3,4],
        env_name='FrozenLake-v0', batch_size=32, min_replay_buffer_size=1000,
        max_steps=5000, epoch=50, test_iters=1, verbose=False):
    args = locals()
    env = gym.make(env_name)
    env = gym.wrappers.TimeLimit(env,36)
    test_env = gym.make(env_name)
    test_env = gym.wrappers.TimeLimit(test_env,36)

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
            q_net=QFunction(layer_sizes=net_structure,input_size=4)
    )

    rewards = []
    state_action_values = []
    done = True
    step_range = range(0,max_steps+1)
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

    utils.save_results(
            args,
            {'rewards': rewards, 'state_action_values': state_action_values},
            directory=directory)
    return (args, rewards, state_action_values)

def run_gridsearch(proc=1):
    directory = os.path.join(utils.get_results_directory(),__name__)
    params = {
            'gamma': [1],
            'alpha': [0.1,0.01,0.001],
            'eps_b': [0, 0.1],
            'eps_t': [0],
            'tau': [0.1, 0.01, 0.001],
            'env_name': ['gym_fourrooms:fourrooms-v0'],
            'batch_size': [32, 64, 128, 256],
            'min_replay_buffer_size': [1000],
            'max_steps': [100000],
            'epoch': [1000],
            'test_iters': [5],
            'verbose': [False],
            'net_structure': [(10,10),(20,20)],
            'directory': [directory]
    }
    funcs = utils.gridsearch(params, run_trial)
    utils.cc(funcs,proc=proc)
    return utils.get_all_results(directory)

def run(proc=3):
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    # Run gridsearch
    run_gridsearch(proc=proc)
    # Look through params for best performance
    def reduce(results,s=[]):
        return s + [results]
    directory = os.path.join(utils.get_results_directory(),__name__)
    results = utils.get_all_results_reduce(directory, reduce, [])

    def compute_performance_max_cum_mean(results):
        performance = {}
        for k,v in results.items():
            """
            v =
            - Trial1
                - Epoch 1
                    - Test iter 1
                    - Test iter 2
                    - ...
                - Epoch 2
                    - ...
                - ...
            - Trial 2
                - etc.
            """
            # Average all iterations under an epoch
            # Do a cumulative mean over all epochs
            # Take a max over the cumulative means for each trial
            # Take a mean over all max cum means over all trials
            max_means = []
            for trial in v:
                mean_rewards = [np.mean(epoch) for epoch in trial['rewards']]
                cum_mean = np.cumsum(mean_rewards)/np.arange(1,len(mean_rewards)+1)
                max_mean = np.max(cum_mean)
                max_means.append(max_mean)
            mean_max_mean = np.mean(max_means)
            performance[k] = mean_max_mean
        return performance
    def compute_performance_abs_max(results):
        performance = {}
        for k,v in results.items():
            trial_rewards = []
            for trial in v:
                trial_rewards.append([np.mean(epoch) for epoch in trial['rewards']])
            performance[k] = np.max(np.mean(trial_rewards,axis=0))
        return performance
    def compute_performance_mean(results):
        performance = {}
        for k,v in results.items():
            trial_rewards = []
            for trial in v:
                trial_rewards.append([np.mean(epoch) for epoch in trial['rewards']])
            performance[k] = np.mean(trial_rewards)
        return performance

    print('-'*80)

    performance = compute_performance_abs_max(results)
    best_param, best_performance = max(performance.items(), key=lambda x: x[1])
    print('Performance by best performance reached at any point')
    print('Best parameter set:', best_param)
    print('Best performance:', best_performance)

    print('-'*80)

    performance = compute_performance_max_cum_mean(results)
    best_param, best_performance = max(performance.items(), key=lambda x: x[1])
    print('Performance by sample complexity')
    print('Best parameter set:', best_param)
    print('Best performance:', best_performance)

    print('-'*80)

    performance = compute_performance_mean(results)
    best_param, best_performance = max(performance.items(), key=lambda x: x[1])
    print('Performance by overall mean')
    print('Best parameter set:', best_param)
    print('Best performance:', best_performance)

    print('-'*80)

    # Plot average performance over time
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plot_dir = os.path.join(utils.get_results_root_directory(),'plots',__name__)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    file_mapping = {}
    plot_data = {}
    max_y = 0
    for i,(k,v) in enumerate(results.items()):
        trial_rewards = []
        trial_predicted_values = []
        for trial in v:
            trial_rewards.append([np.mean(epoch) for epoch in trial['rewards']])
            trial_predicted_values.append([np.mean(epoch) for epoch in trial['state_action_values']])
        params = dict(k)
        y1 = np.mean(trial_rewards,axis=0)
        y2 = np.mean(trial_predicted_values,axis=0)
        x = list(range(0,params['max_steps']+1,params['epoch']))
        plot_data[k] = (x,y1,y2)
        max_y = max(max_y, np.max(y1), np.max(y2))

    for i,(k,(x,y1,y2)) in enumerate(plot_data.items()):
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(10,4)
        # Plot
        ax1.plot(x,y1)
        ax1.plot(x,y2)
        ax1.set_ylim([0,1])
        ax1.set_title('[Insert Title Here]')
        ax1.grid(True,which='both',axis='both',color='grey')
        # Show params
        ax2.set_axis_off()
        for j,(pname,pval) in enumerate(sorted(dict(k).items(), key=lambda x: x[0], reverse=True)):
            ax2.text(0,j/len(k),'%s: %s' % (pname, pval))
        file_name = os.path.join(plot_dir,'%d.png'%i)
        fig.savefig(file_name)
        plt.close(fig)

        print('Saved', file_name)
        file_mapping[k] = file_name
