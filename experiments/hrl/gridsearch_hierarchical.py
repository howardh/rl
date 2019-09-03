import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools

#from agent.hierarchical_dqn_agent import HierarchicalDQNAgent
from agent.dqn_agent import DQNAgent, HierarchicalDQNAgent
from agent.policy import get_greedy_epsilon_policy, greedy_action
from environment.wrappers import FrozenLakeToCoords
import utils

from .model import QFunction

class HRLWrapper(gym.Wrapper):
    def __init__(self, env, options, test=False):
        super().__init__(env)
        self.options = options
        self.test = test
        self.action_space = gym.spaces.Discrete(len(options))
        #self.discount_factor = discount_factor
        #self.rewards_since_last = None
        #self.last_option = None
        self.current_obs = None
        #self.prev_transition = [None]*len(options)

    def step(self, action):
        option = self.options[action]
        # Select primitive action from option
        primitive_action = option.act(self.current_obs,testing=self.test)
        # Save state-action value
        self.last_sa_value = option.last_vals.squeeze()[primitive_action].item()
        # Transition
        obs, reward, done, info = self.env.step(primitive_action)
        # Save transitions
        if not self.test:
            self.options[action].observe_step(self.current_obs,primitive_action,reward,obs,done)
        # Save current obs
        self.current_obs = obs
        return obs, reward, done, info

    def step2(self, action):
        option = self.options[action]
        # Select primitive action from option
        primitive_action = option.act(self.current_obs)
        # Transition
        obs, reward, done, info = self.env.step(primitive_action)
        # Save transitions
        if self.prev_transition[action] is not None:
            o1,a,rs,o2,t = self.prev_transition[action]
            g = self.discount_factor**len(rs)
            rs_discounted_sum = sum((r*self.discount_factor**i for i,r in enumerate(rs)))
            option[action].observe_step(o1,a,rs_discounted_sum,o2,t,g)
        self.prev_transition[action] = (self.current_obs,primitive_action,[],obs,done)
        # Update target for each option
        for _,_,r,_,_ in self.prev_transition:
            r.append(reward)
        self.current_obs = obs
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.current_obs = self.env.reset(**kwargs)
        return self.current_obs

def run_trial(gamma, alpha, eps_b, eps_t, tau, directory=None,
        net_structure=[2,3,4], num_options=4,
        env_name='FrozenLake-v0', batch_size=32, min_replay_buffer_size=1000,
        max_steps=5000, epoch=50, test_iters=1, verbose=False):
    args = locals()
    env = gym.make(env_name)
    env = FrozenLakeToCoords(env)
    test_env = gym.make(env_name)
    test_env = FrozenLakeToCoords(test_env)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    def create_option():
        return HierarchicalDQNAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                learning_rate=alpha,
                discount_factor=gamma,
                polyak_rate=tau,
                device=device,
                behaviour_policy=get_greedy_epsilon_policy(eps_b),
                target_policy=get_greedy_epsilon_policy(eps_t),
                q_net=QFunction(layer_sizes=net_structure,input_size=2)
        )
    options = [create_option() for _ in range(num_options)]
    agent = DQNAgent(
            action_space=gym.spaces.Discrete(num_options),
            observation_space=env.observation_space,
            learning_rate=alpha,
            discount_factor=gamma,
            polyak_rate=tau,
            device=device,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t),
            q_net=QFunction(layer_sizes=net_structure,input_size=2,output_size=num_options)
    )

    env = HRLWrapper(env, options, test=False)
    test_env = HRLWrapper(test_env, options, test=True)

    def value_function(states):
        action_values = agent.q_net(states)
        optimal_actions = greedy_action(action_values)
        values = [options[o].q_net(s).max() for s,o in zip(states,optimal_actions)]
        return torch.tensor(values)

    def test(env, iterations, max_steps=np.inf, render=False, record=True, processors=1):
        def test_once(env, max_steps=np.inf, render=False):
            reward_sum = 0
            sa_vals = [[] for _ in range(env.action_space.n)]
            so_vals = []
            option_freq = np.array([0]*num_options)
            obs = env.reset()
            o = agent.act(obs, testing=True)
            for steps in itertools.count():
                if steps > max_steps:
                    break
                o = agent.act(obs, testing=True)
                so_vals.append(agent.get_state_action_value(obs,o))
                obs, reward, done, _ = env.step(o)
                sa_vals[o].append(env.last_sa_value)
                reward_sum += reward
                if render:
                    env.render()
                if done:
                    break
            prob = option_freq/option_freq.sum()
            entropy = -sum([p*np.log(p) if p > 0 else 0 for p in prob])
            return reward_sum, [np.mean(v) for v in sa_vals], np.mean(so_vals), entropy
        rewards = []
        sa_vals = []
        so_vals = []
        entropies = []
        for i in range(iterations):
            r,sav,sov,e = test_once(env, render=render, max_steps=max_steps)
            rewards.append(r)
            sa_vals.append(sav)
            so_vals.append(sov)
            entropies.append(e)
        return rewards, sa_vals, so_vals, entropies

    rewards = []
    state_action_values = []
    state_option_values = []
    entropies = []
    done = True
    step_range = range(0,max_steps+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                r,sa_vals,so_vals,e = test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                state_action_values.append(sa_vals)
                state_option_values.append(so_vals)
                entropies.append(e)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f \t SA-V: %f \t SO-V: %f \t Ent: %f' % (steps, np.mean(r), np.mean(sa_vals), np.mean(so_vals), np.mean(e)))

            # Linearly Anneal epsilon (Options only)
            for o in options:
                o.behaviour_policy = get_greedy_epsilon_policy((1-min(steps/1000000,1))*(1-eps_b)+eps_b)

            # Run step
            if done:
                obs = env.reset()
            action = agent.act(obs)

            obs2, reward2, done, _ = env.step(action)
            agent.observe_step(obs, action, reward2, obs2, terminal=done)

            # Update weights
            #if steps >= min_replay_buffer_size:
            #    agent.train(batch_size=batch_size,iterations=1)
            if len(options[action].replay_buffer) < min_replay_buffer_size:
                continue
            options[action].train(batch_size=batch_size,iterations=1,value_function=value_function)

            # Next time step
            obs = obs2
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")
            raise

    data = {'rewards': rewards, 
            'state_action_values': state_action_values,
            'state_option_values': state_option_values,
            'entropies': entropies},
    utils.save_results(args, data, directory=directory)
    return (args, rewards, state_action_values)

def run_gridsearch(proc=1):
    directory = os.path.join(utils.get_results_directory(),__name__)
    params = {
            'gamma': [1],
            'alpha': [0.01],
            'eps_b': [0],
            'eps_t': [0],
            'tau': [0.01],
            'env_name': ['FrozenLake-v0'],
            'batch_size': [256],
            'min_replay_buffer_size': [10000],
            'max_steps': [100000],
            'epoch': [1000],
            'test_iters': [5],
            'verbose': [False],
            'net_structure': [(10,10),(8,8),(6,6),(4,4),(2,2),(10,),(8,),(6,)],
            'num_options': [2,4,8],
            'directory': [directory]
    }
    funcs = utils.gridsearch(params, run_trial)
    utils.cc(funcs,proc=proc)
    return utils.get_all_results(directory)

def plot(results_dir, plot_dir):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    def reduce(results,s=[]):
        return s + [results]
    results = utils.get_all_results_reduce(results_dir, reduce, [])

    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    file_mapping = {}
    for i,(k,v) in enumerate(results.items()):
        trial_rewards = []
        trial_predicted_sa_values = []
        trial_predicted_so_values = []
        trial_entropies = []
        for (trial,) in v:
            trial_rewards.append([np.mean(epoch) for epoch in trial['rewards']])
            trial_predicted_sa_values.append([np.mean(epoch, axis=0) for epoch in trial['state_action_values']])
            trial_predicted_so_values.append([np.mean(epoch) for epoch in trial['state_option_values']])
        params = dict(k)
        y1 = np.mean(trial_rewards,axis=0)
        y2 = np.mean(trial_predicted_sa_values,axis=0)
        y3 = np.mean(trial_predicted_so_values,axis=0)
        x = list(range(0,y1.shape[0]*params['epoch'],params['epoch']))

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(10,4)
        # Plot
        ax1.plot(x,y1)
        ax1.plot(x,y2)
        ax1.plot(x,y3)
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
            for (trial,) in v:
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
            for (trial,) in v:
                trial_rewards.append([np.mean(epoch) for epoch in trial['rewards']])
            performance[k] = np.max(np.mean(trial_rewards,axis=0))
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

    # Plot average performance over time
    plot_dir = os.path.join(utils.get_results_root_directory(),'plots',__name__)
    plot(directory, plot_dir)
