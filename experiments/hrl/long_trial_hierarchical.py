import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools
import time

from agent.dqn_agent import DQNAgent, HierarchicalDQNAgent
from agent.policy import get_greedy_epsilon_policy, greedy_action

from environment.wrappers import FrozenLakeToCoords

from .gridsearch_hierarchical import HRLWrapper, plot
from .model import QFunction

import utils

class DummyPolicy():
    def __call__(self,state):
        v0 = (state[:,0] > 1).view(-1,1).float()
        v1 = 1-v0
        return torch.cat((v0,v1),dim=1)
    def parameters(self):
        x = torch.tensor(0)
        return [x]

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
            #q_net=QFunction(layer_sizes=net_structure,input_size=2,output_size=num_options)
            q_net=DummyPolicy()
    )
    print_policy(agent)
    return

    env = HRLWrapper(env, options, test=False)
    test_env = HRLWrapper(test_env, options, test=True)

    def value_function(states):
        action_values = agent.q_net(states)
        optimal_actions = greedy_action(action_values)
        values = [options[o].q_net_target(s).max() for s,o in zip(states,optimal_actions)]
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

    # Create file to save results
    results_file_path = utils.save_results(args,
            {'rewards': [], 'state_action_values': []},
            directory=directory)

    rewards = []
    state_action_values = []
    state_option_values = []
    entropies = []
    done = True
    step_range = itertools.count()
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
                data = {'rewards': rewards, 
                        'state_action_values': state_action_values,
                        'state_option_values': state_option_values,
                        'entropies': entropies},
                utils.save_results(args, data, file_path=results_file_path)

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
            if len(options[action].replay_buffer) >= min_replay_buffer_size:
                options[action].train(batch_size=batch_size,iterations=1,value_function=value_function)

            # Next time step
            obs = obs2
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")
            raise
    except KeyboardInterrupt:
        pass

    return (args, rewards, state_action_values)

def print_policy(agent):
    states = list(itertools.product([0,1,2,3],[0,1,2,3]))
    vals = agent.q_net(torch.tensor(states).float())
    print(vals.argmax(dim=1).view(4,4))

def run():
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)

    run_trial(gamma=1,alpha=0.001,eps_b=0,eps_t=0,tau=0.001,net_structure=(10,10),num_options=2,batch_size=256,epoch=1000,test_iters=10,verbose=True,directory=directory)
    #run_trial(gamma=1,alpha=0.01,eps_b=0,eps_t=0,tau=0.01,net_structure=(),num_options=1,batch_size=10,epoch=10, test_iters=3,verbose=True,directory=directory)
    plot(results_dir=directory,plot_dir=plot_directory)
