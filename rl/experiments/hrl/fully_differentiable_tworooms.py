import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools

from rl.agent.hdqn_agent import HDQNAgentWithDelayAC
from rl.agent.policy import get_greedy_epsilon_policy

from .model import QFunction, PolicyFunction
from .long_trial import plot

from rl import utils

two_rooms_map = """
xxxxxxxxxx
x    x   x
x    x   x
x        x
x    x   x
x    x   x
xxxxxxxxxx"""

def print_policy(agent):
    goals = [(1,1),(5,2),(1,6),(5,7)]
    map_array = np.array([list(r) for r in two_rooms_map.split('\n')[1:]])
    y_len = len(map_array)
    x_len = len(map_array[0])
    output_arrays = []
    empty_column = np.array([[' ']]*y_len)
    for gy,gx in goals:
        output_array = map_array.copy()
        states = list(itertools.product(range(y_len),range(x_len),[gy],[gx]))
        vals = agent.controller_net(torch.tensor(states).float())
        actions = vals.argmax(dim=1).view(y_len,x_len)
        for y,x in itertools.product(range(y_len),range(x_len)):
            if y == gy and x == gx:
                output_array[y,x] = 'G'
            elif output_array[y,x] == ' ':
                output_array[y,x] = str(actions[y,x].item())
        output_arrays.append(output_array)
        output_arrays.append(empty_column)
    concated_arrays = np.concatenate(output_arrays[:-1],axis=1)
    output_str = '\n'.join([' '.join(row) for row in concated_arrays])
    tqdm.write(output_str)
    return output_str

def print_policy2(agent):
    goals = [(1,1),(5,2),(1,6),(5,7)]
    dir_arrows = '↑→↓←'
    map_array = np.array([list(r) for r in two_rooms_map.split('\n')[1:]])
    y_len = len(map_array)
    x_len = len(map_array[0])
    output_arrays = []
    empty_column = np.array([[' ']]*y_len)
    for gy,gx in goals:
        output_array = map_array.copy()
        states = list(itertools.product(range(y_len),range(x_len),[gy],[gx]))
        vals = agent.controller_net(torch.tensor(states).float())
        actions = vals.argmax(dim=1).view(y_len,x_len)
        for y,x in itertools.product(range(y_len),range(x_len)):
            if y == gy and x == gx:
                output_array[y,x] = 'G'
            elif output_array[y,x] == ' ':
                s = torch.tensor([[y,x,gy,gx]]).float()
                prim_action = agent.subpolicy_nets[actions[y,x].item()](s).argmax()
                output_array[y,x] = dir_arrows[prim_action]
        output_arrays.append(output_array)
        output_arrays.append(empty_column)
    concated_arrays = np.concatenate(output_arrays[:-1],axis=1)
    output_str = '\n'.join([' '.join(row) for row in concated_arrays])
    tqdm.write(output_str)
    return output_str

def run_trial(gamma, alpha, eps_b, eps_t, tau, directory=None,
        controller_net_structure=[2,3,4], subpolicy_net_structure=[3],
        num_options=4,
        env_name='gym_fourrooms:fourrooms-v0', batch_size=32,
        min_replay_buffer_size=1000, epoch=50, test_iters=1, verbose=False,
        max_steps=float('inf')):
    args = locals()
    env = gym.make(env_name,env_map=two_rooms_map,fail_prob=0.1)
    env = gym.wrappers.TimeLimit(env,15)
    test_env = gym.make(env_name,env_map=two_rooms_map,fail_prob=0.1)
    test_env = gym.wrappers.TimeLimit(test_env,15)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = HDQNAgentWithDelayAC(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=alpha,
            discount_factor=gamma,
            polyak_rate=tau,
            device=device,
            behaviour_epsilon=eps_b,
            controller_net=PolicyFunction(layer_sizes=controller_net_structure,input_size=4,output_size=num_options),
            subpolicy_nets=[PolicyFunction(layer_sizes=subpolicy_net_structure,input_size=4) for _ in range(num_options)],
            q_net=QFunction(layer_sizes=(15,15),input_size=4,output_size=4),
    )

    # Create file to save results
    results_file_path = utils.save_results(args,
            {'rewards': [], 'state_action_values': []},
            directory=directory)

    rewards = []
    state_action_values = []
    done = True

    if max_steps == float('inf'):
        step_range = itertools.count()
    else:
        step_range = range(max_steps)
    if verbose:
        step_range = tqdm(step_range)

    try:
        training_rewards = []
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                r,sa_vals = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                state_action_values.append(sa_vals)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f \t Train rewards: %f' %
                            (steps, np.mean(r), np.mean(training_rewards)))
                    print_policy(agent)
                    print_policy2(agent)
                training_rewards = []
                utils.save_results(args,
                        {'rewards': rewards, 'state_action_values': state_action_values},
                        file_path=results_file_path)

            # Linearly Anneal epsilon
            agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b

            # Run step
            if done:
                obs = env.reset()
                agent.observe_change(obs, None)
            obs, reward, done, _ = env.step(agent.act())
            agent.observe_change(obs, reward, terminal=done)
            training_rewards.append(reward)

            # Update weights
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")
            raise e
    except KeyboardInterrupt:
        tqdm.write('Keyboard Interrupt')

    return (args, rewards, state_action_values)

def run():
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)

    run_trial(gamma=0.9,alpha=0.001,eps_b=0.05,eps_t=0,tau=0.001,controller_net_structure=(2,2),subpolicy_net_structure=(3,),batch_size=256,epoch=1000,test_iters=10,verbose=True,directory=directory,num_options=3)
    plot(results_directory=directory,plot_directory=plot_directory)
