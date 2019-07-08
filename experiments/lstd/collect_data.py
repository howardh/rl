import collections
from collections import defaultdict
import gym
import numpy as np
from tqdm import tqdm
import torch
import dill
import os
import shelve

from agent.lstd_agent import LSTDAgent
from agent.policy import get_greedy_epsilon_policy
import utils

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def compute_k_means(x, y, k, n):
    keys = [slice(i,i+k) for i in range(0,len(x)-k,n)]
    key_means = [np.mean(x[k]) for k in keys]
    vals = [np.mean(y[k]) for k in keys]
    return key_means, vals

def run_trial(gamma, upd_freq, eps_b, eps_t, sigma, lam,
        directory=None, min_steps=1000, max_steps=5000, points_per_step=500,
        epoch=50, test_iters=1, env_name='FrozenLake-v0',
        plot_results=False, verbose=False):
    args = locals()
    env = gym.make(env_name)
    test_env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = LSTDAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t),
            discount_factor=gamma,
            use_importance_sampling=False,
            use_traces=True,
            trace_factor=lam,
            sigma=sigma,
            device=device
    )
    initial_a_mat = None
    initial_b_mat = None
    new_a_mat = None
    new_b_mat = None
    # Keyed on number of transitions we're adding to the matrix
    weighting_data = dict()
    optimal_lambdas = dict()

    done = True
    step_range = range(0,max_steps+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        for steps in step_range:
            # Run step
            if done:
                obs = env.reset()
            action = agent.act(obs)

            obs2, reward2, done, _ = env.step(action)
            agent.learner.observe_step(obs, action, reward2, obs2, terminal=done)

            # Save matrices
            if steps == min_steps:
                if verbose:
                    tqdm.write('Min steps reached.')
                initial_a_mat = agent.learner.a_mat.clone()
                initial_b_mat = agent.learner.b_mat.clone()
                agent.learner.a_mat *= 0
                agent.learner.b_mat *= 0
            # Compute optimal weighting
            if steps > min_steps:
                num_additional_steps = steps-min_steps
                weighting_data[num_additional_steps] = []
                # Save matrices
                new_a_mat = agent.learner.a_mat.clone()
                new_b_mat = agent.learner.b_mat.clone()
                # Collect data
                r = np.arange(0,1,1/points_per_step)
                if verbose:
                    r = tqdm(r)
                for l in r:
                    agent.learner.a_mat = (1-l)*initial_a_mat+l*new_a_mat
                    agent.learner.b_mat = (1-l)*initial_b_mat+l*new_b_mat
                    agent.update_weights()
                    r = agent.test(test_env, test_iters, render=False, processors=1)
                    weighting_data[num_additional_steps].append([l,r[0]])
                # Find parabola of best fit and get optimal weighting
                x = [x for x,y in weighting_data[num_additional_steps]]
                y = [y for x,y in weighting_data[num_additional_steps]]
                a,b,c = np.polyfit(x,y,2)
                if a > 0:
                    # Upright parabola
                    # Evaluate the endpoints to see which is better
                    y0 = c
                    y1 = a+b+c
                    if y0 > y1:
                        h = 0
                    else:
                        h = 1
                else:
                    # Find the parabola vertex
                    h = -b/(2*a)
                optimal_lambda = h
                # Record optimal weighting for this number of new transitions
                optimal_lambdas[num_additional_steps] = optimal_lambda
                # Save plot
                if plot_results:
                    # Scatter plot
                    plt.scatter(x,y)
                    # Polynomial of best fit
                    poly_x = np.arange(0,1,0.01)
                    poly_y = [a*(x**2)+b*x+c for x in poly_x]
                    plt.plot(poly_x,poly_y,'g')
                    # k-means plot
                    k_means_x, k_means_y = compute_k_means(x,y,200,5)
                    plt.plot(k_means_x,k_means_y,'r')
                    # Draw plots
                    plot_dir = directory
                    if not os.path.exists(plot_dir):
                        os.makedirs(plot_dir)
                    plt.savefig(os.path.join(
                        directory,'%d.png'%num_additional_steps))
                    plt.close()
                # Restore matrices and continue
                agent.learner.a_mat = initial_a_mat
                agent.learner.b_mat = initial_b_mat
                agent.update_weights()
                agent.learner.a_mat = new_a_mat.clone()
                agent.learner.b_mat = new_b_mat.clone()

            # Next time step
            obs = obs2
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")
    except KeyboardInterrupt:
        tqdm.write("Keyboard Interrupt")

    # Save data 
    data = {
            'args': args,
            'weighting_data': weighting_data,
            'optimal_lambdas': optimal_lambdas
    }
    if directory is not None:
        file_name, file_num = utils.find_next_free_file(
                "results", "pkl", directory)
        with open(file_name, "wb") as f:
            dill.dump(data, f)
    return data
