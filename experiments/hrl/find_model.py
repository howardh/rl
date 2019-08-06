import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools
import Levenshtein

from agent.discrete_agent import TabularAgent
from agent.policy import get_greedy_epsilon_policy
import utils
from .model import QFunction

def run_trial_tabular(alpha, gamma, eps_b, eps_t, sigma, lam,
        directory=None, max_iters=5000, epoch=50, test_iters=1,
        env_name='FrozenLake-v0', verbose=False):
    args = locals()
    env = gym.make(env_name)

    agent = TabularAgent(
            action_space=env.action_space,
            discount_factor=gamma,
            learning_rate=alpha,
            trace_factor=lam,
            sigma=sigma,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t))

    rewards = []
    step_range = range(0,max_iters+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        for iters in step_range:
            if epoch is not None and iters % epoch == 0:
                r = agent.test(env, test_iters, render=False, processors=1)
                rewards.append(r)
            agent.run_episode(env)
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")
            raise e

    # TODO: Save value function
    return agent

def get_policy_as_string(agent):
    q = agent.learner.q

    dirs = "<v>^"
    holes = [5,7,11,12]
    output = ''
    for x in range(16):
        vals = [q[str((x,a))] for a in range(4)]
        a = np.argmax(vals)
        if x in holes:
            output += ' '
        else:
            output += dirs[a]
        if x in [3,7,11]:
            output += '\n'
    return output

def compute_q_function(file_name='qfunction.pkl'):
    # Compute tabular value function
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            return dill.load(f)
    else:
        while True:
            agent = run_trial_tabular(alpha=0.1, gamma=1, eps_b=0.1, eps_t=0, sigma=0, lam=0, verbose=True)
            policy_string = get_policy_as_string(agent)

            print('----')
            print(policy_string)
            print('----')

            expected_policy = '<^^^\n< > \n^v<\n >><'
            policy_dist = Levenshtein.distance(policy_string, expected_policy)
            if policy_dist > 5:
                print('Policy found is too far from optimal policy. Trying again.')
                continue
            else:
                with open(file_name, 'wb') as f:
                    dill.dump(agent.learner.q, f)
                return q

def test_model(m, iterations, input_values, output_values):
    net = QFunction(m)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    lowest_loss = np.inf
    for i in tqdm(range(iterations), desc='Testing %s' % str(m)):
        loss = ((net(input_values) - output_values)**2).mean()
        #print(i, loss.item())
        if lowest_loss > loss.item():
            lowest_loss = loss.item()
        if loss < 1e-6:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return lowest_loss

def run(iterations_per_test=10000,tests_per_model=10):
    q = compute_q_function()

    # Compute targets
    input_values = torch.tensor([list(range(16))]).t().float()
    output_values = torch.tensor([[q[str((state,action))] for action in range(4)] for state in range(16)])

    # Try different models and see what can best represent all the values
    models = [(4,4),(7,7),(8,8),(9,9),(10,10),(12,12),(15,15)]
    performance = {}
    for m in tqdm(models, desc='Testing models'):
        performance[m] = np.mean([test_model(m,iterations_per_test,input_values,output_values) for _ in range(tests_per_model)])
    for x in sorted(list(performance.items()), key=lambda x: x[1]):
        print(x)
    return x
