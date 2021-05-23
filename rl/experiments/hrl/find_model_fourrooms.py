import numpy as np
import gym
import gym.wrappers
import torch
from tqdm import tqdm
import dill
import os
import itertools
from collections import defaultdict

from rl.agent.discrete_agent import TabularAgent
from rl.agent.policy import get_greedy_epsilon_policy
from rl import utils
from .model import QFunction

import gym_fourrooms.envs

class FixedGoalWrapper(gym.Wrapper):
    def __init__(self, env, x, y):
        super().__init__(env)
        self.x = x
        self.y = y

    def reset(self):
        obs = self.env.reset()
        self.env.goal = np.array([self.y, self.x])
        return np.array([obs[0],obs[1],self.y,self.x])

def run_trial_tabular(alpha, gamma, eps_b, eps_t, sigma, lam,
        directory=None, max_iters=5000, epoch=50, test_iters=1,
        env_name='gym_fourrooms:fourrooms-v0', verbose=False):
    args = locals()
    env = gym.make(env_name,fail_prob=0)
    env = FixedGoalWrapper(env,4,5)
    #env = gym.wrappers.TimeLimit(env,36)

    agent_file_name = os.path.join(directory,'agent.pkl')

    if os.path.isfile(agent_file_name):
        with open(agent_file_name,'rb') as f:
            agent = dill.load(f)
    else:
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
            #if epoch is not None and iters % epoch == 0:
            #    r = agent.test(env, test_iters, render=False, processors=1)
            #    rewards.append(r)
            r,_ = agent.run_episode(env)
            if r > 0:
                tqdm.write('Rewards: %f' % r)
            if verbose:
                num_states_covered = len(agent.learner.q)
                percent_states_covered = num_states_covered/((11**4-11-11+5)*4)
                tqdm.write('Percentage of states visited: %f' %
                        percent_states_covered)
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")
            raise e
    #except KeyboardInterrupt:
    #    pass

    print('Saving agent...')
    with open(agent_file_name,'wb') as f:
        dill.dump(agent,f)
    print('Saved')

    return agent

#def get_transition_matrix(eps=1/3):
#    env_map = gym_fourrooms.envs.fourrooms_env.env_map
#    directions = gym_fourrooms.envs.fourrooms_env.directions
#
#    # map from index to state
#    states = [(x,y,gx,gy) for x,y,gx,gy in zip(itertools.repeat(range(13),4))]
#    states_dict = {s:i for i,s in enumerate(states)}
#    t = torch.zeros([len(states)*4,len(states)*4])
#    for i,s0 in enumerate(states):
#        pos0 = [s[0],s[1]]
#        for a in range(4):
#            s1 = pos0+directions[a]
#            j = states_dict[(s1[0],s1[1],s0[2],s0[3])]
#            if env_map[s1[0],s1[1]]:
#                t[i+len(states)*a,j+len()]
#            # Check 
#            pass

def get_policy_as_string(agent, gx=None, gy=None):
    q = agent.learner.q

    if gx is None and gy is None:
        # Find goal state with the most data points
        count = defaultdict(lambda: 0)
        array = np.array # Needed for eval()
        for k,v in q.items():
            g = tuple(eval(k)[0][2:].tolist())
            count[g] += v
        most_freq_goal = sorted(count.items(),key=lambda c: c[1])[-1]
        gx,gy = most_freq_goal[0]

    dirs = "^>v<"
    output = ''
    env_map = gym_fourrooms.envs.fourrooms_env.env_map
    for y in range(13):
        for x in range(13):
            vals = [q[str((np.array([y,x,gy,gx]),a))] for a in range(4)]
            a = np.argmax(vals)
            if not env_map[y,x]:
                output += ' '
            elif x == gx and y == gy:
                output += 'G'
            else:
                output += dirs[a]
        output += '\n'
    return output

def compute_q_function(directory):
    # Compute tabular value function
    agent = run_trial_tabular(alpha=0.1, gamma=0.9, eps_b=0.1, eps_t=0,
            sigma=0, lam=0.5, verbose=True, max_iters=10, directory=directory)
    policy_string = get_policy_as_string(agent)

    print('----')
    print(policy_string)
    print('----')

    return agent.learner.q

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
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'hrl'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    q = compute_q_function(directory)
    return q

    ## Compute targets
    #input_values = torch.tensor([list(range(16))]).t().float()
    #output_values = torch.tensor([[q[str((state,action))] for action in range(4)] for state in range(16)])

    ## Try different models and see what can best represent all the values
    #models = [(4,4),(7,7),(8,8),(9,9),(10,10),(12,12),(15,15)]
    #performance = {}
    #for m in tqdm(models, desc='Testing models'):
    #    performance[m] = np.mean([test_model(m,iterations_per_test,input_values,output_values) for _ in range(tests_per_model)])
    #for x in sorted(list(performance.items()), key=lambda x: x[1]):
    #    print(x)
    #return x
