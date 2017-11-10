import gym
import numpy as np
import datetime
import dill
import os
import itertools
from tqdm import tqdm 

from agent.discrete_agent import TabularAgent
from agent.lstd_agent import LSTDAgent
from agent.rbf_agent import RBFAgent

import mountaincar 
import mountaincar.features
import mountaincar.utils

#from mountaincar import exp1
import utils

def tabular_control(discount_factor, learning_rate, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, epoch, max_iters, test_iters):
    env_name = 'MountainCar-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1,2])
    agent = TabularAgent(
            action_space=action_space,
            features=mountaincar.features.get_one_hot(num_pos,num_vel),
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            initial_value=initial_value
    )
    agent.set_behaviour_policy("%f-epsilon" % behaviour_eps)
    #agent.set_behaviour_policy(mountaincar.utils.get_one_hot_optimal_policy(num_pos, num_vel, 0.75))
    #agent.set_behaviour_policy(utils.less_optimal_policy)
    agent.set_target_policy("%f-epsilon" % target_eps)

    rewards = []
    steps_to_learn = None
    iters = 0
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        rewards.append(r)
        print("... %d\r" % iters, end='')
        if r != -200:
            print("Iteration: %d\t Reward: %d"%(iters, r))
        if epoch is not None and iters % epoch == 0:
            print("Testing...")
            #print(agent.learner.weights.transpose())
            rewards = agent.test(e, test_iters, render=False, processors=1)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            if np.mean(rewards) >= -110:
                if steps_to_learn is None:
                    steps_to_learn = iters
        if iters > max_iters:
            break
    return rewards,steps_to_learn

def parse_results(directory):
    # List files in directory
    # Load all files
    # From each file...
    #   Compute the mean of rewards
    #   Add to dictionary
    # Sort dictionary by mean rewards and print top 10 along with scores
    # Sort dictionary by time to learn and print top 10 along with scores

    results = []
    file_names = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    for file_name in file_names:
        with open(os.path.join(directory,file_name), 'rb') as f:
            x = dill.load(f)
            if x[2] is None:
                results.append((x[0], file_name, np.mean(x[1]), np.inf))
            else:
                results.append((x[0], file_name, np.mean(x[1]), x[2]))
    print("\nSorting by mean reward...")
    results.sort(key=lambda x: x[2], reverse=True)
    for i in range(min(10,len(results))):
        print(results[i][2])
        print("\t%s" % results[i][0])
        print("\t%s" % results[i][1])
        print("\t%s" % results[i][3])

    print("\nSorting by time to learn...")
    results.sort(key=lambda x: x[3], reverse=True)
    for i in range(min(10,len(results))):
        print(results[i][3])
        print("\t%s" % results[i][0])
        print("\t%s" % results[i][1])
        print("\t%s" % results[i][2])

def rbf_control(discount_factor, learning_rate, initial_value, num_pos,
        num_vel, behaviour_eps, target_eps, epoch, max_iters, test_iters):
    global agent
    env_name = 'MountainCar-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1,2])
    obs_space = np.array([[-1.2, .6], [-0.07, 0.07]])
    def norm(x):
        return np.array([(s-r[0])/(r[1]-r[0]) for s,r in zip(x, obs_space)])
    agent = RBFAgent(
            action_space=action_space,
            observation_space=np.array([[0,1],[0,1]]),
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            initial_value=initial_value,
            features=norm
    )
    agent.set_behaviour_policy("%f-epsilon" % behaviour_eps)
    agent.set_target_policy("%f-epsilon" % target_eps)

    rewards = []
    steps_to_learn = None
    iters = 0
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        rewards.append(r)
        print("... %d\r" % iters, end='')
        if r != -200:
            print("Iteration: %d\t Reward: %d"%(iters, r))
        if epoch is not None and iters % epoch == 0:
            print("Testing...")
            #print(agent.learner.weights.transpose())
            rewards = agent.test(e, test_iters, render=False, processors=1)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            if np.mean(rewards) >= -110:
                if steps_to_learn is None:
                    steps_to_learn = iters
        if iters > max_iters:
            break
    return rewards,steps_to_learn

def lstd_control():
    env_name = 'MountainCar-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1,2])
    #agent = LSTDAgent(
    #        action_space=action_space,
    #        num_features=mountaincar.features.ONE_HOT_NUM_FEATURES,
    #        features=mountaincar.features.one_hot,
    #        #num_features=mountaincar.features.IDENTITY_NUM_FEATURES,
    #        #features=mountaincar.features.identity,
    #        discount_factor=1,
    #        use_importance_sampling=False,
    #        use_traces=False,
    #        sigma=None,
    #        trace_factor=None,
    #)
    agent = TabularAgent(
            action_space=action_space,
            features=mountaincar.features.one_hot,
            discount_factor=1,
            learning_rate=0.05
    )
    #agent.set_behaviour_policy("1.0-epsilon")
    agent.set_behaviour_policy(mountaincar.utils.less_optimal_policy2)
    #agent.set_behaviour_policy(utils.less_optimal_policy)
    agent.set_target_policy("0.1-epsilon")

    iters = 0
    #for _ in range(100):
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        if r != -200:
            break
        if iters % 100 == 0:
            print("...")
    print("Yay!")
    #agent.set_behaviour_policy("0.1-epsilon")
    while True:
        iters += 1
        r,s = agent.run_episode(e)
        #if r != -200:
        #    print("%d %d" % (iters, r))
        if iters % 100 == 0:
            #agent.update_weights()
            print("Testing...")
            #print(agent.learner.weights.transpose())
            rewards = agent.test(e, 100, render=False, processors=3)
            print("Iteration %d\t Rewards: %f" % (iters, np.mean(rewards)))
            if np.mean(rewards) >= -110:
                break
        if iters > 3000:
            break

def lstd_control_steps():
    env_name = 'CartPole-v0'
    e = gym.make(env_name)
    start_time = datetime.datetime.now()

    action_space = np.array([0,1])
    agent = LSTDAgent(
            action_space=action_space,
            num_features=cartpole.features.IDENTITY_NUM_FEATURES,
            discount_factor=0.99,
            features=cartpole.features.identity,
            use_importance_sampling=False,
            use_traces=False,
            sigma=0,
            trace_factor=0.5,
    )
    agent.set_behaviour_policy(utils.optimal_policy)
    agent.set_target_policy("0-epsilon")

    steps = 0
    while True:
        steps += 1
        agent.run_step(e)
        if steps % 5000 == 0:
            agent.update_weights()
            rewards = agent.test(e, 100, render=False, processors=3)
            print("Steps %d\t Rewards: %f" % (steps, np.mean(rewards)))
            print(agent.learner.weights.transpose())
            if np.mean(rewards) >= 190:
                break

if __name__ == "__main__":
    #rbf_control(1, 4.7, 0, 8, 8, 0, 0, None, 3000, 0)
    rbf_control(1, 0.5, 0, 8, 8, 0, 0, None, 3000, 0)
    #import torch
    #from torch.autograd import Variable

    #centres = list(itertools.product([-1,-0.333,0.333,1],[-1,-0.333,0.333,1]))
    ##weights = [1]*len(centres)
    #weights = np.random.rand(len(centres))-0.5

    #s = Variable(torch.Tensor([2]).float(), requires_grad=True)
    #c = Variable(torch.from_numpy(np.array(centres,dtype='float32')), requires_grad=True)
    #w = Variable(torch.from_numpy(np.array([weights],dtype='float32').transpose()), requires_grad=True)

    #class RBFFunction(torch.autograd.Function):
    #    def forward(self, x, c, w, s=None):
    #        # f(x,c,w) = w*exp(-s*(c-x)^2)
    #        diff = c-x
    #        dist = (diff*diff).sum(dim=1, keepdim=True)
    #        v = torch.exp(-1*s*dist)
    #        self.dist = dist
    #        self.unweighted_rbf = v
    #        y = torch.mm(w.t(),v)
    #        self.save_for_backward(x,c,w,s,y)
    #        return y

    #    def backward(self, grad_output):
    #        # df/dx = w*exp(-s*(c-x)^2)*(-s)*2*(c-x)*(-1)
    #        # df/dc = w*exp(-s*(c-x)^2)*(-s)*2*(c-x)
    #        # df/dw = exp(-s*(c-x)^2)
    #        # f(r) = w*exp(-s*r^2)
    #        x,c,w,s,y = self.saved_variables
    #        v = self.unweighted_rbf
    #        x = x.data
    #        c = c.data
    #        w = w.data
    #        dist = self.dist
    #        #grad_c = -w*torch.exp(-s*dist)*s*2*(c-x) # TODO: Doesn't work
    #        grad_c = None
    #        grad_w = grad_output*v
    #        grad_s = grad_output*(w*v*dist).sum()
    #        return None, grad_c, grad_w, grad_s

    #rbf = RBFFunction(spread=1)
    ##for i in tqdm(range(10000)):
    #for i in range(1000):
    #    inputs = [np.random.randint(2), np.random.randint(2)]
    #    output = inputs[0]^inputs[1]
    #    x = Variable(torch.from_numpy(np.array([inputs],dtype='float32')), requires_grad=False)
    #    y = Variable(torch.from_numpy(np.array([[output]],dtype='float32')), requires_grad=False)

    #    y_pred = rbf(x,c,w,s)
    #    loss = (y_pred-y).pow(2)
    #    print(i, loss.data[0][0])

    #    loss.backward(retain_graph=True)

    #    #w.data -= 0.05*w.grad.data
    #    #w.grad.data.zero_()
    #    s.data -= 0.05*s.grad.data
    #    s.grad.data.zero_()

    #print(rbf(Variable(torch.from_numpy(np.array([[1,1]],dtype='float32')),requires_grad=False),c,w,s))
    #print(rbf(Variable(torch.from_numpy(np.array([[0,1]],dtype='float32')),requires_grad=False),c,w,s))
    #print(rbf(Variable(torch.from_numpy(np.array([[1,0]],dtype='float32')),requires_grad=False),c,w,s))
    #print(rbf(Variable(torch.from_numpy(np.array([[0,0]],dtype='float32')),requires_grad=False),c,w,s))
