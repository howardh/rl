import os
import gym
import numpy as np
import itertools
from tqdm import tqdm
import torch

import utils
from agent.dqn_agent import DQNAgent
from agent.mf_dqn_agent import MultiFidelityDQNAgent2, MultiFidelityDiscreteAgent
from agent.policy import get_greedy_epsilon_policy, get_softmax_policy

import warnings
warnings.warn = lambda *x,**y: None

class RewardNetwork(torch.nn.Module):
    def __init__(self,in_size):
        super().__init__()
        self.seq = torch.nn.Sequential(
                torch.nn.Linear(in_size,int(in_size/2)),
                torch.nn.ReLU(),
                torch.nn.Linear(int(in_size/2),1),
                torch.nn.Tanh()
        )
    def forward(self,x):
        return self.seq(x)

class QNetwork(torch.nn.Module):
    def __init__(self,in_size,out_size):
        super().__init__()
        self.seq = torch.nn.Sequential(
                torch.nn.Linear(in_size,int(in_size/2)),
                torch.nn.ReLU(),
                torch.nn.Linear(int(in_size/2),out_size)
                #torch.nn.Tanh()
        )
    def forward(self,x):
        return self.seq(x)

class HighFidelityEnv(gym.Env):
    def __init__(self, num_actions=5, default_val=0, time_limit=25):
        self.num_actions = num_actions
        self.fingerprint_len = num_actions
        self.default_val = default_val
        self.time_limit = time_limit

        self.action_space = gym.spaces.Discrete(num_actions+1) # +1 ending the episode and receiving a reward
        # Observation = Morgan Fingerprint + a value if an oracle was queries
        self.observation_space = gym.spaces.Box(
                low=np.array([0]*self.fingerprint_len),
                high=np.array([time_limit]*self.fingerprint_len)
        )
        # Create reward function
        self.reward = RewardNetwork(self.fingerprint_len) # True reward function

    def step(self, action):
        self.step_count += 1
        o = torch.tensor(self.state).float()
        if self.step_count >= self.time_limit or action == self.num_actions:
            return self.state, self.reward(o).item(), True, {'runtime': 10}
        else:
            self.state[action] += 1
            return self.state.copy(), self.default_val, False, {'runtime': 0}

    def reset(self):
        self.state = np.zeros([self.fingerprint_len])
        self.step_count = 0
        return self.state.copy()

class LowFidelityEnv(HighFidelityEnv):
    def __init__(self, num_actions=5, default_val=0, time_limit=25):
        super().__init__(num_actions, default_val, time_limit)

        self.action_space = gym.spaces.Discrete(num_actions+2) # +2 for low-fidelity query, and high-fidelity final reward
        # Observation = Morgan Fingerprint + a value if an oracle was queries
        self.observation_space = gym.spaces.Box(
                low=np.array([0]*self.fingerprint_len+[-1]),
                high=np.array([time_limit]*self.fingerprint_len+[1])
        )
        # Create reward function
        self.reward_lf = RewardNetwork(self.fingerprint_len) # Low fidelity reward

    def train_lf_rewards(self,n=10):
        opt = torch.optim.Adam(self.reward_lf.parameters())
        criterion = torch.nn.MSELoss()
        for _ in tqdm(range(n),desc='Training LF Rewards'):
            o = torch.tensor(self.observation_space.sample()[:-1])
            r_lf = self.reward_lf(o)
            r_hf = self.reward(o)
            opt.zero_grad()
            loss = criterion(r_lf,r_hf)
            loss.backward()
            opt.step()
    def evaluate_lf_rewards(self,n=10):
        total = 0
        criterion = torch.nn.MSELoss()
        for _ in tqdm(range(n),desc='Evaluate LF Rewards'):
            o = torch.tensor(self.observation_space.sample()[:-1])
            r_lf = self.reward_lf(o)
            r_hf = self.reward(o)
            loss = criterion(r_lf,r_hf)
            total += loss
        return total/n

    def step(self, action):
        self.step_count += 1
        o = torch.tensor(self.state[:-1]).float()
        self.state[-1] = self.default_val
        if self.step_count >= self.time_limit or action == self.num_actions+1:
            return self.state, self.reward(o).item(), True, {'runtime': 10}
        if action == self.num_actions:
            self.state[-1] = self.reward_lf(o).item()
            return self.state, self.default_val, False, {'runtime': 1}
        else:
            self.state[action] += 1
            return self.state, self.default_val, False, {'runtime': 0}

    def reset(self):
        self.state = np.zeros([self.fingerprint_len+1])
        self.step_count = 0
        return self.state.copy()

class MultiFidelityEnv(gym.Env):
    def __init__(self, num_actions=5, default_val=0, time_limit=25):
        self.num_actions = num_actions
        self.fingerprint_len = num_actions
        self.default_val = default_val
        self.time_limit = time_limit

        self.action_space = gym.spaces.Discrete(num_actions) # +1 ending the episode and receiving a reward
        # Observation = Morgan Fingerprint + a value if an oracle was queries
        self.observation_space = gym.spaces.Box(
                low=np.array([0]*self.fingerprint_len),
                high=np.array([time_limit]*self.fingerprint_len)
        )
        # Create reward function
        self.reward = RewardNetwork(self.fingerprint_len) # True reward function

    def step(self, action):
        self.state[action] += 1
        terminal = self.state.sum() >= self.time_limit
        return self.state.copy(), self.default_val, terminal, {}

    def reset(self):
        self.state = np.zeros([self.fingerprint_len])
        self.step_count = 0
        return self.state.copy()

    def create_reward_estimates(self, iterations=100):
        net = RewardNetwork(self.fingerprint_len)
        self.train_rewards(net,iterations=iterations)
        return net

    def train_rewards(self,net,iterations=10):
        opt = torch.optim.Adam(net.parameters())
        criterion = torch.nn.MSELoss()
        for _ in tqdm(range(iterations),desc='Training LF Rewards'):
            o = torch.tensor(self.observation_space.sample())
            r_lf = net(o)
            r_hf = self.reward(o)
            opt.zero_grad()
            loss = criterion(r_lf,r_hf)
            loss.backward()
            opt.step()
    def evaluate_rewards(self,net,n=10):
        total = 0
        criterion = torch.nn.MSELoss()
        for _ in tqdm(range(n),desc='Evaluate LF Rewards'):
            o = torch.tensor(self.observation_space.sample())
            r_lf = net(o)
            r_hf = self.reward(o)
            loss = criterion(r_lf,r_hf)
            total += loss
        return total/n

class AugmentedMultiFidelityEnv(MultiFidelityEnv):
    def __init__(self, num_actions=5, default_val=0, time_limit=25, lf_iters=[100], costs=[1,10]):
        super().__init__(num_actions, default_val, time_limit)

    def step(self, action):
        self.state[action] += 1
        terminal = self.state.sum() >= self.time_limit
        return self.state.copy(), self.default_val, terminal, {}

    def reset(self):
        self.state = np.zeros([self.fingerprint_len])
        self.step_count = 0
        return self.state.copy()

    def create_reward_estimates(self, iterations=100):
        net = RewardNetwork(self.fingerprint_len)
        self.train_rewards(net,iterations=iterations)
        return net

    def train_rewards(self,net,iterations=10):
        opt = torch.optim.Adam(net.parameters())
        criterion = torch.nn.MSELoss()
        for _ in tqdm(range(iterations),desc='Training LF Rewards'):
            o = torch.tensor(self.observation_space.sample())
            r_lf = net(o)
            r_hf = self.reward(o)
            opt.zero_grad()
            loss = criterion(r_lf,r_hf)
            loss.backward()
            opt.step()
    def evaluate_rewards(self,net,n=10):
        total = 0
        criterion = torch.nn.MSELoss()
        for _ in tqdm(range(n),desc='Evaluate LF Rewards'):
            o = torch.tensor(self.observation_space.sample())
            r_lf = net(o)
            r_hf = self.reward(o)
            loss = criterion(r_lf,r_hf)
            total += loss
        return total/n

def run_trial_mf_discrete(discount=1, eps_b=0.5, eps_t=0, evaluation_method='val', evaluation_criterion='kandasamy', directory=None, max_depth=5, max_steps=500, epoch=10, test_iters=1, verbose=False, oracle_iters=[100,None], oracle_costs=[1,10]):
    args = locals()
    env = MultiFidelityEnv(num_actions=5, time_limit=max_depth)
    test_env = MultiFidelityEnv(num_actions=5, time_limit=max_depth)
    test_env.reward = env.reward
    oracles = []
    for iters in oracle_iters:
        if iters is None:
            oracles.append(env.reward)
        else:
            oracles.append(env.create_reward_estimates(iters))
    true_reward = env.reward

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    def transition_function(s,a):
        s = s.copy()
        s[a] += 1
        return s
    agent = MultiFidelityDiscreteAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t),
            oracles=[
                    lambda x: oracle(torch.tensor(x).float()).item() for oracle in oracles
            ],
            true_reward = lambda x: true_reward(torch.tensor(x).float()).item(),
            oracle_costs=oracle_costs,
            transition_function=transition_function,
            max_depth=max_depth,
            evaluation_method=evaluation_method,
            evaluation_criterion=evaluation_criterion
    )

    rewards = []
    state_action_values = []
    done = True
    step_range = range(0,max_steps+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        for steps in step_range:
            agent.init_oracle_data() # Does nothing if already initialized

            # Run tests
            if steps % epoch == 0:
                r,sa_vals = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f \t len(WS): %d' % (steps, np.mean(r), len(agent.warmup_states)))

            # Skip step if we're busy evaluating an observation
            if agent.is_evaluating_obs(steps):
                continue
            agent.evaluate_obs(steps) # Call oracle on current state if needed

            # Run step
            if done:
                obs = env.reset()
                done = False
            else:
                action = agent.act()
                obs, reward, done, info = env.step(action)
            agent.observe_change(obs)

            # Check if we need to evaluate the new state
            agent.check_needs_evaluation(steps)

            # Update weights
            agent.train()
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

def run_trial_mf_approx(discount=1, eps_b=0.5, eps_t=0, temp_b=None, evaluation_method='val', evaluation_criterion='kandasamy', directory=None, max_depth=5, max_steps=500, epoch=10, test_iters=1, verbose=False, oracle_iters=[100,None], oracle_costs=[1,10], min_replay_buffer_size=1000, learning_rate=1e-3, batch_size=10, polyak_rate=1, warmup_steps=100, training_data='replaybuffer', v_net_arch=[5,2,1]):
    args = locals()
    env = MultiFidelityEnv(num_actions=5, time_limit=max_depth)
    test_env = MultiFidelityEnv(num_actions=5, time_limit=max_depth)
    test_env.reward = env.reward
    oracles = []
    for iters in oracle_iters:
        if iters is None:
            oracles.append(env.reward)
        else:
            oracles.append(env.create_reward_estimates(iters))
    true_reward = env.reward

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    def transition_function(s,a):
        if type(s) is np.ndarray:
            s = torch.tensor(s)
        d = torch.zeros_like(s)
        d[a] = 1
        return s+d
    agent = MultiFidelityDQNAgent2(
            action_space=env.action_space,
            observation_space=env.observation_space,
            behaviour_policy=get_greedy_epsilon_policy(eps_b) if temp_b is None else get_softmax_policy(temp_b),
            target_policy=get_greedy_epsilon_policy(eps_t),
            oracles=[
                    lambda x: oracle(x).item() for oracle in oracles
            ],
            true_reward = lambda x: true_reward(torch.tensor(x).float()).item(),
            oracle_costs=oracle_costs,
            transition_function=transition_function,
            max_depth=max_depth,
            evaluation_method=evaluation_method,
            evaluation_criterion=evaluation_criterion,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            v_net_arch=v_net_arch
    )

    rewards = []
    state_action_values = []
    done = True
    step_range = range(0,max_steps+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        for steps in step_range:
            agent.init_oracle_data() # Does nothing if already initialized

            # Run tests
            if steps % epoch == 0:
                r,sa_vals = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))

            # Skip step if we're busy evaluating an observation
            if agent.is_evaluating_obs(steps):
                continue
            agent.evaluate_obs(steps) # Call oracle on current state if needed

            # Run step
            if done:
                obs = env.reset()
                done = False
            else:
                action = agent.act()
                obs, reward, done, info = env.step(action)
            agent.observe_change(obs)

            # Check if we need to evaluate the new state
            agent.check_needs_evaluation(steps)

            # Update weights
            if training_data == 'replaybuffer':
                agent.train(batch_size=batch_size)
            elif training_data == 'discrete':
                agent.train_discrete_targets()
            elif training_data == 'all':
                agent.train_all_states()
    except ValueError as e:
        if verbose:
            tqdm.write(str(e))
            tqdm.write("Diverged")
            raise e

    print('Points evaluated',[len(od) for od in agent.oracle_data])
    utils.save_results(
            args,
            {'rewards': rewards, 'state_action_values': state_action_values, 'evaluated_points': agent.oracle_data},
            directory=directory)
    return (args, rewards, state_action_values)

def plot(results_directory, plot_directory, exp_names):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    for exp_name in exp_names:
        print('searching',os.path.join(results_directory,exp_name))
        results = utils.get_all_results(os.path.join(results_directory,exp_name), ignore_errors=True)
        data = [np.array(r[1]['rewards']).flatten() for r in results]
        if len(data) == 0:
            continue
        if len(data) > 1:
            max_len = max(*[len(d) for d in data])
            data = [d for d in data if len(d) == max_len]
        data = np.array(data)
        y = data.mean(axis=0)
        x = range(0,len(y)*10,10)
        plt.plot(x,y,label='%s (%d)'%(exp_name,data.shape[0]))
    plt.grid(which='both')
    plt.xlabel('Resources Spent')
    plt.ylabel('Instantaneous Regret')
    plt.legend(loc='best')
    #plt.show()
    plot_path = os.path.join(plot_directory,'plot.png')
    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)
    plt.savefig(plot_path)
    print('Saved fig to',plot_path)

def run():
    utils.set_results_directory(
            os.path.join(utils.get_results_root_directory(),'multifid'))
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)
    print(directory)

    experiments = {
            'baseline-hf-k': {
                'oracle_iters': [None],
                'oracle_costs': [10],
                'evaluation_method': 'val',
                'evaluation_criterion': 'kandasamy'
            },
            'baseline-hf-ucb-k': {
                'oracle_iters': [None],
                'oracle_costs': [10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'kandasamy'
            },
            'baseline-hf-a': {
                'oracle_iters': [None],
                'oracle_costs': [10],
                'evaluation_method': 'val',
                'evaluation_criterion': 'always'
            },
            'baseline-hf-ucb-a': {
                'oracle_iters': [None],
                'oracle_costs': [10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'always'
            },
            'baseline-lf-100-k': {
                'oracle_iters': [100],
                'oracle_costs': [1],
                'evaluation_method': 'val',
                'evaluation_criterion': 'kandasamy'
            },
            'baseline-lf-100-ucb-k': {
                'oracle_iters': [100],
                'oracle_costs': [1],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'kandasamy'
            },
            'mf-100-k': {
                'oracle_iters': [100,None],
                'oracle_costs': [1,10],
                'evaluation_method': 'val',
                'evaluation_criterion': 'kandasamy'
            },
            'mf-100-a': {
                'oracle_iters': [100,None],
                'oracle_costs': [1,10],
                'evaluation_method': 'val',
                'evaluation_criterion': 'always'
            },
            'mf-100-ucb-k': {
                'oracle_iters': [100,None],
                'oracle_costs': [1,10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'kandasamy'
            },
            'mf-100-ucb-a': {
                'oracle_iters': [100,None],
                'oracle_costs': [1,10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'always'
            },
            'approx-mf-100-ucb-k': {
                'model': 'approx',
                'batch_size': 100,
                'warmup_steps': 100,
                'oracle_iters': [100,None],
                'oracle_costs': [1,10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'kandasamy'
            },
            'approx-baseline-hf-ucb-k': {
                'model': 'approx',
                'batch_size': 100,
                'warmup_steps': 100,
                'oracle_iters': [None],
                'oracle_costs': [10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'kandasamy'
            },
            'approx-mf-100-ucb-a': {
                'model': 'approx',
                'batch_size': 100,
                'warmup_steps': 100,
                'oracle_iters': [100,None],
                'oracle_costs': [1,10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'always'
            },
            'approx-baseline-hf-ucb-a': {
                'model': 'approx',
                'batch_size': 100,
                'warmup_steps': 100,
                'oracle_iters': [None],
                'oracle_costs': [10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'always'
            },
            'approx-baseline-hf-ucb-a-sm': {
                'model': 'approx',
                'batch_size': 100,
                'warmup_steps': 100,
                'oracle_iters': [None],
                'oracle_costs': [10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'always',
                'temp_b': 1
            },
            'debug-approx-mf-100-ucb-a-same-oracles': {
                'model': 'approx',
                'batch_size': 100,
                'warmup_steps': 100,
                'oracle_iters': [None,None],
                'oracle_costs': [1,10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'always'
            },
            'debug-approx-cheap-hf': {
                'model': 'approx',
                'batch_size': 100,
                'warmup_steps': 100,
                'oracle_iters': [None],
                'oracle_costs': [1],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'always'
            },
            'approx-baseline-hf-ucb-k-001': { # Using training on all states instead of only replay buffer
                'model': 'approx',
                'batch_size': 100,
                'warmup_steps': 100,
                'oracle_iters': [None],
                'oracle_costs': [10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'kandasamy',
                'training_data': 'all'
            },
            'approx-mf-100-ucb-a-001': {
                'model': 'approx',
                'batch_size': 100,
                'warmup_steps': 100,
                'oracle_iters': [100,None],
                'oracle_costs': [1,10],
                'evaluation_method': 'ucb',
                'evaluation_criterion': 'always',
                'training_data': 'all'
            },
    }

    # Train on the values learned in the discrete algorithm
    experiments['approx-baseline-hf-ucb-k-002'] = experiments['approx-baseline-hf-ucb-k-001'].copy()
    experiments['approx-baseline-hf-ucb-k-002']['training_data'] = 'discrete'
    experiments['approx-mf-100-ucb-a-002'] = experiments['approx-mf-100-ucb-a-001'].copy()
    experiments['approx-mf-100-ucb-a-002']['training_data'] = 'discrete'

    # Might not be working because network capacity is too low
    # Increase network capacity
    experiments['approx-baseline-hf-ucb-k-003'] = experiments['approx-baseline-hf-ucb-k-002'].copy()
    experiments['approx-baseline-hf-ucb-k-003']['v_net_arch'] = [5,5,1]
    experiments['approx-mf-100-ucb-a-003'] = experiments['approx-mf-100-ucb-a-002'].copy()
    experiments['approx-mf-100-ucb-a-003']['v_net_arch'] = [5,5,1]

    # The HF alg is still doing better than MF, but it looks like MF catches up after ~450 steps.
    # Let's look at what happens if we let it run for 1k steps.
    experiments['approx-baseline-hf-ucb-k-004'] = experiments['approx-baseline-hf-ucb-k-003'].copy()
    experiments['approx-baseline-hf-ucb-k-004']['max_steps'] = 1000
    experiments['approx-mf-100-ucb-a-004'] = experiments['approx-mf-100-ucb-a-003'].copy()
    experiments['approx-mf-100-ucb-a-004']['max_steps'] = 1000

    import sys
    print(sys.argv)
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'plot':
            if len(sys.argv) == 2:
                plot(directory,plot_directory,experiments.keys())
            else:
                # Every argument following 'plot' is the name of an experiment to plot
                exp_names = sys.argv[2:]
                plot(directory,plot_directory,exp_names)
            #plot(directory,plot_directory,['baseline-lf-100-ucb-k', 'baseline-hf-ucb-k','mf-100-ucb-k'])
            #plot(directory,plot_directory,['approx-mf-100-ucb-k','approx-baseline-hf-ucb-k','approx-mf-100-ucb-a','approx-baseline-hf-ucb-a'])
        else:
            exp_name = sys.argv[1]
            model = experiments[exp_name].pop('model','discrete')
            while True:
                if model == 'discrete':
                    run_trial_mf_discrete(
                            directory=os.path.join(directory,exp_name),
                            verbose=True,
                            **experiments[exp_name])
                elif model == 'approx':
                    run_trial_mf_approx(
                            directory=os.path.join(directory,exp_name),
                            verbose=True,
                            **experiments[exp_name])
    else:
        exp_name = 'baseline-hf'
        exp_name = 'baseline-lf-100'
        exp_name = 'mf-100-ucb-k'
        exp_name = 'approx-mf-100-ucb-k'
        run_trial_mf_approx(
                directory=os.path.join(directory,exp_name),
                verbose=True,
                batch_size=100,
                **experiments[exp_name])

if __name__=='__main__':
    pass

"""
Environment:
    Only transitions?
Training loop:
    Sample 1 transition according to exploration policy
    Save transition
    Decide if we need to evaluate the current state
        If so, decide level of fidelity and evaluate
        Update estimate for that fidelity
    Train agent
        Use transitions from replay buffer
        Values are the minimum estimate from any fidelity
Agent needs access to estimates of different fidelities
"""
