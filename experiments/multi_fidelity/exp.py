import os
import gym
import numpy as np
import itertools
from tqdm import tqdm
import torch

import utils
from agent.dqn_agent import DQNAgent
from agent.mf_dqn_agent import MultiFidelityDQNAgent2, MultiFidelityDiscreteAgent
from agent.policy import get_greedy_epsilon_policy

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

def run_trial_lf(discount=1, learning_rate=1e-3, eps_b=0.1, eps_t=0, directory=None, batch_size=32, min_replay_buffer_size=1000, max_steps=5000, epoch=50,test_iters=1,verbose=False):
    args = locals()
    env = LowFidelityEnv()
    env.train_lf_rewards(100)
    test_env = LowFidelityEnv()
    test_env.reward = env.reward
    test_env.reward_lf = env.reward_lf
    #env = HighFidelityEnv()
    #test_env = HighFidelityEnv()
    #test_env.reward = env.reward

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    agent = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=learning_rate,
            discount_factor=discount,
            device=device,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t),
            q_net=QNetwork(env.observation_space.shape[0],env.action_space.n)
    )

    rewards = []
    state_action_values = []
    done = True
    step_range = range(0,max_steps+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        skip_steps = 0
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                r,sa_vals = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                state_action_values.append(sa_vals)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))

            # Skip steps in accordance with time cost of different actions
            if skip_steps > 0:
                skip_steps -= 1
                continue

            # Linearly Anneal epsilon
            agent.behaviour_policy = get_greedy_epsilon_policy((1-min(steps/min(1000000,max_steps),1))*(1-eps_b)+eps_b)

            # Run step
            if done:
                obs = env.reset()
                agent.observe_change(obs)
            action = agent.act()
            obs, reward, done, info = env.step(action)
            skip_steps += info['runtime']
            agent.observe_change(obs, reward, done)

            # Update weights
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
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

def run_trial_mf(discount=1, learning_rate=1e-3, eps_b=0.5, eps_t=0, directory=None, batch_size=32, min_replay_buffer_size=1000, max_steps=2000, epoch=50,test_iters=1,verbose=False):
    args = locals()
    env = MultiFidelityEnv()
    test_env = MultiFidelityEnv()
    test_env.reward = env.reward
    oracles = [
            env.create_reward_estimates(100),
            env.reward
    ]
    oracle_costs = [1,10]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    def transition_function(s,a):
        s = s.copy()
        s[a] += 1
        return s
    agent = MultiFidelityDQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            learning_rate=learning_rate,
            discount_factor=discount,
            device=device,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
            target_policy=get_greedy_epsilon_policy(eps_t),
            q_net=QNetwork(env.observation_space.shape[0],1),
            oracles=[
                    lambda x: oracle(torch.tensor(x).float()).item() for oracle in oracles
            ],
            oracle_costs=oracle_costs,
            transition_function=transition_function
    )

    rewards = []
    state_action_values = []
    done = True
    step_range = range(0,max_steps+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        skip_steps = 0
        for steps in step_range:
            agent.init_oracle_data() # Does nothing if already initialized

            # Run tests
            if steps % epoch == 0:
                r,sa_vals = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                state_action_values.append(sa_vals)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))

            # Skip steps in accordance with time cost of different actions
            if skip_steps > 0:
                skip_steps -= 1
                continue

            # Linearly Anneal epsilon
            agent.behaviour_policy = get_greedy_epsilon_policy((1-min(steps/min(1000000,max_steps),1))*(1-eps_b)+eps_b)

            # Run step
            if done:
                obs = env.reset()
                agent.observe_change(obs)
            action = agent.act()
            obs, reward, done, info = env.step(action)
            #skip_steps += info['runtime']
            agent.observe_change(obs)
            skip_steps += agent.evaluate_obs() # Agent checks if it wants to evaluate, and returns runtime

            # Update weights
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
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
        skip_steps = 0
        for steps in step_range:
            agent.init_oracle_data() # Does nothing if already initialized

            # Run tests
            if steps % epoch == 0:
                if skip_steps == 0: # Results from an oracle call should not affect anything until after it's done
                    r,sa_vals = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))

            # Skip steps in accordance with time cost of different actions
            if skip_steps > 0:
                skip_steps -= 1
                continue

            # Linearly Anneal epsilon
            #agent.behaviour_policy = get_greedy_epsilon_policy((1-min(steps/min(1000000,max_steps),1))*(1-eps_b)+eps_b)

            # Run step
            if done:
                obs = env.reset()
                agent.observe_change(obs)
            action = agent.act()
            obs, reward, done, info = env.step(action)
            #skip_steps += info['runtime']
            agent.observe_change(obs)
            skip_steps += agent.evaluate_obs() # Agent checks if it wants to evaluate, and returns runtime

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

#def run_trial_mf(discount=1, learning_rate=1e-3, eps_b=0.5, eps_t=0, directory=None, batch_size=32, min_replay_buffer_size=1000, max_steps=2000, epoch=50,test_iters=1,verbose=False):
def run_trial_mf_approx(discount=1, eps_b=0.5, eps_t=0, evaluation_method='val', evaluation_criterion='kandasamy', directory=None, max_depth=5, max_steps=500, epoch=10, test_iters=1, verbose=False, oracle_iters=[100,None], oracle_costs=[1,10], min_replay_buffer_size=1000, learning_rate=1e-3, batch_size=10, polyak_rate=1, warmup_steps=100):
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
        d = torch.zeros_like(s)
        d[a] = 1
        return s+d
    agent = MultiFidelityDQNAgent2(
            action_space=env.action_space,
            observation_space=env.observation_space,
            behaviour_policy=get_greedy_epsilon_policy(eps_b),
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
            warmup_steps=warmup_steps
    )

    rewards = []
    state_action_values = []
    done = True
    step_range = range(0,max_steps+1)
    if verbose:
        step_range = tqdm(step_range)
    try:
        skip_steps = 0
        for steps in step_range:
            # Run tests
            if steps % epoch == 0:
                if skip_steps == 0: # Results from an oracle call should not affect anything until after it's done
                    r,sa_vals = agent.test(test_env, test_iters, render=False, processors=1)
                rewards.append(r)
                if verbose:
                    tqdm.write('steps %d \t Reward: %f' % (steps, np.mean(r)))

            # Skip steps in accordance with time cost of different actions
            if skip_steps > 0:
                skip_steps -= 1
                continue

            # Linearly Anneal epsilon
            #agent.behaviour_policy = get_greedy_epsilon_policy((1-min(steps/min(1000000,max_steps),1))*(1-eps_b)+eps_b)

            # Run step
            if done:
                obs = env.reset()
                agent.observe_change(obs)
            action = agent.act()
            obs, reward, done, info = env.step(action)
            #skip_steps += info['runtime']
            agent.observe_change(obs)

            agent.init_oracle_data() # Does nothing if already initialized
            skip_steps += agent.evaluate_obs() # Agent checks if it wants to evaluate, and returns runtime

            # Update weights
            agent.train(batch_size=batch_size)
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

def plot(results_directory, plot_directory, exp_names):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    for exp_name in exp_names:
        print('searching',os.path.join(results_directory,exp_name))
        results = utils.get_all_results(os.path.join(results_directory,exp_name))
        data = [np.array(r[1]['rewards']).flatten() for r in results]
        if len(data) == 0:
            continue
        if len(data) > 1:
            max_len = max(*[len(d) for d in data])
            data = [d for d in data if len(d) == max_len]
        data = np.array(data)
        x = range(0,501,10)
        y = data.mean(axis=0)
        plt.plot(x,y,label='%s (%d)'%(exp_name,data.shape[0]))
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
    }

    import sys
    print(sys.argv)
    if len(sys.argv) == 2:
        if sys.argv[1] == 'plot':
            #plot(directory,plot_directory,experiments.keys())
            #plot(directory,plot_directory,['baseline-lf-100-ucb-k', 'baseline-hf-ucb-k','mf-100-ucb-k'])
            plot(directory,plot_directory,['approx-mf-100-ucb-k','approx-baseline-hf-ucb-k'])
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
