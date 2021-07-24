"""
Replication of experiments done by the policy distillation paper by Rusu et al. 2016
https://arxiv.org/pdf/1511.06295.pdf

TODO:
    Experiments
    - Single game distillation
    - Multi-task distillation
    - Online distillation
    Objectives
    - MSE
    - NLL
    - KL

Experiment details in appendix A
- Run the trained agent with epsilon=0.05 chance of random action
- Replay buffer capacity: 540,000
    - Single game: One buffer
    - Multi-task: One buffer per game with capacity 540,000 each
- Perform 10,000 mini-batch after every hour (~54,000 steps)
    - Multi-task: Each mini-batch comes entirely from one game (randomly chosen)
- Optimizer: RMSProp
- Learning rate: between 1e-4 and 1e-3
- Batch size: ?? 
"""

import os
import itertools
from typing import Optional, NamedTuple, List
from pprint import pprint

from tqdm import tqdm
import torch
import torch.nn
import torch.cuda
import torch.optim
import torch.utils.data
import numpy as np
import dill
import gym.spaces
import gym
import gym.envs
from gym.wrappers import FrameStack, AtariPreprocessing
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from simple_slurm import Slurm

import rl.utils
from rl.agent.agent import Agent
from experiment.experiment import make_experiment_runner
from experiment import Experiment
from experiment.logger import Logger
import rl.agent.smdp.dqn as dqn
import rl.agent.smdp.rand as rand
from rl.agent import ReplayBuffer

class QNetworkCNN(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.LeakyReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=512,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,7*7*64)
        x = self.fc(x)
        return x
class QNetworkCNN_1(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=16,kernel_size=4,stride=2),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=16*9*9,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,9*9*16)
        x = self.fc(x)
        return x
class QNetworkCNN_2(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=16,kernel_size=4,stride=2),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=16*9*9,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,9*9*16)
        x = self.fc(x)
        return x
class QNetworkCNN_3(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=16,kernel_size=4,stride=2),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=16*9*9,out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,9*9*16)
        x = self.fc(x)
        return x

def count_parameters(model):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DiscreteToOnehotObs(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        obs_size = env.observation_space.n
        self.observation_space = gym.spaces.Box(
                low=np.array([0]*obs_size),
                high=np.array([1]*obs_size),
        )
        self.action_space = env.action_space
    def reset(self):
        obs = self.env.reset()
        obs = self._to_onehot(obs)
        return obs
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._to_onehot(obs)
        return obs, reward, done, info
    def _to_onehot(self, val : int):
        output = np.zeros([self.env.observation_space.n])
        output[val] = 1
        return output

def make_env(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    return env

def test(env : gym.Env, agent : Agent):
    total_reward = 0
    total_steps = 0

    obs = env.reset()
    agent.observe(obs, testing=True)
    for total_steps in tqdm(itertools.count(), desc='test episode'):
        #env.render()
        obs, reward, done, _ = env.step(agent.act(testing=True))
        total_reward += reward
        agent.observe(obs, reward, done, testing=True)
        if done:
            break
    env.close()

    return {
        'total_steps': total_steps,
        'total_reward': total_reward,
    }

class TrainExperiment(Experiment):
    def setup(self, config, output_directory):
        self.output_directory = output_directory
        self.device = self._init_device()

        self._test_iterations = config.get('test_iterations',5)
        self._test_frequency = config.get('test_frequency',10_000)
        self._deploy_state_checkpoint_frequency = 250_000

        self.env = self._init_envs(config['env_name'])
        self.agent = self._init_agents(
                env=self.env[0],
                device=self.device
        )

        self.done = True
        self._best_score = float('-inf')
        self.logger = Logger(key_name='step')
    def _init_device(self):
        if torch.cuda.is_available():
            print('GPU found')
            return torch.device('cuda')
        else:
            print('No GPU found. Running on CPU.')
            return torch.device('cpu')
    def _init_envs(self,env_name):
        env = [make_env(env_name), make_env(env_name)]
        return env
    def _init_agents(self, env, device):
        q_net = QNetworkCNN(num_actions=env.action_space.n).to(device)
        agent = dqn.DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            q_net=q_net,
            device=device,
        )
        return agent

    def run_step(self, iteration):
        # Save agent specs that can be deployed
        if iteration % self._deploy_state_checkpoint_frequency == 0:
            filename = os.path.join(self.output_directory,'deploy_state-%d.pkl'%iteration)
            self._save_deploy_state(filename)
            tqdm.write('Saved deploy state at %s' % os.path.abspath(filename))
        # Test and plot results
        if iteration % self._test_frequency == 0:
            self._test(iteration)
            self._plot()
            # Save best model
            score = np.mean(self.logger[-1]['total_reward'])
            if score > self._best_score:
                self._best_score = score
                # Deploy State
                filename = os.path.join(self.output_directory,'deploy_state-best.pkl')
                self._save_deploy_state(filename)
                tqdm.write('Saved deploy state at %s with score %f' % (os.path.abspath(filename), score))
                # Q Network
                filename = os.path.join(self.output_directory,'qnet-best.pkl')
                self._save_model(filename)
                tqdm.write('Q Network saved at %s' % os.path.abspath(filename))
        # Train agent
        self._train()
    def _train(self):
        env = self.env[0]
        done = self.done
        if done:
            obs = env.reset()
            self.agent.observe(obs, testing=False)
        obs, reward, done, _ = env.step(self.agent.act(testing=False))
        reward = np.clip(reward,-1,1)
        self.agent.observe(obs, reward, done, testing=False)
        self.done = done
    def _test(self,iteration):
        for _ in range(self._test_iterations):
            results = test(self.env[1], self.agent)
            self.logger.append(step=iteration, total_steps=results['total_steps'], total_reward=results['total_reward'])
    def _plot(self):
        plot_directory = os.path.join(self.output_directory,'plots')
        if not os.path.isdir(plot_directory):
            os.makedirs(plot_directory)

        test_indices = [i for i,x in enumerate(self.logger) if 'total_reward' in x]
        keys = [self.logger[i]['step'] for i in test_indices]

        y = [np.mean(self.logger[i]['total_reward']) for i in test_indices]
        plt.plot(keys,y)
        plt.ylabel('Total Reward')
        plt.xlabel('Steps')
        plt.grid()

        filename = os.path.join(plot_directory,'plot-reward.png')
        plt.savefig(filename)
        print('Saved plot to %s' % os.path.abspath(filename))

        plt.yscale('log')
        filename = os.path.join(plot_directory,'plot-reward-log.png')
        plt.savefig(filename)
        print('Saved plot to %s' % os.path.abspath(filename))

        plt.close()
    def _save_deploy_state(self, filename):
        data = self.agent.state_dict_deploy()
        with open(filename, 'wb') as f:
            dill.dump(data,f)
    def _save_model(self, filename):
        torch.save(self.agent.q_net, filename)

    def state_dict(self):
        return {
            'best_score': self._best_score,
            'logger': self.logger.state_dict()
        }
    def load_state_dict(self, state):
        self.done = True # Environment needs to be reset. There's no general way of saving the environment state.
        self._best_score = state['best_score']
        self.logger.load_state_dict(state['logger'])

class TrainExperimentDebug(TrainExperiment):
    def _init_agents(self, env, device):
        if isinstance(env.observation_space,gym.spaces.Discrete):
            obs_size = 1
        elif isinstance(env.observation_space,gym.spaces.Box):
            obs_size = env.observation_space.shape[0]
        else:
            raise NotImplementedError()
        #q_net = dqn.QNetworkFCNN([obs_size,5,5,env.action_space.n]).to(device)
        q_net = dqn.QNetworkFCNN([obs_size,env.action_space.n]).to(device)
        agent = dqn.DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            q_net=q_net,
            device=device,
            warmup_steps=32,
            batch_size=64,
            target_eps=0,
            eps_annealing_steps=50_000,
            learning_rate=0.1,
            replay_buffer_size=1_000
        )
        return agent
    def _init_envs(self,env_name):
        def make_env(env_name):
            env = gym.make(env_name)
            env = DiscreteToOnehotObs(env)
            return env
        env = [
                make_env(env_name),
                make_env(env_name),
        ]
        return env

class DistillationDataPoint(NamedTuple):
    obs : torch.Tensor
    output : torch.Tensor
def make_dataset(env, agent, num_samples) -> List[DistillationDataPoint]:
    data : List[DistillationDataPoint] = []
    done = True
    for _ in tqdm(range(num_samples),'Generating Dataset'):
        if done:
            done = False
            obs = env.reset()
            agent.observe(obs,testing=True)
        else:
            obs,reward,done,_ = env.step(agent.act(testing=True))
            agent.observe(obs,reward,done,testing=True)
        data.append(
                DistillationDataPoint(
                    obs=torch.tensor(obs),
                    output=agent.q_net(torch.tensor(obs).unsqueeze(0).float()).squeeze().detach()
                )
        )
    return data

def distil_model(student_model : torch.nn.Module,
        data : List[DistillationDataPoint],
        method : str = 'MSE',
        n_steps : int = 10_000*10):
    optimizer = torch.optim.RMSprop(student_model.parameters(), lr=1e-3)
    data_obs = torch.stack([d.obs for d in data]).float()
    data_output = torch.stack([d.output for d in data]).float()
    for _ in tqdm(range(n_steps), desc='Distilling'):
        indices = np.random.choice(500,32)
        obs = data_obs[indices,:,:,:]
        y_target = data_output[indices,:]
        y_pred = student_model(obs)
        
        if method == 'MSE':
            criterion = torch.nn.MSELoss(reduction='mean')
            loss = criterion(y_pred,y_target)
        elif method == 'NLL':
            criterion = torch.nn.NLLLoss(reduction='mean')
            log_softmax = torch.nn.LogSoftmax()
            loss = criterion(log_softmax(y_pred),y_target.max(1)[1])
        elif method == 'KL':
            criterion = torch.nn.KLDivLoss(reduction='mean')
            log_softmax = torch.nn.LogSoftmax()
            loss = criterion(log_softmax(y_pred),log_softmax(y_target))
        else:
            raise Exception('Invalid method: %s' % method)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return student_model

class DistillationExperiment(Experiment):
    def setup(self, config, output_directory):
        self.device = self._init_device()

        self.output_directory = output_directory

        self.agent = dqn.make_agent_from_deploy_state(
                config['teacher_filename'], device=self.device)
        self.replay_buffer = ReplayBuffer(
                max_size=config.get('replay_buffer_size',500_000))
        self.batch_size = config.get('batch_size',32)
        self.train_frequency = config.get('train_frequency',50_000)
        self.train_iterations = config.get('train_iterations',10_000)
        self.test_iterations = config.get('test_iterations',5)

        self.env_name = config['env_name']
        self.env = self._make_env(self.env_name)
        self.done = True

        # Students
        self.student_models = {
            'MSE': {
                'QNetworkCNN': QNetworkCNN(self.env.action_space.n),
                #'QNetworkCNN_1': QNetworkCNN_1(self.env.action_space.n),
                #'QNetworkCNN_2': QNetworkCNN_2(self.env.action_space.n),
                #'QNetworkCNN_3': QNetworkCNN_3(self.env.action_space.n),
            },
            'NLL': {
                'QNetworkCNN': QNetworkCNN(self.env.action_space.n),
                #'QNetworkCNN_1': QNetworkCNN_1(self.env.action_space.n),
                #'QNetworkCNN_2': QNetworkCNN_2(self.env.action_space.n),
                #'QNetworkCNN_3': QNetworkCNN_3(self.env.action_space.n),
            },
            'KL': {
                'QNetworkCNN': QNetworkCNN(self.env.action_space.n),
                #'QNetworkCNN_1': QNetworkCNN_1(self.env.action_space.n),
                #'QNetworkCNN_2': QNetworkCNN_2(self.env.action_space.n),
                #'QNetworkCNN_3': QNetworkCNN_3(self.env.action_space.n),
            },
        }
        for models in self.student_models.values():
            for model in models.values():
                model.to(self.device)
                #model.load_state_dict(self.agent.q_net.state_dict()) # XXX: DEBUG (Copy parameters from teacher model)
        self.losses = {'MSE': [], 'NLL': [], 'KL': []} # XXX: Debug
        # Optimizer
        self.optimizer = {}
        for method,models in self.student_models.items():
            self.optimizer[method] = {}
            for model_name,model in models.items():
                self.optimizer[method][model_name] = torch.optim.RMSprop(
                        model.parameters(),
                        lr=1e-4)
        # Parameter count
        self.num_params = {}
        for models in self.student_models.values():
            for model_name,model in models.items():
                if model_name in self.num_params:
                    continue
                self.num_params[model_name] = count_parameters(model)
        # Baselines
        self.teacher_score = None
        self.random_score = None
        self.teacher_model_name = 'QNetworkCNN'
        # Logs
        self.logger = Logger(key_name='step')
    def _init_device(self):
        if torch.cuda.is_available():
            print('GPU found')
            return torch.device('cuda')
        else:
            print('No GPU found. Running on CPU.')
            return torch.device('cpu')
    def _make_env(self, env_name):
        return make_env(env_name)
    def run_step(self, iteration):
        if iteration == 0:
            print('Testing teacher...')
            self.teacher_score = self._test_teacher()
            print('Teacher score', self.teacher_score)
            print('Testing random agent...')
            self.random_score = self._test_random()
            print('Random agent score', self.random_score)
        #if len(self.replay_buffer) >= self.batch_size:
        if iteration % self.train_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
            self._train()
            result = self._test()
            self.logger.log(step=iteration,result=result)
            self._plot()
            self._plot_debug()
            pprint(result)
        self._generate_datapoint()
    def _generate_datapoint(self):
        env = self.env
        if self.done:
            self.done = False
            obs = env.reset()
            self.agent.observe(obs, testing=False)
        else:
            obs, reward, self.done, _ = env.step(self.agent.act(testing=False))
            self.agent.observe(obs, reward, self.done, testing=False)
        obs = torch.tensor(obs)
        self.replay_buffer.add(
                DistillationDataPoint(
                    obs=obs,
                    output=self.agent.q_net(obs.to(self.device).unsqueeze(0).float()/255).squeeze().detach()
                )
        )
    def _train(self):
        optimizer = self.optimizer
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        debug_loss = {'MSE': [], 'NLL': [], 'KL': []}
        for _,x in tqdm(zip(range(self.train_iterations),dataloader),desc='Training'):
            obs = x.obs/255
            obs = obs.to(self.device)
            for method,models in self.student_models.items():
                for model_name,model in models.items():
                    optimizer = self.optimizer[method][model_name]
                    y_target = x.output
                    y_pred = model(obs)
                    if method == 'MSE':
                        criterion = torch.nn.MSELoss(reduction='mean')
                        loss = criterion(y_pred,y_target)
                    elif method == 'NLL':
                        criterion = torch.nn.NLLLoss(reduction='mean')
                        log_softmax = torch.nn.LogSoftmax()
                        loss = criterion(log_softmax(y_pred),y_target.max(1)[1])
                    elif method == 'KL':
                        criterion = torch.nn.KLDivLoss(reduction='mean',log_target=True)
                        log_softmax = torch.nn.LogSoftmax()
                        loss = criterion(log_softmax(y_pred),log_softmax(y_target))
                    else:
                        raise Exception('Invalid method: %s' % method)
                    debug_loss[method].append(loss.detach().item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        for k,v in debug_loss.items():
            self.losses[k].append(v)
    def _test(self):
        results = {}
        q_net = next(iter(next(iter(self.student_models.values())).values())) # Choose an arbitrary network
        agent = dqn.DQNAgent(
                action_space=self.env.action_space,
                observation_space=self.env.observation_space,
                q_net=q_net,
                device=self.device,
        )
        env = self._make_env(self.env_name)
        for method,models in self.student_models.items():
            results[method] = {}
            for model_name,model in models.items():
                print(method, model_name)
                agent.q_net = model
                results[method][model_name] = [test(env, agent) for _ in range(self.test_iterations)]
        return results
    def _test_teacher(self):
        agent = self.agent
        env = self._make_env(self.env_name)
        return [test(env,agent) for _ in range(self.test_iterations)]
    def _test_random(self):
        env = self._make_env(self.env_name)
        agent = rand.RandomAgent(env.action_space)
        return [test(env,agent) for _ in range(self.test_iterations)]
    def _plot(self):
        plot_filename = os.path.join(self.output_directory,'plot.png')
        bar_plot_filename = os.path.join(self.output_directory,'bar-plot.png')
        line_plot_filename = os.path.join(self.output_directory,'line-plot.png')
        num_params = self.num_params

        if self.teacher_score is None:
            raise Exception('_test_random() must be run before plotting.')
        if self.random_score is None:
            raise Exception('_test_teacher() must be run before plotting.')

        teacher_score_mean = np.mean([x['total_reward'] for x in self.teacher_score])
        random_score_mean = np.mean([x['total_reward'] for x in self.random_score])

        labels = sorted([k for k in num_params.keys()], key=lambda k: num_params[k],reverse=True)
        means = {}
        results = self.logger[-1]['result']
        for method,_ in results.items():
            means[method] = {}
            for l in labels:
                if l in results[method]:
                    score = np.mean([x['total_reward'] for x in results[method][l]])
                    means[method][l] = (score-random_score_mean)/(teacher_score_mean-random_score_mean)
                else:
                    means[method][l] = np.nan

        # Plot bar graph comparing different models and methods (i.e. multiple groups of bars, where each group consists of three bars, one for each method. See https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html and https://www.python-graph-gallery.com/11-grouped-barplot
        bar_width = 0.25
        x = np.arange(len(labels))
        plt.figure()
        for i,method in enumerate(means.keys()):
            plt.bar(x + bar_width*i,
                    [means[method][l] for l in labels],
                    width=bar_width, edgecolor='white', label=method)
        plt.xlabel('group', fontweight='bold')
        plt.xticks([r + bar_width for r in range(len(labels))], labels)
        plt.legend()
        plt.savefig(bar_plot_filename)
        print('Bar plot saved at %s' % os.path.abspath(bar_plot_filename))
        plt.close()

        # Plot a line graph with performance versus relative number of parameters compared to the teacher model (student size / teacher size)
        teacher_model_size = num_params[self.teacher_model_name]
        plt.figure()
        for i,method in enumerate(means.keys()):
            x = [num_params[l]/teacher_model_size for l in labels if l in means[method].keys()]
            y = [means[method][l] for l in labels if l in means[method].keys()]
            plt.plot(x,y,label=method,marker='o')
        plt.xlabel('Model size relative to teacher')
        plt.ylabel('Student score')
        plt.legend()
        plt.grid()
        plt.savefig(line_plot_filename)
        print('Line plot saved at %s' % os.path.abspath(line_plot_filename))
        plt.close()

        # Plot the student performances over time
        data = [x for x in self.logger if 'result' in x]
        if len(data) > 2:
            plt.figure()

            for method,models in self.student_models.items():
                for model_name in models.keys():
                    x = [v['step'] for v in data]
                    y = [np.mean([r['total_reward'] for r in v['result'][method][model_name]]) for v in data]
                    plt.plot(x,y,label='%s (%s)' % (model_name,method))
            plt.xlabel('Training points')
            plt.ylabel('Student score')
            plt.title('Student performance over time')
            plt.legend()
            plt.grid()
            plt.savefig(plot_filename)
            print('Plot saved at %s' % os.path.abspath(plot_filename))
            plt.close()
    def _plot_debug(self):
        # Loss over time
        fig, axs = plt.subplots(len(self.losses), sharex=True)
        for ax,(method,data) in zip(axs,self.losses.items()):
            y = [np.mean(d) for d in data]
            x = range(len(y))
            ax.plot(x,y,label=method)
            ax.grid()
            ax.set_xlabel('Training iterations')
            ax.set_ylabel('%s Loss' % method)
        fig.suptitle('DEBUG: Losses')

        filename = os.path.join(self.output_directory,'debug.png')
        plt.savefig(filename)
        tqdm.write('Debug plot saved to %s' % filename)

        for ax in axs:
            ax.set_yscale('log')
        filename = os.path.join(self.output_directory,'debug-log.png')
        fig.savefig(filename)
        tqdm.write('Debug plot saved to %s' % filename)
        plt.close(fig)

        # Minibatch loss
        fig, axs = plt.subplots(len(self.losses), sharex=True)
        for ax,(method,data) in zip(axs,self.losses.items()):
            y = data[-1]
            x = range(len(y))
            ax.scatter(x,y,label=method)
            ax.grid()
            ax.set_xlabel('Number of mini-batches')
            ax.set_ylabel('%s Mini-batch Loss' % method)
        fig.suptitle('DEBUG: Losses')

        filename = os.path.join(self.output_directory,'debug-minibatch.png')
        plt.savefig(filename)
        tqdm.write('Debug plot saved to %s' % filename)

        for ax in axs:
            ax.set_yscale('log')
        filename = os.path.join(self.output_directory,'debug-minibatch-log.png')
        fig.savefig(filename)
        tqdm.write('Debug plot saved to %s' % filename)
        plt.close(fig)

    def state_dict(self):
        return {
                'replay_buffer': self.replay_buffer.state_dict(),
                'replay_buffer_current_size': len(self.replay_buffer),
                'student_models': {
                    method: {
                        name: model.state_dict()
                        for name,model in models.items()
                    }
                    for method,models in self.student_models.items()
                },
                'optimizer': {
                    method: {
                        name: optimizer.state_dict()
                        for name,optimizer in optimizers.items()
                    }
                    for method,optimizers in self.optimizer.items()
                },
                #'optimizer': self.optimizer.state_dict(),
                'logger': self.logger.state_dict(),
                'teacher_score': self.teacher_score,
                'random_score': self.random_score,
                'DEBUG-losses': self.losses
        }
    def load_state_dict(self, state):
        if 'replay_buffer' in state:
            self.replay_buffer.load_state_dict(state['replay_buffer'])
        else:
            for _ in tqdm(range(state['replay_buffer_current_size']),desc='Regenerating data'):
                # This takes up too much disk space. It's faster to regenerate than to save/load from disk. It all comes from the same distribution anyway.
                self._generate_datapoint()
        for method,models in self.student_models.items():
            for name,model in models.items():
                model.load_state_dict(state['student_models'][method][name])
        for method,optimizers in self.optimizer.items():
            for name,optimizer in optimizers.items():
                optimizer.load_state_dict(state['optimizer'][method][name])
        #self.optimizer.load_state_dict(state['optimizer'])
        self.logger.load_state_dict(state['logger'])
        self.teacher_score = state['teacher_score']
        self.random_score = state['random_score']
        self.losses = state['DEBUG-losses']

class DistillationExperimentDebug(DistillationExperiment):
    def setup(self, config, output_directory):
        super().setup(config, output_directory)

        #self.test_iterations = 100

        # Students
        if isinstance(self.env.observation_space,gym.spaces.Discrete):
            obs_size = 1
        elif isinstance(self.env.observation_space,gym.spaces.Box):
            obs_size = self.env.observation_space.shape[0]
        else:
            raise NotImplementedError()
        self.student_models = {
            'MSE': {
                'QNetworkFCNN': dqn.QNetworkFCNN([obs_size,self.env.action_space.n]),
            },
            'NLL': {
                'QNetworkFCNN': dqn.QNetworkFCNN([obs_size,self.env.action_space.n]),
            },
            'KL': {
                'QNetworkFCNN': dqn.QNetworkFCNN([obs_size,self.env.action_space.n]),
            },
        }
        self.losses = {'MSE': [], 'NLL': [], 'KL': []} # XXX: Debug
        # Optimizer
        self.optimizer = {}
        for method,models in self.student_models.items():
            self.optimizer[method] = {}
            for model_name,model in models.items():
                self.optimizer[method][model_name] = torch.optim.SGD(
                        model.parameters(),
                        lr=1,
                        momentum=0)
                #self.optimizer[method][model_name] = torch.optim.RMSprop(
                #        model.parameters(),
                #        lr=1e-1)
        # Parameter count
        self.num_params = {}
        for models in self.student_models.values():
            for model_name,model in models.items():
                if model_name in self.num_params:
                    continue
                self.num_params[model_name] = count_parameters(model)
        # Baselines
        self.teacher_score = None
        self.random_score = None
        self.teacher_model_name = 'QNetworkFCNN'
        # Logs
        self.logger = Logger(key_name='step')
    def _make_env(self, env_name):
        env = gym.make(env_name)
        env = DiscreteToOnehotObs(env)
        return env
    def _train(self):
        optimizer = self.optimizer
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        debug_loss = {'MSE': [], 'NLL': [], 'KL': []}
        obs = torch.eye(16)
        teacher_output = self.agent.q_net(obs).detach()
        for _ in tqdm(zip(range(self.train_iterations),dataloader),desc='Training'):
            #obs = x.obs.float()
            for method,models in self.student_models.items():
                for model_name,model in models.items():
                    optimizer = self.optimizer[method][model_name]
                    #y_target = x.output
                    y_target = teacher_output
                    y_pred = model(obs)
                    if method == 'MSE':
                        criterion = torch.nn.MSELoss(reduction='mean')
                        loss = criterion(y_pred,y_target)
                    elif method == 'NLL':
                        criterion = torch.nn.NLLLoss(reduction='mean')
                        log_softmax = torch.nn.LogSoftmax()
                        loss = criterion(log_softmax(y_pred),y_target.max(1)[1])
                    elif method == 'KL':
                        criterion = torch.nn.KLDivLoss(reduction='mean',log_target=True)
                        log_softmax = torch.nn.LogSoftmax()
                        loss = criterion(log_softmax(y_pred),log_softmax(y_target))
                    else:
                        raise Exception('Invalid method: %s' % method)
                    debug_loss[method].append(loss.detach().item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        for k,v in debug_loss.items():
            self.losses[k].append(v)
    def _test(self):
        self._compute_policy_diff()
        return super()._test()
    def _compute_policy_diff(self):
        obs = torch.eye(16)
        teacher_vals = self.agent.q_net(obs)
        teacher_actions = teacher_vals.max(1)[1]
        for method,models in self.student_models.items():
            for model_name,model in models.items():
                student_vals = model(obs)
                student_actions = student_vals.max(1)[1]
                diff = (teacher_actions==student_actions).sum()
                print(method, model_name, diff)

def make_app():
    import typer
    app = typer.Typer()

    environment_names = [
            #'BeamRiderNoFrameskip-v0',
            #'BreakoutNoFrameskip-v0',
            #'EnduroNoFrameskip-v0',
            #'FreewayNoFrameskip-v0',
            #'MsPacmanNoFrameskip-v0',
            'PongNoFrameskip-v4',
            #'QbertNoFrameskip-v0',
            #'RiverraidNoFrameskip-v0',
            #'SeaquestNoFrameskip-v0',
            #'SpaceInvadersNoFrameskip-v0',
    ]
    models = {
            'QNetworkCNN': QNetworkCNN,
            'QNetworkCNN_1': QNetworkCNN_1,
            'QNetworkCNN_2': QNetworkCNN_2,
            'QNetworkCNN_3': QNetworkCNN_3,
    }
    PROJECT_ROOT = './rl'
    #EXPERIMENT_GROUP_NAME = 'rusu2016-2'
    EXPERIMENT_GROUP_NAME = 'rusu2016'
    #EXPERIMENT_GROUP_NAME = 'rusu2016-debug'
    RESULTS_ROOT_DIRECTORY = rl.utils.get_results_root_directory()
    EXPERIMENT_GROUP_DIRECTORY = os.path.join(RESULTS_ROOT_DIRECTORY, EXPERIMENT_GROUP_NAME)

    @app.command()
    def train(env_name : str,
            results_directory : Optional[str] = None,
            debug : bool = typer.Option(False, '--debug')):
        # Train
        exp_runner = make_experiment_runner(
                TrainExperimentDebug if debug else TrainExperiment,
                max_iterations=100_000 if debug else 50_000_000,
                verbose=True,
                results_directory=results_directory,
                config={
                    'env_name': env_name,
                    'test_iterations': 10 if debug else 5,
                    'test_frequency': 100 if debug else 250_000,
                }
        )
        exp_runner.run()

    @app.command()
    def test_saved_model(
            env_name : str,
            filename : str,
            output_filename : str):
        n_test_runs = 5 # TODO: Make a parameter
        env = make_env(env_name)
        agent = dqn.make_agent_from_deploy_state(filename)
        result = [test(env,agent)['total_reward'] for _ in range(n_test_runs)]
        os.makedirs(os.path.join(*os.path.split(output_filename)[:-1]),exist_ok=True)
        with open(output_filename,'wb') as f:
            dill.dump(result,f)
        print('Model test result saved in %s' % os.path.abspath(output_filename))

    @app.command()
    def test_random(
            env_name : str,
            output_filename : str):
        n_test_runs = 5 # TODO: Make a parameter
        env = make_env(env_name)
        agent = rand.RandomAgent(action_space=env.action_space)
        result = [test(env,agent)['total_reward'] for _ in range(n_test_runs)]
        os.makedirs(os.path.join(*os.path.split(output_filename)[:-1]),exist_ok=True)
        with open(output_filename,'wb') as f:
            dill.dump(result,f)
        print('Random test result saved in %s' % os.path.abspath(output_filename))

    @app.command()
    def generate_dataset(env_name : str,
            teacher_filename : str,
            output_filename: str,
            debug : bool = typer.Option(False, '--debug')):
        env = make_env(env_name)
        agent = dqn.make_agent_from_deploy_state(teacher_filename)
        agent.eps = [0.05,0.05]
        dataset = make_dataset(
                env,
                agent, 
                500 if debug else 500_000
        )
        os.makedirs(os.path.join(*os.path.split(output_filename)[:-1]),exist_ok=True)
        with open(output_filename, 'wb') as f:
            dill.dump(dataset, f)
        print('Dataset saved in %s' % os.path.abspath(output_filename))

    @app.command()
    def distill(env_name : str,
            dataset_filename : str,
            output_filename: str,
            model_name : str,
            method : str = 'KL',
            n_test_runs : int = 5,
            debug : bool = typer.Option(False, '--debug')):
        # Environment
        env = make_env(env_name)

        # Select student model
        student_model = models[model_name](env.action_space.n)

        # Load dataset
        with open(dataset_filename,'rb') as f:
            dataset : List[DistillationDataPoint] = dill.load(f)
            # XXX: How do I check `dataset`'s type?

        # Run distillation
        distilled_model = distil_model(
                student_model=student_model,
                data=dataset,
                method=method,
                n_steps=100 if debug else 10_000*10)

        # Evaluate distilled model
        distilled_agent = dqn.DQNAgent(
                action_space=env.action_space,
                observation_space=env.observation_space,
                q_net=distilled_model,
                target_eps=0.05
        )
        result = [test(env,distilled_agent)['total_reward'] for _ in range(n_test_runs)]

        # Save results
        os.makedirs(os.path.join(*os.path.split(output_filename)[:-1]),exist_ok=True)
        with open(output_filename,'wb') as f:
            dill.dump(result,f)
        print('Distillation result saved in %s' % os.path.abspath(output_filename))

    @app.command()
    def distill2(env_name : str,
            teacher_filename : str,
            results_directory : Optional[str] = None,
            debug : bool = typer.Option(False, '--debug')):
        if debug:
            if env_name == 'FrozenLake-v0':
                exp_runner = make_experiment_runner(
                        DistillationExperimentDebug,
                        max_iterations=10_000,
                        verbose=True,
                        results_directory=results_directory,
                        checkpoint_frequency=10_000,
                        config={
                            'env_name': env_name,
                            'batch_size': 32,
                            'train_iterations': 100,
                            'train_frequency': 100,
                            'teacher_filename': teacher_filename
                        }
                )
            else:
                exp_runner = make_experiment_runner(
                        DistillationExperiment,
                        max_iterations=100,
                        verbose=True,
                        results_directory=results_directory,
                        checkpoint_frequency=50_000,
                        num_checkpoints=2,
                        config={
                            'env_name': env_name,
                            'batch_size': 8,
                            'train_iterations': 1,
                            'train_frequency': 5,
                            'teacher_filename': teacher_filename
                        }
                )
        else:
            exp_runner = make_experiment_runner(
                    DistillationExperiment,
                    max_iterations=5_000_000,
                    verbose=True,
                    results_directory=results_directory,
                    checkpoint_frequency=50_000,
                    num_checkpoints=2,
                    config={
                        'env_name': env_name,
                        'batch_size': 32,
                        'train_iterations': 10_000,
                        'train_frequency': 50_000,
                        'teacher_filename': teacher_filename
                    }
            )
        exp_runner.run()

    @app.command()
    def run_local(debug : bool = typer.Option(False, '--debug')):
        #results = {}
        for env_name in environment_names:
            train_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'train',env_name)
            #test_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'test')
            distill2_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'distill2',env_name)
            #teacher_test_filename = os.path.join(test_directory,'teacher','%s.pkl' % env_name)
            #random_test_filename = os.path.join(test_directory,'random','%s.pkl' % env_name)
            teacher_model_filename = os.path.join(train_directory,'output/deploy_state-best.pkl')
            #dataset_filename = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'dataset','%s.pkl'%env_name)
            #train(
            #        env_name=env_name,
            #        results_directory=train_directory,
            #        debug=debug
            #)
            #test_saved_model(
            #        env_name=env_name,
            #        filename=teacher_model_filename,
            #        output_filename=teacher_test_filename,
            #)
            #test_random(
            #        env_name=env_name,
            #        output_filename=random_test_filename,
            #)
            distill2(
                    env_name=env_name,
                    results_directory=distill2_directory,
                    teacher_filename=teacher_model_filename,
                    debug=debug,
            )
            #generate_dataset(
            #        env_name=env_name,
            #        teacher_filename=teacher_model_filename,
            #        output_filename=dataset_filename,
            #        debug=debug,
            #)
            #for i in range(1,2):
            #    for model_name in models.keys():
            #        for method in ['MSE', 'NLL', 'KL']:
            #            distill(
            #                    env_name=env_name,
            #                    dataset_filename=dataset_filename,
            #                    output_filename=os.path.join(
            #                        EXPERIMENT_GROUP_DIRECTORY,'distill',env_name,model_name,'%s-%d.pkl'%(method,i)
            #                    ),
            #                    model_name=model_name,
            #                    method=method,
            #                    n_test_runs=5,
            #                    debug=debug
            #            )
            ## Save results
            #results[env_name] = {}
            #with open(teacher_test_filename,'rb') as f:
            #    results[env_name]['teacher_score'] = dill.load(f)
            #with open(random_test_filename,'rb') as f:
            #    results[env_name]['random_score'] = dill.load(f)
            #results[env_name]['student_score'] = {
            #        model_name: {method: [] for method in ['MSE','NLL','KL']}
            #        for model_name in models.keys()
            #}
            #for model_name in models.keys():
            #    directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'distill',env_name,model_name)
            #    for filename in os.listdir(directory):
            #        match = re.match(r'([A-Z]+)-(\d+).pkl', filename)
            #        method = match.group(1)
            #        with open(os.path.join(directory,filename),'rb') as f:
            #            results[env_name]['student_score'][model_name][method].append(
            #                    dill.load(f)
            #            )

    @app.command()
    def run_slurm(debug : bool = typer.Option(False, '--debug')):
        """
        Train
        Generate Dataset
        """
        print('Initializing experiments')

        train_job_ids = {}
        for env_name in environment_names:
            train_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'train',env_name)
            test_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'test')
            teacher_test_filename = os.path.join(test_directory,'teacher','%s.pkl' % env_name)
            random_test_filename = os.path.join(test_directory,'random','%s.pkl' % env_name)
            teacher_model_filename = os.path.join(train_directory,'output/deploy_state-best.pkl')
            dataset_filename = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'dataset','%s.pkl'%env_name)
            ##################################################
            # Train
            slurm = Slurm(
                    cpus_per_task=1,
                    output='/miniscratch/huanghow/slurm/%A_%a.out',
                    time='24:00:00',
            )
            command = './sbatch37.sh {script} train {env_name} --results-directory {results_directory} {debug}'
            command = command.format(
                    script = os.path.join(PROJECT_ROOT, 'rl/experiments/distillation/rusu2016.py'),
                    env_name = env_name,
                    results_directory = train_directory,
                    debug = '--debug' if debug else ''
            )
            print(command)
            train_job_id = slurm.sbatch(command)

            ##################################################
            # Test Teacher
            slurm = Slurm(
                    cpus_per_task=1,
                    output='/miniscratch/huanghow/slurm/%A_%a.out',
                    time='5:00:00',
                    dependency=dict(afterok=train_job_ids[env_name])
            )
            command = './sbatch37.sh {script} test-saved-model {env_name} --filename {teacher_filename} --output-filename {output_filename}'
            command = command.format(
                    script = os.path.join(PROJECT_ROOT, 'rl/experiments/distillation/rusu2016.py'),
                    env_name = env_name,
                    teacher_filename = teacher_model_filename,
                    output_filename = teacher_test_filename,
                    debug = '--debug' if debug else ''
            )
            print(command)
            slurm.sbatch(command)

            ##################################################
            # Test Random Agent
            slurm = Slurm(
                    cpus_per_task=1,
                    output='/miniscratch/huanghow/slurm/%A_%a.out',
                    time='5:00:00',
                    dependency=dict(afterok=train_job_ids[env_name])
            )
            command = './sbatch37.sh {script} test-random {env_name} --output-filename {output_filename}'
            command = command.format(
                    script = os.path.join(PROJECT_ROOT, 'rl/experiments/distillation/rusu2016.py'),
                    env_name = env_name,
                    output_filename = random_test_filename,
                    debug = '--debug' if debug else ''
            )
            print(command)
            slurm.sbatch(command)

            ##################################################
            # Dataset
            slurm = Slurm(
                    cpus_per_task=1,
                    output='/miniscratch/huanghow/slurm/%A_%a.out',
                    time='5:00:00',
                    dependency=dict(afterok=train_job_id)
            )
            command = './sbatch37.sh {script} generate-dataset {env_name} --teacher-filename {teacher_filename} --output-filename {output_filename} {debug}'
            command = command.format(
                    script = os.path.join(PROJECT_ROOT, 'rl/experiments/distillation/rusu2016.py'),
                    env_name = env_name,
                    teacher_filename = teacher_model_filename,
                    output_filename = dataset_filename,
                    debug = '--debug' if debug else ''
            )
            print(command)
            gen_dataset_job_id = slurm.sbatch(command)

            ##################################################
            # Distill
            for i in range(1,2):
                for model_name in models.keys():
                    for method in ['MSE', 'NLL', 'KL']:
                        slurm = Slurm(
                                cpus_per_task=1,
                                output='/miniscratch/huanghow/slurm/%A_%a.out',
                                time='5:00:00',
                                dependency=dict(afterok=gen_dataset_job_id)
                        )
                        command = './sbatch37.sh {script} distill {env_name} --dataset-filename {dataset_filename} --output-filename {output_filename} --model-name {model_name} --method {method} --n-test-runs {n_test_runs} {debug}'
                        command = command.format(
                                script = os.path.join(PROJECT_ROOT, 'rl/experiments/distillation/rusu2016.py'),
                                env_name = env_name,
                                dataset_filename = dataset_filename,
                                output_filename = os.path.join(
                                    EXPERIMENT_GROUP_DIRECTORY,'distill',env_name,model_name,'%s-%d.pkl'%(method,i)
                                ),
                                model_name=model_name,
                                method=method,
                                n_test_runs=5,
                                debug = '--debug' if debug else ''
                        )
                        print(command)
                        slurm.sbatch(command)

        ##################################################
        # Plot?
        # TODO

    @app.command()
    def run_local_debug():
        #results = {}
        env_name = 'FrozenLake-v0'
        train_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'train',env_name)
        test_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'test')
        distill2_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'distill2',env_name)
        teacher_test_filename = os.path.join(test_directory,'teacher','%s.pkl' % env_name)
        random_test_filename = os.path.join(test_directory,'random','%s.pkl' % env_name)
        teacher_model_filename = os.path.join(train_directory,'output/deploy_state-best.pkl')
        train(
                env_name=env_name,
                results_directory=train_directory,
                debug=True
        )
        distill2(
                env_name=env_name,
                results_directory=distill2_directory,
                teacher_filename=teacher_model_filename,
                debug=True,
        )

    commands = {
            'train': train,
            'test_saved_model': test_saved_model,
            'test_random': test_random,
            'generate_dataset': generate_dataset,
            'distill': distill,
            'distill2': distill2,
            'run_local': run_local,
            'run_slurm': run_slurm,
    }

    return app, commands

if __name__ == '__main__':
    app,_ = make_app()
    app()
