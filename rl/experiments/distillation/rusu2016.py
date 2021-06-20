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
from rl.agent.agent import Agent
from experiment.experiment import make_experiment_runner
from typing import Optional
from pprint import pprint

from tqdm import tqdm
import torch
import torch.nn
import torch.cuda
import torch.optim
import numpy as np
import dill
import gym.spaces
import gym
import gym.envs
from gym.wrappers import FrameStack, AtariPreprocessing
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import rl.utils
from experiment import Experiment
from experiment.logger import Logger
import rl.agent.smdp.dqn as dqn

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
        self._test_frequency = config.get('test_frequency',1000)
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
        q_net = dqn.QNetworkCNN(num_actions=env.action_space.n).to(device)
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
                filename = os.path.join(self.output_directory,'deploy_state-best.pkl')
                self._save_deploy_state(filename)
                tqdm.write('Saved deploy state at %s with score %f' % (os.path.abspath(filename), score))
        # Train agent
        self._train()
    def _train(self):
        env = self.env[0]
        done = self.done
        if done:
            obs = env.reset()
            self.agent.observe(obs, testing=False)
        obs, reward, done, _ = env.step(self.agent.act(testing=False))
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

def make_dataset(env, agent, num_samples):
    data = []
    done = True
    for _ in tqdm(range(num_samples),'Generating Dataset'):
        if done:
            done = False
            obs = env.reset()
            agent.observe(obs,testing=True)
        else:
            obs,reward,done,_ = env.step(agent.act(testing=True))
            agent.observe(obs,reward,done,testing=True)
        data.append(obs)
    return data

def distil_model(teacher_model, student_model, data,
        method : str = 'MSE',
        n_steps : int = 10_000):
    optimizer = torch.optim.RMSprop(student_model.parameters(), lr=1e-3)
    data = torch.stack([torch.tensor(d) for d in data]).float()
    for _ in tqdm(range(n_steps), desc='Distilling'):
        indices = np.random.choice(500,32)
        obs = data[indices,:,:,:]
        y_target = teacher_model(obs)
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
        self.output_directory = output_directory
        self.target_model = config['target_model']
        self.agent = None
    def run_step(self, iteration):
        return super().run_step(iteration)

def make_app():
    import typer
    app = typer.Typer()

    environment_names = [
            'BeamRiderNoFrameskip-v0',
            'BreakoutNoFrameskip-v0',
            'EnduroNoFrameskip-v0',
            'FreewayNoFrameskip-v0',
            'MsPacmanNoFrameskip-v0',
            'PongNoFrameskip-v4',
            'QbertNoFrameskip-v0',
            'RiverraidNoFrameskip-v0',
            'SeaquestNoFrameskip-v0',
            'SpaceInvadersNoFrameskip-v0',
    ]
    models = {
            'QNetworkCNN': dqn.QNetworkCNN,
            'QNetworkCNN_1': dqn.QNetworkCNN_1,
            'QNetworkCNN_2': dqn.QNetworkCNN_2,
            'QNetworkCNN_3': dqn.QNetworkCNN_3,
    }

    @app.command()
    def train(env_name : str,
            results_directory : Optional[str] = None,
            debug : bool = typer.Option(False, '--debug')):
        exp_runner = make_experiment_runner(
                TrainExperiment,
                max_iterations=100 if debug else 50_000_000,
                verbose=True,
                results_directory=results_directory,
                config={
                    'env_name': env_name,
                    'test_iterations': 1 if debug else 5,
                    'test_frequency': 250_000,
                }
        )
        exp_runner.run()

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
            teacher_filename : str,
            output_filename: str,
            model_name : str,
            method : str = 'KL',
            n_test_runs : int = 5,
            debug : bool = typer.Option(False, '--debug')):
        # Environment
        env = make_env(env_name)

        # Select student model
        student_model = models[model_name](env.action_space.n)

        # Load teacher model
        agent = dqn.make_agent_from_deploy_state(teacher_filename)

        # Load dataset
        with open(dataset_filename,'rb') as f:
            dataset = dill.load(f)

        # Run distillation
        distilled_model = distil_model(
                teacher_model=agent.q_net,
                student_model=student_model,
                data=dataset,
                method=method,
                n_steps=100 if debug else 10_000)

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
    def run_local(results_directory : Optional[str] = None,
            debug : bool = typer.Option(False, '--debug')):
        if results_directory is None:
            results_directory = os.path.join(
                    rl.utils.get_results_root_directory(),
                    'rusu2016'
            )
        for env_name in environment_names:
            train_directory = os.path.join(results_directory,'train',env_name)
            teacher_model_filename = os.path.join(train_directory,'output/deploy_state-best.pkl')
            dataset_filename = os.path.join(results_directory,'dataset','%s.pkl'%env_name)
            train(
                    env_name=env_name,
                    results_directory=train_directory,
                    debug=debug
            )
            generate_dataset(
                    env_name=env_name,
                    teacher_filename=teacher_model_filename,
                    output_filename=dataset_filename
            )
            for i in range(1,6):
                for model_name in models.keys():
                    for method in ['MSE', 'NLL', 'KL']:
                        distill(
                                env_name=env_name,
                                dataset_filename=dataset_filename,
                                teacher_filename=teacher_model_filename,
                                output_filename=os.path.join(
                                    results_directory,'distill',env_name,'%d.pkl'%i
                                ),
                                model_name=model_name,
                                method=method,
                                n_test_runs=5,
                                debug=debug
                        )

    commands = {
            'train': train,
            'generate_dataset': generate_dataset,
            'distill': distill,
            'run_local': run_local
    }

    return app, commands

def foo():
    n_test_runs = 5 # Number of test episodes per distilled student model
    n_distillations = 5 # Number of distillations per model architecture
    checkpoint_filename = ''
    teacher_model_filename = ''

    environment_names = [
            'BeamRiderNoFrameskip-v0',
            'BreakoutNoFrameskip-v0',
            'EnduroNoFrameskip-v0',
            'FreewayNoFrameskip-v0',
            'MsPacmanNoFrameskip-v0',
            'PongNoFrameskip-v4',
            'QbertNoFrameskip-v0',
            'RiverraidNoFrameskip-v0',
            'SeaquestNoFrameskip-v0',
            'SpaceInvadersNoFrameskip-v0',
    ]
    models = [
            dqn.QNetworkCNN,
            dqn.QNetworkCNN_1,
            dqn.QNetworkCNN_2,
            dqn.QNetworkCNN_3,
    ]
    results = {}
    for env_name in environment_names:
        results[env_name] = {}
        results[env_name]['teacher'] = []
        for model in models:
            results[env_name][model.__name__] = []
    
    for env_name in environment_names:
        exp_runner = make_experiment_runner(
                TrainExperiment,
                #max_iterations=50_000_000,
                max_iterations=100, # DEBUG
                verbose=True,
                config={
                    'env_name': env_name,
                    'test_iterations': 1,
                    'test_frequency': 250_000,
                }
        )
        exp_runner.run()

        # Get output directory
        train_directory = exp_runner.exp.output_directory

        # Make dataset from the saved model
        env = make_env(env_name)
        agent = dqn.make_agent_from_deploy_state(
                os.path.join(train_directory,'deploy_state-best.pkl'))
        agent.eps = [0.05,0.05]
        #dataset = make_dataset(env, agent, 500_000)
        dataset = make_dataset(env, agent, 500) # DEBUG

        # Evaluate teacher model
        results[env_name]['teacher'] = [test(env,agent)['total_reward'] for _ in range(n_test_runs)]

        # Evaluate distilled models
        for model in models:
            for method in ['MSE', 'NLL', 'KL']:
                for _ in range(n_distillations):
                    # Run distillation
                    distilled_model = distil_model(
                            teacher_model=agent.q_net,
                            student_model=model(env.action_space.n),
                            data=dataset,
                            method=method)

                    # Evaluate distilled model
                    distilled_agent = dqn.make_agent_from_deploy_state(
                            os.path.join(train_directory,'deploy_state-best.pkl'))
                    distilled_agent.q_net = distilled_model
                    results[env_name][model.__name__].append(
                            [test(env,distilled_agent)['total_reward'] for _ in range(n_test_runs)]
                    )

                    pprint(results)

if __name__ == '__main__':
    app,_ = make_app()
    app()
