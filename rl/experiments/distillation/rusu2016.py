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

import re
import os
import itertools
from rl.agent.agent import Agent
from experiment.experiment import make_experiment_runner
from typing import Optional, NamedTuple, List

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
from simple_slurm import Slurm

import rl.utils
from experiment import Experiment
from experiment.logger import Logger
import rl.agent.smdp.dqn as dqn
import rl.agent.smdp.rand as rand

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
        n_steps : int = 10_000):
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
    PROJECT_ROOT = './rl'
    EXPERIMENT_GROUP_NAME = 'rusu2016'
    RESULTS_ROOT_DIRECTORY = rl.utils.get_results_root_directory()
    EXPERIMENT_GROUP_DIRECTORY = os.path.join(RESULTS_ROOT_DIRECTORY, EXPERIMENT_GROUP_NAME)

    @app.command()
    def train(env_name : str,
            results_directory : Optional[str] = None,
            debug : bool = typer.Option(False, '--debug')):
        # Train
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
    def run_local(debug : bool = typer.Option(False, '--debug')):
        results = {}
        for env_name in environment_names:
            train_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'train',env_name)
            test_directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'test')
            teacher_test_filename = os.path.join(test_directory,'teacher','%s.pkl' % env_name)
            random_test_filename = os.path.join(test_directory,'random','%s.pkl' % env_name)
            teacher_model_filename = os.path.join(train_directory,'output/deploy_state-best.pkl')
            dataset_filename = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'dataset','%s.pkl'%env_name)
            train(
                    env_name=env_name,
                    results_directory=train_directory,
                    debug=debug
            )
            test_saved_model(
                    env_name=env_name,
                    filename=teacher_model_filename,
                    output_filename=teacher_test_filename,
            )
            test_random(
                    env_name=env_name,
                    output_filename=random_test_filename,
            )
            generate_dataset(
                    env_name=env_name,
                    teacher_filename=teacher_model_filename,
                    output_filename=dataset_filename
            )
            for i in range(1,2):
                for model_name in models.keys():
                    for method in ['MSE', 'NLL', 'KL']:
                        distill(
                                env_name=env_name,
                                dataset_filename=dataset_filename,
                                output_filename=os.path.join(
                                    EXPERIMENT_GROUP_DIRECTORY,'distill',env_name,model_name,'%s-%d.pkl'%(method,i)
                                ),
                                model_name=model_name,
                                method=method,
                                n_test_runs=5,
                                debug=debug
                        )
            # Save results
            results[env_name] = {}
            with open(teacher_test_filename,'rb') as f:
                results[env_name]['teacher_score'] = dill.load(f)
            with open(random_test_filename,'rb') as f:
                results[env_name]['random_score'] = dill.load(f)
            results[env_name]['student_score'] = {
                    model_name: {method: [] for method in ['MSE','NLL','KL']}
                    for model_name in models.keys()
            }
            for model_name in models.keys():
                directory = os.path.join(EXPERIMENT_GROUP_DIRECTORY,'distill',env_name,model_name)
                for filename in os.listdir(directory):
                    match = re.match(r'([A-Z]+)-(\d+).pkl', filename)
                    method = match.group(1)
                    with open(os.path.join(directory,filename),'rb') as f:
                        results[env_name]['student_score'][model_name][method].append(
                                dill.load(f)
                        )
        breakpoint()

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


    commands = {
            'train': train,
            'test_saved_model': test_saved_model,
            'test_random': test_random,
            'generate_dataset': generate_dataset,
            'distill': distill,
            'run_local': run_local,
            'run_slurm': run_slurm,
    }

    return app, commands

if __name__ == '__main__':
    app,_ = make_app()
    app()
