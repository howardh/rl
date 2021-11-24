from collections import defaultdict
from typing import Optional, Dict
import os

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
try:
    import pandas as pd
except:
    pass
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from experiment import Experiment, make_experiment_runner
from experiment.logger import Logger

from rl.agent import ReplayBuffer
from rl.agent.option_critic import OptionCriticAgent, make_agent_from_deploy_state
from rl.experiments.training.basic import make_env
from rl.experiments.distillation.rusu2016 import count_parameters, test

class OptionCriticNetworkCNN_1(torch.nn.Module):
    def __init__(self, num_actions, num_options):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=16,kernel_size=4,stride=2),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=16*9*9,out_features=512),
            torch.nn.ReLU(),
        )
        # Termination
        self.beta = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=num_options),
            torch.nn.Sigmoid(),
        )
        # Intra-option policies
        self.iop = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=num_actions*num_options),
            torch.nn.Unflatten(1, (num_options,num_actions))
        )
        # Policy over options
        self.poo = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=num_options),
        )
        # Option-value
        self.q = torch.nn.Sequential(
            torch.nn.Linear(in_features=512,out_features=num_options),
        )
    def forward(self, obs):
        x = obs
        x = self.head(x)
        x = {
            'beta': self.beta(x),
            'iop': self.iop(x),
            'poo': self.poo(x),
            'q': self.q(x),
        }
        return x
class OptionCriticNetworkCNN_2(torch.nn.Module):
    def __init__(self, num_actions, num_options):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=16,kernel_size=4,stride=2),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=16*9*9,out_features=256),
            torch.nn.ReLU(),
        )
        # Termination
        self.beta = torch.nn.Sequential(
            torch.nn.Linear(in_features=256,out_features=num_options),
            torch.nn.Sigmoid(),
        )
        # Intra-option policies
        self.iop = torch.nn.Sequential(
            torch.nn.Linear(in_features=256,out_features=num_actions*num_options),
            torch.nn.Unflatten(1, (num_options,num_actions))
        )
        # Policy over options
        self.poo = torch.nn.Sequential(
            torch.nn.Linear(in_features=256,out_features=num_options),
        )
        # Option-value
        self.q = torch.nn.Sequential(
            torch.nn.Linear(in_features=256,out_features=num_options),
        )
    def forward(self, obs):
        x = obs
        x = self.head(x)
        x = {
            'beta': self.beta(x),
            'iop': self.iop(x),
            'poo': self.poo(x),
            'q': self.q(x),
        }
        return x
class OptionCriticNetworkCNN_3(torch.nn.Module):
    def __init__(self, num_actions, num_options):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=16,out_channels=16,kernel_size=4,stride=2),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=16*9*9,out_features=128),
            torch.nn.ReLU(),
        )
        # Termination
        self.beta = torch.nn.Sequential(
            torch.nn.Linear(in_features=128,out_features=num_options),
            torch.nn.Sigmoid(),
        )
        # Intra-option policies
        self.iop = torch.nn.Sequential(
            torch.nn.Linear(in_features=128,out_features=num_actions*num_options),
            torch.nn.Unflatten(1, (num_options,num_actions))
        )
        # Policy over options
        self.poo = torch.nn.Sequential(
            torch.nn.Linear(in_features=128,out_features=num_options),
        )
        # Option-value
        self.q = torch.nn.Sequential(
            torch.nn.Linear(in_features=128,out_features=num_options),
        )
    def forward(self, obs):
        x = obs
        x = self.head(x)
        x = {
            'beta': self.beta(x),
            'iop': self.iop(x),
            'poo': self.poo(x),
            'q': self.q(x),
        }
        return x

class DistillationExperiment(Experiment):
    def setup(self, config, output_directory):
        self.output_directory = output_directory

        self.replay_buffer = ReplayBuffer(
                max_size=config.get('replay_buffer_size',500_000))
        self.batch_size = config.get('batch_size',32)
        self.train_frequency = config.get('train_frequency',50_000)
        self.train_iterations = config.get('train_iterations',10_000)
        self.test_iterations = config.get('test_iterations',2)
        self.test_frequency = config.get('test_frequency',50_000)
        #self.test_frequency = 10 # XXX: DEBUG
        #self.train_frequency = 10 # XXX: DEBUG

        env_name = config.get('env_name','ALE/Pong-v5')
        self.env = self._init_env(env_name)
        self.test_env = self._init_env(env_name)
        self.done = True

        # Init teachers
        self.teacher_agent = self._init_teacher(config['teacher_filename'])

        # Init Students
        self.student_models = self._init_students(self.env, self.teacher_agent)

        # Optimizer
        self.optimizer = {
                model_name: torch.optim.RMSprop(
                    model.parameters(),
                    lr=1e-4)
                for model_name,model in self.student_models.items()
        }

        # Parameter count
        self.num_params_teacher = count_parameters(self.teacher_agent.net)
        self.num_params = {
            model_name: count_parameters(model)
            for model_name,model in self.student_models.items()
        }

        # Baselines
        self.teacher_score = None
        self.random_score = None

        # Best Scores
        self.best_score = {
            model_name: float('-inf')
            for model_name in self.student_models.keys()
        }

        # Logs
        self.logger = Logger(key_name='step', allow_implicit_key=True)
    def _init_env(self, env_name):
        config = {
                'frameskip': 1,
                'mode': 0,
                'difficulty': 0,
                'repeat_action_probability': 0.25,
        }
        env = make_env(env_name, config=config, atari=True)
        return env
    def _init_teacher(self, filename) -> OptionCriticAgent:
        agent = make_agent_from_deploy_state(filename)
        return agent
    def _init_students(self, env, teacher) -> Dict[str,torch.nn.Module]:
        num_actions = env.action_space.n
        num_options = teacher.net.q[0].out_features
        students = {
            'OptionCriticNetworkCNN_1': OptionCriticNetworkCNN_1(
                num_actions=num_actions, num_options=num_options),
            'OptionCriticNetworkCNN_2': OptionCriticNetworkCNN_2(
                num_actions=num_actions, num_options=num_options),
            'OptionCriticNetworkCNN_3': OptionCriticNetworkCNN_3(
                num_actions=num_actions, num_options=num_options),
        }
        return students

    def run_step(self, iteration):
        self.logger.log(step=iteration)
        if iteration % self.test_frequency == 0:
            self._test()
            if iteration > 1:
                self._plot()
        self._generate_datapoint()
        if iteration % self.train_frequency == 0:
            self._train()
    def _generate_datapoint(self):
        env = self.env
        if self.done:
            self.done = False
            obs = env.reset()
            self.teacher_agent.observe(obs, testing=False)
        else:
            obs, reward, self.done, _ = env.step(self.teacher_agent.act(testing=False))
            self.teacher_agent.observe(obs, reward, self.done, testing=False)
        obs = torch.tensor(obs)
        self.replay_buffer.add(obs)
    def _train(self):
        criterion_kl = torch.nn.KLDivLoss(reduction='mean',log_target=True)
        criterion_mse = torch.nn.MSELoss(reduction='mean')
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        debug_loss = defaultdict(lambda: [])
        for _,obs in tqdm(zip(range(self.train_iterations),dataloader),desc='Training'):
            obs = obs.float()
            y_target = self.teacher_agent.net(obs)
            y_target['iop'] = y_target['iop'].detach()
            y_target['poo'] = y_target['poo'].detach()
            y_target['beta'] = y_target['beta'].detach()
            for model_name,model in self.student_models.items():
                optimizer = self.optimizer[model_name]
                y_pred = model(obs)

                loss_iop = criterion_kl(
                        y_pred['iop'].log_softmax(2),y_target['iop'].log_softmax(2))
                loss_poo = criterion_kl(
                        y_pred['poo'].log_softmax(1),y_target['poo'].log_softmax(1))
                loss_beta = criterion_mse(y_pred['beta'],y_target['beta'])
                loss = loss_iop + loss_poo + loss_beta

                debug_loss[model_name].append(loss.detach().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.logger.log(train_loss=dict(debug_loss))
    def _test(self):
        env = self.test_env
        agent = self.teacher_agent
        test_results = defaultdict(lambda: [])
        # Test parent
        temp = agent.net
        for model_name,model in self.student_models.items():
            agent.net = model
            results = [test(env,agent)['total_reward'] for _ in range(self.test_iterations)]
            test_results[model_name] = results
            # Check if it surpassed the best results so far
            if self.best_score[model_name] < np.mean(results):
                self.best_score[model_name] = np.mean(results)
                directory = os.path.join(self.output_directory,'best_models','parent')
                os.makedirs(directory,exist_ok=True)
                filename = os.path.join(directory,'%s.pt' % model_name)
                torch.save(model.state_dict(),filename)
                tqdm.write('Saved model %s with score %f (%s)' % (model_name, np.mean(results),os.path.abspath(filename)))
        agent.net = temp
        # Save results
        self.logger.log(test_results=dict(test_results))
    def _plot(self):
        # Training loss
        data = self.logger['train_loss']
        x = data[0]
        y_df = pd.DataFrame(data[1])
        y = {k:[np.array(l).mean() for l in v.to_list()] for k,v in y_df.items()}
        for k,v in y.items():
            plt.plot(x,v,label=k)
        plt.legend()
        plt.grid()
        plt.title('Distillation Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Training loss')

        filename = os.path.join(self.output_directory,'plot-loss.png')
        plt.savefig(filename)
        tqdm.write('Saved plot to %s' % os.path.abspath(filename))

        plt.yscale('log')
        filename = os.path.join(self.output_directory,'plot-loss-log.png')
        plt.savefig(filename)
        tqdm.write('Saved plot to %s' % os.path.abspath(filename))
        plt.close()

        # Testing reward (one plot for all models)
        data = self.logger['test_results']
        x = data[0]
        y_df = pd.DataFrame(data[1])
        y = {k:[np.array(l).mean() for l in v.to_list()] for k,v in y_df.items()}
        for k,v in y.items():
            plt.plot(x,v,label=k)
        plt.legend()
        plt.grid()
        plt.xlabel('Steps')
        plt.ylabel('Test Rewards')
        plt.title('Return Obtained from Distilled Models')
        filename = os.path.join(self.output_directory,'plot-rewards.png')
        plt.savefig(filename)
        tqdm.write('Saved plot to %s' % os.path.abspath(filename))
        plt.close()

        # Testing reward (one plot per model)
        num_subplots = len(y)
        nrows = int(np.ceil(np.sqrt(num_subplots)))
        ncols = int(np.ceil(num_subplots/nrows))
        for i,(k,v) in enumerate(y.items()):
            plt.subplot(nrows,ncols,i+1)
            plt.plot(x,v)
            plt.title(k)
            plt.grid()
            plt.xlabel('Steps')
            plt.ylabel('Test Rewards')
        plt.suptitle('Return Obtained from Distilled Models')
        plt.tight_layout()
        filename = os.path.join(self.output_directory,'plot-rewards-separate.png')
        plt.savefig(filename)
        tqdm.write('Saved plot to %s' % os.path.abspath(filename))
        plt.close()

        # Return versus compression percentage
        model_names = sorted(self.num_params.keys(),key=lambda k:self.num_params[k])
        x = [
                self.num_params[model_name]/self.num_params_teacher
                for model_name in model_names
        ]
        y = [ self.best_score[model_name] for model_name in model_names ]
        plt.plot(x,y,marker='o')
        plt.title('Performance vs Compression')
        plt.grid()
        plt.xlabel('Compression Relative to Teacher')
        plt.ylabel('Test Rewards')
        plt.gca().invert_xaxis()
        filename = os.path.join(self.output_directory,'plot-compression.png')
        plt.savefig(filename)
        tqdm.write('Saved plot to %s' % os.path.abspath(filename))
        plt.close()

def make_app():
    import typer
    app = typer.Typer()

    @app.command()
    def run(filename : str,
            results_directory : Optional[str] = None,
            slurm : bool = typer.Option(False, '--slurm'),
            debug : bool = typer.Option(False, '--debug')):

        exp_name = 'distillation'

        if debug:
            exp = make_experiment_runner(
                    DistillationExperiment,
                    experiment_name=exp_name,
                    verbose=True,
                    checkpoint_frequency=5,
                    max_iterations=30*5,
                    slurm_split=slurm,
                    results_directory=results_directory,
                    config={
                        'teacher_filename': filename
                    })
        else:
            exp = make_experiment_runner(
                    DistillationExperiment,
                    experiment_name=exp_name,
                    verbose=True,
                    checkpoint_frequency=50_000,
                    max_iterations=5_000_000,
                    slurm_split=slurm,
                    results_directory=results_directory,
                    config={
                        'teacher_filename': filename
                    })
        exp.run()

    @app.command()
    def foo():
        pass # Need a second command, or else it won't take a command name as the first argument (i.e. we'll have to run `script.py args` instead of `script.py run args`)

    commands = {
            'run': run,
            'foo': foo,
    }

    return app,commands

def run():
    app,_ = make_app()
    app()

if __name__ == "__main__":
    run()

