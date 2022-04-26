import os
from pathlib import Path
from typing import Optional, Tuple, List
from pprint import pprint

import torch
import numpy as np
import gym
import gym.spaces
import dill
import experiment.logger
from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger

from rl.experiments.training.vectorized import TrainExperiment
from rl.experiments.training._utils import ExperimentConfigs
from rl.agent.smdp.a2c import PolicyValueNetworkRecurrent


class NArmBandits(gym.Env):
    def __init__(self, num_arms, num_trials, p: List[float] = None, reward: float = 1):
        self.num_arms = num_arms
        self.num_trials = num_trials
        self.reward = reward
        self._p = p

        self.observation_space = gym.spaces.Dict({
            'time': gym.spaces.Box(low=0, high=num_trials, shape=(), dtype=np.int32),
            'action': gym.spaces.Box(low=0, high=1, shape=(num_arms,), dtype=np.float32),
            'reward': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'done': gym.spaces.Discrete(2),
        })
        self.action_space = gym.spaces.Discrete(num_arms)

        self.current_step = 0

    def reset(self):
        if self._p is None:
            self.arms = np.random.rand(self.num_arms)
        else:
            self.arms = np.array(self._p)

        self._total_return = 0
        self._total_expected_return = 0

        self.current_step = 0
        obs = {
                'time': self.current_step,
                'reward': 0,
                'action': np.zeros(self.num_arms),
                'done': False,
        }
        return obs

    def step(self, action):
        self.current_step += 1
        if action >= self.num_arms:
            raise Exception(f'Invalid action {action}')

        reward = int(np.random.rand() < self.arms[action]) * self.reward
        done = self.current_step >= self.num_trials
        one_hot_action = np.zeros(self.num_arms)
        one_hot_action[action] = 1
        obs = {
                'time': self.current_step,
                'action': one_hot_action,
                'reward': reward,
                'done': done,
        }

        self._total_return += reward
        self._total_expected_return += self.arms[action]

        info = {
                'total_return': self._total_return,
                'total_expected_return': self._total_expected_return,
                'regret': self.arms.max()*self.current_step - self._total_expected_return,
                'max_regret': (self.arms.max() - self.arms.min()) * self.current_step,
                'mean_regret': self.arms.mean() * self.current_step,
        }
        return obs, reward, done, info


gym.register(
    id='NArmBandits-v0',
    entry_point=NArmBandits,
)


class Model(PolicyValueNetworkRecurrent):
    def __init__(self, num_actions):
        super().__init__()
        self._lstm_hidden_size = 48
        self.fc1 = torch.nn.Linear(num_actions+2, self._lstm_hidden_size)
        self.lstm = torch.nn.LSTMCell(self._lstm_hidden_size, self._lstm_hidden_size)
        self.fc2 = torch.nn.Linear(self._lstm_hidden_size, 32)
        self.baseline = torch.nn.Linear(32, 1)
        self.action = torch.nn.Linear(32, num_actions)

    def forward(self, x, hidden):
        concat = torch.cat([
            x['action'],
            x['time'].unsqueeze(1),
            x['reward'].unsqueeze(1),
        ], dim=1)
        x = self.fc1(concat)
        hidden = self.lstm(x, hidden)
        x = self.fc2(hidden[0])
        return {
            'action': self.action(x),
            'value': self.baseline(x),
            'hidden': hidden,
        }

    def init_hidden(self, batch_size=1) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self._lstm_hidden_size, device=device),
            torch.zeros(batch_size, self._lstm_hidden_size, device=device),
        )


def get_params():
    from rl.agent.smdp.a2c import PPOAgentRecurrentVec as AgentPPO

    params = ExperimentConfigs()

    num_envs = 16
    env_name = 'NArmBandits-v0'
    num_arms = 2
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'config': {
                'num_trials': 100,
                'num_arms': num_arms,
            },
        }] * num_envs
    }

    params.add('exp-001', {
        'agent': {
            'type': AgentPPO,
            'parameters': {
                'net': Model(num_actions=num_arms),
                'num_train_envs': num_envs,
                'num_test_envs': 1,
                'max_rollout_length': 100,
                'obs_scale': {},
                'learning_rate': 1e-3,
            },
        },
        'env_test': env_config,
        'env_train': env_config,
        'test_frequency': None,
        'save_model_frequency': None,
        'verbose': True,
    })

    return params


def make_app():
    import typer
    app = typer.Typer()

    @app.command()
    def run(exp_name : str,
            trial_id : Optional[str] = None,
            results_directory : Optional[str] = None,
            max_iterations : int = 5_000_000,
            slurm : bool = typer.Option(False, '--slurm'),
            wandb : bool = typer.Option(False, '--wandb'),
            debug : bool = typer.Option(False, '--debug')):
        config = get_params()[exp_name]
        pprint(config)
        if debug:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config={
                        **config,
                        #'save_model_frequency': 100_000, # This is number of iterations, not the number of transitions experienced
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    #checkpoint_frequency=250_000,
                    checkpoint_frequency=None,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
        else:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config={
                        **config,
                        #'save_model_frequency': 100_000, # This is number of iterations, not the number of transitions experienced
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=250_000,
                    #checkpoint_frequency=None,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
        if wandb:
            exp_runner.exp.logger.init_wandb({
                'project': f'PPO-bandits-{exp_name}'
            })
        def ep_end_callback(exp, x):
            _, _, done, info = x
            regrets = []
            mean_regrets = []
            normalized_regrets = []
            for i in range(len(done)):
                if done[i]:
                    regrets.append(info[i]['regret'])
                    mean_regrets.append(info[i]['mean_regret'])
                    normalized_regrets.append(info[i]['regret'] / info[i]['mean_regret'])
            exp.logger.log(
                regret = np.array(regrets).mean(),
                mean_regret = np.array(mean_regrets).mean(),
                normalized_regret = np.array(normalized_regrets).mean(),
            )
        exp_runner.exp.callbacks['on_episode_end'].append(ep_end_callback)
        exp_runner.run()
        exp_runner.exp.logger.finish_wandb()

    @app.command()
    def checkpoint(filename):
        exp = load_checkpoint(TrainExperiment, filename)
        exp.run()

    @app.command()
    def plot(result_directory : Path):
        import experiment.plotter as eplt
        from experiment.plotter import EMASmoothing

        checkpoint_filename = os.path.join(result_directory,'checkpoint.pkl')
        with open(checkpoint_filename,'rb') as f:
            x = dill.load(f)
        logger = Logger()
        logger.load_state_dict(x['exp']['logger'])
        if isinstance(logger.data, experiment.logger.FileBackedList):
            logger.data.iterate_past_end = True
        logger.load_to_memory(verbose=True)
        output_directory = x['exp']['output_directory']
        plot_directory = os.path.join(output_directory,'plots')
        os.makedirs(plot_directory,exist_ok=True)

        for k in ['agent_train_state_value_target_net', 'agent_train_state_value', 'train_reward', 'reward']:
            try:
                filename = os.path.abspath(os.path.join(plot_directory,f'plot-{k}.png'))
                eplt.plot(logger,
                        filename=filename,
                        curves=[{
                            'key': k,
                            'smooth_fn': EMASmoothing(0.9),
                        }],
                        min_points=2,
                        xlabel='Steps',
                        ylabel=k,
                        aggregate='mean',
                        show_unaggregated=False,
                )
                print(f'Plot saved to {filename}')
            except KeyError:
                print(f'Could not plot {k}. Key not found.')

    @app.command()
    def test(checkpoint_filename : Path, p : float = 0.1, max_reward: float = 1.0, output : Path = None):
        import matplotlib
        if output is not None:
            matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        num_trials = 10
        num_arms = 2
        exp = load_checkpoint(TrainExperiment, checkpoint_filename)
        env = gym.vector.SyncVectorEnv([
            lambda: NArmBandits(num_arms = num_arms, num_trials = 100, reward=max_reward, p=[p, 1-p]),
            lambda: NArmBandits(num_arms = num_arms, num_trials = 100, reward=max_reward, p=[1-p, p]),
        ])

        results = {}

        # Test agent
        results['agent'] = []
        agent = exp.exp.agent
        for _ in range(num_trials):
            results['agent'].append([])
            agent.reset()
            obs = env.reset()
            done = np.array([False] * env.num_envs)
            agent.observe(obs, testing=True)
            while not done[0]:
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                agent.observe(obs, reward, done, testing=True)
                print(f'{action} {reward} {info[0]["regret"]}')
                #results['agent'][-1].append(info[0]['regret'])
                results['agent'][-1].append([info[i]['regret'] for i in range(env.num_envs)])

        # Test UCB
        results['ucb'] = []
        for _ in range(num_trials):
            results['ucb'].append([])
            obs = env.reset()
            done = np.array([False] * env.num_envs)
            n = np.zeros([num_arms, env.num_envs])
            r = np.zeros([num_arms, env.num_envs])
            while not done[0]:
                if n.min() == 0:
                    action = np.argmin(n, axis=0)
                else:
                    ucb = r/n + np.sqrt(2 / n * np.log(1/0.05))
                    action = np.argmax(ucb, axis=0)
                obs, reward, done, info = env.step(action)
                for i in range(env.num_envs):
                    n[action[i], i] += 1
                    r[action[i], i] += reward[i]
                print(f'{action} {reward} {info[0]["regret"]}')
                results['ucb'][-1].append([info[i]['regret'] for i in range(env.num_envs)])

        # Plot
        for k,v in results.items():
            y = np.array(v).mean(2).mean(0)
            x = range(len(y))
            plt.plot(x,y,label=k)
        plt.title(f'p = {{ {p}, {1-p} }}, reward = {max_reward}')
        plt.xlabel('Trial #')
        plt.ylabel('Cumulative Regret')
        plt.grid()
        plt.legend()
        if output is not None:
            plt.savefig(output)
            print(f'Plot saved to {os.path.abspath(output)}')
        else:
            plt.show()

    commands = {
            'run': run,
            'checkpoint': checkpoint,
            'plot': plot,
            'test': test,
    }

    return app, commands


def run():
    app,_ = make_app()
    app()


if __name__ == "__main__":
    run()
