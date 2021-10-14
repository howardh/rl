import os
import itertools
import inspect
#from typing import Mapping

import gym
from gym.wrappers import FrameStack, AtariPreprocessing
import torch
import dill
import numpy as np
from tqdm import tqdm

from experiment import Experiment
from experiment.logger import Logger
from experiment.plotter import plot

from rl.utils import default_state_dict, default_load_state_dict
import rl.debug_tools.frozenlake

def _run_test(env, agent, verbose=False):
    steps = itertools.count()
    if verbose:
        steps = tqdm(steps, desc='test episode')

    total_reward = 0
    total_steps = 0

    obs = env.reset()
    agent.observe(obs, testing=True)
    for total_steps in steps:
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

def make_env(env_name, atari=False, one_hot_obs=False):
    env = gym.make(env_name)
    if atari:
        env = AtariPreprocessing(env)
        env = FrameStack(env, 4)
    if one_hot_obs:
        env = rl.debug_tools.frozenlake.OnehotObs(env)
    return env

class TrainExperiment(Experiment):
    """ An experiment for training a standard RL algorithm. """
    def setup(self, config, output_directory=None):
        self.logger = Logger(key_name='step',allow_implicit_key=True)
        self.logger.log(step=0)

        self.device = self._init_device()

        self.env_test = make_env(**config['env_test'])
        self.env_train = make_env(**config['env_train'])

        self.agent = config.get('agent')
        self.agent = self._init_agent(
                agent_config=config.get('agent'),
                env = self.env_train,
                device = self.device,
        )

        self.output_directory = output_directory
        self.test_frequency = config.get('test_frequency')
        self.save_model_frequency = config.get('save_model_frequency')
        self.verbose = config.get('verbose',False)

        self.done = True
        self._ep_len = 0
    def _init_device(self):
        if torch.cuda.is_available():
            print('GPU found')
            return torch.device('cuda')
        else:
            print('No GPU found. Running on CPU.')
            return torch.device('cpu')
    def _init_agent(self, agent_config, env, device):
        cls = agent_config['type']
        parameters = agent_config['parameters']
        return cls(
            action_space=env.action_space,
            observation_space=env.observation_space,
            logger=self.logger.make_sublogger('agent_'),
            **parameters,
            device=device,
        )
    def run_step(self, i):
        self.logger.log(step=i)
        if self.test_frequency is not None and i % self.test_frequency == 0:
            self._test(i)
        if self.save_model_frequency is not None and i % self.save_model_frequency == 0:
            self._save_model(i)
        self._train(i)
    def _test(self,i):
        test_results = [_run_test(self.env_test, self.agent, verbose=self.verbose) for _ in tqdm(range(5), desc='testing')]
        test_rewards = [x['total_reward'] for x in test_results]
        #action_values = np.mean(self.agent.logger[-1]['testing_action_value'])
        #tqdm.write(f'Iteration {i}\t Average reward: {avg}\t Action values: {action_values}')
        #tqdm.write(pprint.pformat(test_results[i], indent=4))
        #tqdm.write('Mean weights:')
        #tqdm.write(pprint.pformat([x.abs().mean().item() for x in agent.qu_net.parameters()], indent=4))
        self.logger.log(test_rewards=test_rewards, test_rewards_mean=np.mean(test_rewards))
        # Plot
        plot_filename = os.path.join(self.output_directory,'plot.png')
        plot(logger=self.logger, curves=['test_rewards'], filename=plot_filename, xlabel='steps', ylabel='test rewards')
        # Terminal output
        if self.verbose:
            mean_reward = np.mean(test_rewards)
            tqdm.write(f'Iteration {i}\t Average testing reward: {mean_reward}')
            tqdm.write('Plot saved to %s' % os.path.abspath(plot_filename))
    def _train(self,i):
        agent = self.agent
        env = self.env_train

        if self.done:
            # TODO: Episode end callback?
            if self.verbose and self._ep_len > 0:
                total_reward = sum(self.logger['train_reward'][1][-self._ep_len:])
                tqdm.write(f'Iteration {i}\t Training reward: {total_reward}')
                self.logger.log(train_reward_by_episode=total_reward)
            # Reset
            obs = env.reset()
            agent.observe(obs, testing=False)
            self._ep_len = 0

        # Transition
        obs, reward, self.done, _ = env.step(agent.act(testing=False))
        agent.observe(obs, reward, self.done, testing=False)
        # Logging
        self.logger.log(train_reward=reward)
        self._ep_len += 1
    def _save_model(self,i):
        """ Save the model parameters """
        models_path = os.path.join(self.output_directory, 'models')
        os.makedirs(models_path, exist_ok=True)
        model_file_name = os.path.join(models_path, '%d.pkl' % i)
        if hasattr(self.agent,'state_dict_deploy'):
            model = self.agent.state_dict_deploy()
        else:
            model = self.agent.state_dict()
        with open(model_file_name,'wb') as f:
            dill.dump(model, f)
        # Verbose
        if self.verbose:
            tqdm.write(f'Trained model saved to {model_file_name}')
    def state_dict(self):
        """ Return the experiment state as a dictionary. """
        return default_state_dict(self, [
            'output_directory',
            'test_frequency',
            'save_model_frequency',
            'verbose',
            'logger',
            'done',
            'env_train',
            '_ep_len',
        ])
    def load_state_dict(self, state):
        default_load_state_dict(self, state)
        self.done = True
        self._ep_len = 0

if __name__=='__main__':
    from experiment import make_experiment_runner

    from rl.agent.smdp.dqn import DQNAgent

    env_name = 'PongNoFrameskip-v4'
    #env_name = 'SeaquestNoFrameskip-v4'
    #env_name = 'MsPacmanNoFrameskip-v4'

    #agent = DQNAgent(
    #    action_space=env_train.action_space,
    #    observation_space=env_train.observation_space,
    #    #discount_factor=0.99,
    #    #behaviour_eps=0.02,
    #    #learning_rate=1e-4,
    #    #update_frequency=1,
    #    #target_update_frequency=1_000,
    #    #polyak_rate=1,
    #    #warmup_steps=10_000,
    #    #replay_buffer_size=100_000,
    #    device=device,
    #)

    exp_runner = make_experiment_runner(
            TrainExperiment,
            config={
                'agent': {
                    'type': DQNAgent,
                    'parameters': {
                        'discount_factor': 0.99,
                        'behaviour_eps': 0.02,
                        'learning_rate': 1e-4,
                        'update_frequency': 1,
                        'target_update_frequency': 1_000,
                        'polyak_rate': 1,
                        'warmup_steps': 10_000,
                        'replay_buffer_size': 100_000,
                    }
                },
                'env_test': {'env_name': env_name, 'atari': True},
                'env_train': {'env_name': env_name, 'atari': True},
                'test_frequency': 10_000,
                #'test_frequency': 50_000,
                #'save_model_frequency': 100_000,
                #'save_model_frequency': 250_000,
                'save_model_frequency': 50_000,
                'verbose': True,
            },
            #trial_id='checkpointtest',
            checkpoint_frequency=50_000,
            #checkpoint_frequency=250_000,
            #checkpoint_frequency=1_000_000,
            max_iterations=5_000_000,
            #max_iterations=50_000_000,
            verbose=True,
    )
    exp_runner.exp.logger.init_wandb({
        'project': 'DQN-%s' % env_name
    })
    try:
        exp_runner.run()
    except KeyboardInterrupt:
        breakpoint()
