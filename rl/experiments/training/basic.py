import os
import itertools
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

        self._train_env_keys = config.get('train_env_keys',[0])
        self._test_env_keys = config.get('test_env_keys',['test'])
        self.env_test = make_env(**config['env_test'])
        self.env_train = {
                k:make_env(**config['env_train'])
                for k in self._train_env_keys
        }

        self.agent = config.get('agent')
        self.agent = self._init_agent(
                agent_config=config.get('agent'),
                env = self.env_test,
                device = self.device,
        )

        self.output_directory = output_directory
        self.test_frequency = config.get('test_frequency')
        self.save_model_frequency = config.get('save_model_frequency')
        self.verbose = config.get('verbose',False)

        self.done = {k:True for k in self._train_env_keys}
        self._ep_len = {k:0 for k in self._train_env_keys}
        self._ep_rewards = {k:[] for k in self._train_env_keys}
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
        env_key = self._train_env_keys[i%len(self._train_env_keys)]
        env = self.env_train[env_key]
        self._train_one_env(i,env,env_key)
    def _train_one_env(self,i,env,env_key):
        agent = self.agent

        if self.done[env_key]:
            # TODO: Episode end callback?
            if self.verbose and self._ep_len[env_key] > 0:
                total_reward = sum(self._ep_rewards[env_key])
                self.logger.append(train_reward_by_episode=total_reward)
                #running_avg = np.mean([np.mean(x) for x in self.logger['train_reward_by_episode'][1][-100:]])
                #tqdm.write(f'Iteration {i}\t Training reward: {total_reward}\t avg: {running_avg}')
                tqdm.write(f'Iteration {i}\t Training reward: {total_reward}')
            # Reset
            self.done[env_key] = False
            obs = env.reset()
            agent.observe(obs, testing=False, env_key=env_key)
            self._ep_len[env_key] = 0
            self._ep_rewards[env_key].clear()
        else:
            action = agent.act(testing=False,env_key=env_key)
            obs, reward, self.done[env_key], _ = env.step(action)
            agent.observe(obs, reward, self.done[env_key], testing=False, env_key=env_key)
            # Logging
            self.logger.append(train_reward=reward)
            self._ep_len[env_key] += 1
            self._ep_rewards[env_key].append(reward)
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
        self.done = {k:True for k in self._train_env_keys}
        self._ep_len = {k:0 for k in self._train_env_keys}

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
