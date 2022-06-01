from typing import Mapping
import os
import gym
import torch
import dill
import numpy as np
from tqdm import tqdm
import envpool

from experiment import Experiment
from experiment.logger import Logger

from rl.utils import default_state_dict, default_load_state_dict
from rl.experiments.training._utils import make_env


#def make_vec_env(env_name, atari=False, atari_config={}):
#    if atari:
#        env = envpool.make(env_name, env_type="gym", **atari_config)
#        return env
#    else:
#        raise NotImplementedError()

def make_vec_env(env_type, env_configs):
    if env_type == 'envpool':
        if env_configs.get('atari',False):
            return envpool.make(env_configs['env_name'], env_type="gym", **env_configs['atari_config'])
        else:
            raise NotImplementedError()
    elif env_type == 'gym_async':
        return gym.vector.AsyncVectorEnv(
            [lambda: make_env(**config) for config in env_configs]
        )
    elif env_type == 'gym_sync':
        return gym.vector.SyncVectorEnv(
            [lambda: make_env(**config) for config in env_configs]
        )
    else:
        raise NotImplementedError()

class TrainExperiment(Experiment):
    """ An experiment for training a standard RL algorithm. """
    def setup(self, config, output_directory=None):
        assert output_directory is not None
        self.logger = Logger(
                key_name='step',
                allow_implicit_key=True,
                in_memory=False,
                max_file_length=config.get('logger_max_file_length', None),
                filename=os.path.join(output_directory,'log.pkl'))
        self.logger.log(step=0)

        self.device = self._init_device()

        self.test_frequency = config.get('test_frequency', None)
        self._num_train_envs = config['agent']['parameters'].get('num_train_envs', 8)
        self._num_test_envs = config['agent']['parameters'].get('num_test_envs', 8)

        if isinstance(config['env_test'], Mapping):
            self.env_test = make_vec_env(**config['env_test'])
        else:
            self.env_test = config['env_test']

        if isinstance(config['env_train'], Mapping):
            self.env_train = make_vec_env(**config['env_train'])
        else:
            self.env_train = config['env_train']

        self.agent = config.get('agent')
        self.agent = self._init_agent(
                agent_config=config.get('agent'),
                env = self.env_test,
                device = self.device,
        )

        self.output_directory = output_directory
        self.num_test_episodes = config.get('num_test_episodes', 5)
        self.save_model_frequency = config.get('save_model_frequency')
        self.verbose = config.get('verbose',False)

        self.done = np.array([False] * self._num_train_envs)
        self._ep_len = np.array([0] * self._num_train_envs)
        self._ep_rewards = np.array([0.] * self._num_train_envs)
        self._first_step = True

        self.callbacks = {
            'on_episode_end': [],
        }
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

        if isinstance(env, gym.vector.VectorEnv):
            obs_space = env.single_observation_space
            action_space = env.single_action_space
        else:
            obs_space = env.observation_space
            action_space = env.action_space

        return cls(
            action_space=action_space,
            observation_space=obs_space,
            logger=self.logger.make_sublogger('agent_'),
            **parameters,
            device=device,
        )
    def run_step(self, i):
        self.logger.log(step=i*self._num_train_envs) # x-axis = number of transitions experienced
        if self.test_frequency is not None and i % self.test_frequency == 0:
            self._test(i)
        if self.save_model_frequency is not None and i % self.save_model_frequency == 0:
            self._save_model(i)
        self._train(i)
    def _test(self,i):
        raise NotImplementedError(i)
    def _train(self,i):
        env = self.env_train
        agent = self.agent

        if self._first_step:
            obs = env.reset()
            agent.observe(obs, None, None, testing=False)
            self._first_step = False
        else:
            action = agent.act(testing=False)
            obs, reward, done, info = env.step(action)
            agent.observe(obs, reward, done, testing=False)
            # Logging
            self.logger.log(train_reward=reward)
            self._ep_len += 1
            self._ep_rewards += reward

            if done.any():
                if 'lives' in info:
                    real_done = np.logical_and(done, info['lives'] == 0)
                    if real_done.any():
                        mean_reward = self._ep_rewards[real_done].mean().item()
                        mean_length = self._ep_len[real_done].mean().item()
                        self.logger.log(reward=mean_reward, episode_length=mean_length)
                        self._ep_rewards = self._ep_rewards*(1-real_done)
                        self._ep_len = self._ep_len*(1-real_done)
                        tqdm.write(f'Iteration {i*self._num_train_envs:,}\t Training reward: {mean_reward}')
                else:
                    mean_reward = self._ep_rewards[done].mean().item()
                    mean_length = self._ep_len[done].mean().item()
                    self.logger.log(reward=mean_reward, episode_length=mean_length)
                    self._ep_rewards = self._ep_rewards*(1-done)
                    self._ep_len = self._ep_len*(1-done)
                    tqdm.write(f'Iteration {i*self._num_train_envs:,}\t Training reward: {mean_reward}')
                for callback in self.callbacks['on_episode_end']:
                    callback(self, (obs, reward, done, info))

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
            tqdm.write(f'Trained model saved to {os.path.abspath(model_file_name)}')
    def state_dict(self):
        """ Return the experiment state as a dictionary. """
        return default_state_dict(self, [
            'agent',
            'env_train','env_test',
            'output_directory',
            'test_frequency',
            'save_model_frequency',
            'verbose',
            'logger',
            'done',
            'env_train',
            '_ep_len',
            '_ep_rewards',
        ])
    def load_state_dict(self, state):
        default_load_state_dict(self, state)
        self.done = np.array([False] * self._num_train_envs)
        self._ep_len = np.array([0] * self._num_train_envs)
        self._ep_rewards = np.array([0.] * self._num_train_envs)
        self._first_step = True


if __name__=='__main__':
    from experiment import make_experiment_runner

    from rl.agent.smdp.a2c import A2CAgentVec as Agent
    #from rl.agent.smdp.a2c import A2CAgentRecurrentVec as Agent

    def train():
        num_envs = 32
        env_name = 'Pong-v5'
        #env_name = 'ALE/Pong-v5'
        #env_name = 'SeaquestNoFrameskip-v4'
        #env_name = 'MsPacmanNoFrameskip-v4'

        exp_runner = make_experiment_runner(
                TrainExperiment,
                config={
                    'agent': {
                        'type': Agent,
                        'parameters': {
                            'target_update_frequency': 32_000,
                            'num_train_envs':num_envs,
                            'num_test_envs': 5,
                            'max_rollout_length': 8,
                            #'hidden_reset_min_prob': 0,
                            #'hidden_reset_max_prob': 0,
                        }
                    },
                    'env_test': {
                        'env_type': 'envpool',
                        'env_configs': {'env_name': env_name, 'atari': True, 'atari_config': {'num_envs': num_envs}},
                    },
                    'env_train': {
                        'env_type': 'envpool',
                        'env_configs': {'env_name': env_name, 'atari': True, 'atari_config': {'num_envs': num_envs}},
                    },
                    #'env_test': {
                    #    'env_type': 'gym_async',
                    #    'env_configs': [{
                    #        'env_name': env_name,
                    #        'atari': True,
                    #        'config': {
                    #            'frameskip': 1,
                    #            'mode': 0,
                    #            'difficulty': 0,
                    #            'repeat_action_probability': 0.25,
                    #        }
                    #    } for _ in range(num_envs)],
                    #},
                    #'env_train': {
                    #    'env_type': 'gym_async',
                    #    'env_configs': [{
                    #        'env_name': env_name,
                    #        'atari': True,
                    #        'config': {
                    #            'frameskip': 1,
                    #            'mode': 0,
                    #            'difficulty': 0,
                    #            'repeat_action_probability': 0.25,
                    #        }
                    #    } for _ in range(num_envs)],
                    #},
                    'test_frequency': None,
                    'save_model_frequency': None,
                    'verbose': True,
                },
                #trial_id='checkpointtest',
                checkpoint_frequency=None,
                max_iterations=5_000_000,
                #max_iterations=1_000,
                #max_iterations=50_000_000,
                verbose=True,
        )
        exp_runner.exp.logger.init_wandb({
            'project': 'A2C-vec-%s' % env_name.replace('/','_')
        })
        try:
            exp_runner.run()
        except KeyboardInterrupt:
            breakpoint()

    def plot():
        pass

    train()
