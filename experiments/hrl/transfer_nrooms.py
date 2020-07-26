import numpy as np
import gym
import torch
from tqdm import tqdm
import dill
import os
import itertools
from collections import defaultdict
import pprint
from skopt import gp_minimize

from agent.hdqn_agent import HDQNAgentWithDelayAC, HDQNAgentWithDelayAC_v2, HDQNAgentWithDelayAC_v3
from agent.policy import get_greedy_epsilon_policy

from .model import QFunction, PolicyFunction, PolicyFunctionAugmentatedState
from .long_trial import plot

import utils
import hyperparams
import hyperparams.utils
from hyperparams.distributions import Uniform, LogUniform, CategoricalUniform, DiscreteUniform

##################################################
# Search Space
##################################################

def get_search_space():
    actor_critic_hyperparam_space = {
            'agent_name': 'ActorCritic',
            'gamma': 0.9,
            'controller_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_learning_rate': LogUniform(1e-4,1e-1),
            'q_net_learning_rate': LogUniform(1e-4,1e-1),
            'eps_b': Uniform(0,0.5),
            'polyak_rate': 0.001,
            'batch_size': 256,
            'min_replay_buffer_size': 1000,
            'steps_per_task': 10000,
            'total_steps': 100000,
            'epoch': 1000,
            'test_iters': 5,
            'verbose': True,
            'num_options': DiscreteUniform(2,10),
            'cnet_n_layers': DiscreteUniform(1,2),
            'cnet_layer_size': DiscreteUniform(1,20),
            'snet_n_layers': DiscreteUniform(1,2),
            'snet_layer_size': DiscreteUniform(1,5),
            'qnet_n_layers': DiscreteUniform(1,2),
            'qnet_layer_size': DiscreteUniform(10,20),
            'directory': None
    }
    hrl_delay_augmented_hyperparam_space = {
            'agent_name': 'HDQNAgentWithDelayAC_v3',
            'gamma': 0.9,
            'controller_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_learning_rate': LogUniform(1e-4,1e-1),
            'q_net_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_q_net_learning_rate': LogUniform(1e-4,1e-1),
            'eps_b': Uniform(0,0.5),
            'polyak_rate': 0.001,
            'batch_size': 256,
            'min_replay_buffer_size': 1000,
            'steps_per_task': 10000,
            'total_steps': 100000,
            'epoch': 1000,
            'test_iters': 5,
            'verbose': True,
            'num_options': DiscreteUniform(2,10),
            'cnet_n_layers': DiscreteUniform(1,2),
            'cnet_layer_size': DiscreteUniform(1,20),
            'snet_n_layers': DiscreteUniform(1,2),
            'snet_layer_size': DiscreteUniform(1,5),
            'qnet_n_layers': DiscreteUniform(1,2),
            'qnet_layer_size': DiscreteUniform(10,20),
            'directory': None
    }
    hrl_delay_memoryless_hyperparam_space = {
            'agent_name': 'HDQNAgentWithDelayAC_v2',
            'gamma': 0.9,
            'controller_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_learning_rate': LogUniform(1e-4,1e-1),
            'q_net_learning_rate': LogUniform(1e-4,1e-1),
            'subpolicy_q_net_learning_rate': LogUniform(1e-4,1e-1),
            'eps_b': Uniform(0,0.5),
            'polyak_rate': 0.001,
            'batch_size': 256,
            'min_replay_buffer_size': 1000,
            'steps_per_task': 10000,
            'total_steps': 100000,
            'epoch': 1000,
            'test_iters': 5,
            'verbose': True,
            'num_options': DiscreteUniform(2,10),
            'cnet_n_layers': DiscreteUniform(1,2),
            'cnet_layer_size': DiscreteUniform(1,20),
            'snet_n_layers': DiscreteUniform(1,2),
            'snet_layer_size': DiscreteUniform(1,5),
            'qnet_n_layers': DiscreteUniform(1,2),
            'qnet_layer_size': DiscreteUniform(10,20),
            'directory': None
    }
    return {
            'ActorCritic': actor_critic_hyperparam_space,
            'HDQNAgentWithDelayAC_v2': hrl_delay_memoryless_hyperparam_space,
            'HDQNAgentWithDelayAC_v3': hrl_delay_augmented_hyperparam_space
    }

space = get_search_space()

##################################################
# Environment
##################################################

class TimeLimit(gym.Wrapper):
    """ Copied from gym.wrappers.TimeLimit with small modifications."""
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info
    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

##################################################
# Agent
##################################################

def create_agent(agent_name, env, device, seed, **agent_params):
    before_step = lambda s: None
    after_step = lambda s: None

    gamma = agent_params.pop('gamma',0.9)
    eps_b = agent_params.pop('eps_b',0.05)
    num_options = agent_params.pop('num_options',3)
    min_replay_buffer_size = agent_params.pop('min_replay_buffer_size',1000)
    batch_size = agent_params.pop('batch_size',256)
    controller_learning_rate = agent_params.pop('controller_learning_rate',0.01)
    subpolicy_learning_rate = agent_params.pop('subpolicy_learning_rate',0.01)
    q_net_learning_rate = agent_params.pop('q_net_learning_rate',0.01)
    polyak_rate = agent_params.pop('polyak_rate',0.001)
    replay_buffer_size = agent_params.pop('replay_buffer_size',10000)
    delay = agent_params.pop('delay',1)
    if agent_name == 'HDQNAgentWithDelayAC':
        cnet_n_layers = agent_params.pop(
                'cnet_n_layers',None)
        cnet_layer_size = agent_params.pop(
                'cnet_layer_size',None)
        snet_n_layers = agent_params.pop(
                'snet_n_layers',None)
        snet_layer_size = agent_params.pop(
                'snet_layer_size',None)
        qnet_n_layers = agent_params.pop(
                'qnet_n_layers',None)
        qnet_layer_size = agent_params.pop(
                'qnet_layer_size',None)
        controller_net_structure = tuple([cnet_layer_size]*cnet_n_layers)
        subpolicy_net_structure = tuple([snet_layer_size]*snet_n_layers)
        q_net_structure = tuple([qnet_layer_size]*qnet_n_layers)
        agent = HDQNAgentWithDelayAC(
                action_space=env.action_space,
                observation_space=env.observation_space,
                controller_learning_rate=controller_learning_rate,
                subpolicy_learning_rate=subpolicy_learning_rate,
                q_net_learning_rate=q_net_learning_rate,
                discount_factor=gamma,
                polyak_rate=polyak_rate,
                device=device,
                behaviour_epsilon=eps_b,
                replay_buffer_size=replay_buffer_size,
                delay_steps=1,
                controller_net=PolicyFunction(
                    layer_sizes=controller_net_structure,
                    input_size=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(
                    layer_sizes=subpolicy_net_structure,input_size=4)
                    for _ in range(num_options)],
                q_net=QFunction(layer_sizes=q_net_structure,
                    input_size=4,output_size=4),
                seed=seed,
        )
        def before_step(steps):
            #agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
            pass
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'HDQNAgentWithDelayAC_v2':
        cnet_n_layers = agent_params.pop(
                'cnet_n_layers',None)
        cnet_layer_size = agent_params.pop(
                'cnet_layer_size',None)
        snet_n_layers = agent_params.pop(
                'snet_n_layers',None)
        snet_layer_size = agent_params.pop(
                'snet_layer_size',None)
        qnet_n_layers = agent_params.pop(
                'qnet_n_layers',None)
        qnet_layer_size = agent_params.pop(
                'qnet_layer_size',None)
        controller_net_structure = tuple([cnet_layer_size]*cnet_n_layers)
        subpolicy_net_structure = tuple([snet_layer_size]*snet_n_layers)
        q_net_structure = tuple([qnet_layer_size]*qnet_n_layers)
        subpolicy_q_net_learning_rate = agent_params.pop(
                'subpolicy_q_net_learning_rate',1e-3)
        agent = HDQNAgentWithDelayAC_v2(
                action_space=env.action_space,
                observation_space=env.observation_space,
                controller_learning_rate=controller_learning_rate,
                subpolicy_learning_rate=subpolicy_learning_rate,
                q_net_learning_rate=q_net_learning_rate,
                subpolicy_q_net_learning_rate=subpolicy_q_net_learning_rate,
                discount_factor=gamma,
                polyak_rate=polyak_rate,
                device=device,
                behaviour_epsilon=eps_b,
                replay_buffer_size=replay_buffer_size,
                delay_steps = delay,
                controller_net=PolicyFunction(
                    layer_sizes=controller_net_structure,
                    input_size=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(
                    layer_sizes=subpolicy_net_structure,input_size=4)
                    for _ in range(num_options)],
                q_net=QFunction(layer_sizes=q_net_structure,
                    input_size=4,output_size=4),
                seed=seed,
        )
        def before_step(steps):
            #agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
            pass
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'HDQNAgentWithDelayAC_v3':
        cnet_n_layers = agent_params.pop(
                'cnet_n_layers',None)
        cnet_layer_size = agent_params.pop(
                'cnet_layer_size',None)
        snet_n_layers = agent_params.pop(
                'snet_n_layers',None)
        snet_layer_size = agent_params.pop(
                'snet_layer_size',None)
        qnet_n_layers = agent_params.pop(
                'qnet_n_layers',None)
        qnet_layer_size = agent_params.pop(
                'qnet_layer_size',None)
        controller_net_structure = tuple([cnet_layer_size]*cnet_n_layers)
        subpolicy_net_structure = tuple([snet_layer_size]*snet_n_layers)
        q_net_structure = tuple([qnet_layer_size]*qnet_n_layers)
        subpolicy_q_net_learning_rate = agent_params.pop(
                'subpolicy_q_net_learning_rate',1e-3)
        agent = HDQNAgentWithDelayAC_v3(
                action_space=env.action_space,
                observation_space=env.observation_space,
                controller_learning_rate=controller_learning_rate,
                subpolicy_learning_rate=subpolicy_learning_rate,
                q_net_learning_rate=q_net_learning_rate,
                subpolicy_q_net_learning_rate=subpolicy_q_net_learning_rate,
                discount_factor=gamma,
                polyak_rate=polyak_rate,
                device=device,
                behaviour_epsilon=eps_b,
                replay_buffer_size=replay_buffer_size,
                delay_steps = delay,
                controller_net=PolicyFunctionAugmentatedState(
                    layer_sizes=controller_net_structure,state_size=4,
                    num_actions=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(
                    layer_sizes=subpolicy_net_structure,input_size=4)
                    for _ in range(num_options)],
                q_net=QFunction(layer_sizes=q_net_structure,
                    input_size=4,output_size=4),
                seed=seed,
        )
        def before_step(steps):
            #agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
            pass
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'ActorCritic':
        ['cnet_n_layers', 'cnet_layer_size', 'snet_n_layers',
                'snet_layer_size', 'qnet_n_layers', 'qnet_layer_size']
        cnet_n_layers = agent_params.pop(
                'cnet_n_layers',None)
        cnet_layer_size = agent_params.pop(
                'cnet_layer_size',None)
        snet_n_layers = agent_params.pop(
                'snet_n_layers',None)
        snet_layer_size = agent_params.pop(
                'snet_layer_size',None)
        qnet_n_layers = agent_params.pop(
                'qnet_n_layers',None)
        qnet_layer_size = agent_params.pop(
                'qnet_layer_size',None)
        controller_net_structure = tuple([cnet_layer_size]*cnet_n_layers)
        subpolicy_net_structure = tuple([snet_layer_size]*snet_n_layers)
        q_net_structure = tuple([qnet_layer_size]*qnet_n_layers)
        agent = HDQNAgentWithDelayAC(
                action_space=env.action_space,
                observation_space=env.observation_space,
                controller_learning_rate=controller_learning_rate,
                subpolicy_learning_rate=subpolicy_learning_rate,
                q_net_learning_rate=q_net_learning_rate,
                discount_factor=gamma,
                polyak_rate=polyak_rate,
                device=device,
                behaviour_epsilon=eps_b,
                replay_buffer_size=replay_buffer_size,
                delay_steps = 0,
                controller_net=PolicyFunction(
                    layer_sizes=controller_net_structure,
                    input_size=4,output_size=num_options),
                subpolicy_nets=[PolicyFunction(
                    layer_sizes=subpolicy_net_structure,input_size=4)
                    for _ in range(num_options)],
                q_net=QFunction(layer_sizes=q_net_structure,
                    input_size=4,output_size=4),
                seed=seed,
        )
        def before_step(steps):
            #agent.behaviour_epsilon = (1-min(steps/1000000,1))*(1-eps_b)+eps_b
            pass
        def after_step(steps):
            if steps >= min_replay_buffer_size:
                agent.train(batch_size=batch_size,iterations=1)
    elif agent_name == 'OptionCritic':
        raise NotImplementedError('Option Critic not implemented yet.')

    if len(agent_params) > 0:
        raise Exception('Unused agent parameters: %s' % agent_params.keys())

    return agent, before_step, after_step

##################################################
# Experiment
##################################################

def checkpoint_directory():
    """ Return the path where all checkpoints are saved. """
    directory = os.path.join(utils.get_results_directory(),__name__)
    checkpoint_directory = os.path.join(utils.get_results_directory(),'checkpoints')
    if not os.path.isdir(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    return checkpoint_directory

def checkpoint_path():
    """ Return the path of checkpoint associated with the current job ID and array task ID. """
    directory = checkpoint_directory()

    try:
        job_id = os.environ['SLURM_ARRAY_JOB_ID']
        task_id = os.environ['SLURM_ARRAY_TASK_ID']
        file_name = '%s_%s.checkpoint.pkl' % (job_id,task_id)
    except:
        job_id = os.environ['SLURM_JOB_ID']
        file_name = '%s.checkpoint.pkl' % (job_id)
    file_path = os.path.join(directory,file_name)
    return file_path

def list_checkpoints():
    """ Generate the path to all available checkpoints """
    directory = checkpoint_directory()
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            yield path

def delete_checkpoint():
    os.remove(checkpoint_path())

def run_trial_with_checkpoint(**params):
    print(params['directory'])
    path = checkpoint_path()
    if os.path.isfile(path):
        print('Found Checkpoint. Restoring state...')
        exp = Experiment.from_checkpoint(path)
    else:
        print('No Checkpoint.')
        exp = Experiment(**params)
    #if state is not None:
    #    print('Found Checkpoint. Restoring state...')
    #    for k,v in params.items():
    #        if k in state['args'] and state['args'][k] != v:
    #            print('Overwriting %s=%s with %s' % (k,state['args'][k],v))
    #            state['args'][k] = v
    #    exp.load_state_dict(state)
    #else:
    #    print('No Checkpoint.')
    return exp.run()

class Experiment:
    def __init__(self, directory=None, steps_per_task=100, total_steps=1000,
            epoch=50, test_iters=1, verbose=False, seed=None, keep_checkpoint=False, checkpoint_path=None,
            agent_name='HDQNAgentWithDelayAC', **agent_params):
        self.args = locals()
        del self.args['self']

        self.directory = directory
        self.steps_per_task = steps_per_task
        self.total_steps = total_steps
        self.epoch = epoch
        self.test_iters = test_iters
        self.verbose = verbose
        self.seed = seed
        self.keep_checkpoint = keep_checkpoint
        self.checkpoint_path = checkpoint_path
        self.agent_name = agent_name
        self.agent_params = agent_params
        self.steps = 0

        env_name='gym_fourrooms:fourrooms-v0'
        #pprint.pprint(self.args)
        if seed is not None:
            torch.manual_seed(seed) # Required for consistent random initialization of neural net weights

        rand = np.random.RandomState(seed)

        self.env = gym.make(env_name,goal_duration_steps=steps_per_task).unwrapped
        self.env = TimeLimit(self.env,36)
        self.test_env = gym.make(env_name,goal_duration_steps=float('inf')).unwrapped
        self.test_env = TimeLimit(self.test_env,500)
        if seed is not None:
            self.env.seed(seed)
            self.test_env.seed(seed+1) # Use a different seed so we don't get the same sequence of states as env

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.agent, self.before_step, self.after_step = create_agent(
                agent_name, self.env, device, seed, **agent_params)

        self.results_file_path = None
        self.rewards = []
        self.state_action_values = []
        self.steps_to_reward = []
        self.eval_steps = []
        self.step_range = range(self.total_steps)
        self.done = True

    def run_step(self):
        # Checkpoint
        if self.steps % self.steps_per_task == 0:
            state = self.state_dict()
            path = self.checkpoint_path
            if path is None:
                path = checkpoint_path()
            with open(path,'wb') as f:
                dill.dump(state,f)
            tqdm.write('Checkpoint saved')
        # Run tests
        if self.steps % self.epoch == 0:
            self.test_env.reset_goal(self.env.goal)
            test_results = self.agent.test(
                    self.test_env, self.test_iters, render=False, processors=1)
            self.rewards.append(np.mean(
                [r['total_rewards'] for r in test_results]))
            self.state_action_values.append(np.mean(
                [r['state_action_values'] for r in test_results]))
            self.steps_to_reward.append(np.mean(
                [r['steps'] for r in test_results]))
            self.eval_steps.append(self.steps)
            if self.verbose:
                tqdm.write('steps %d \t Reward: %f \t Steps: %f' % (
                    self.steps, self.rewards[-1], self.steps_to_reward[-1]))

        self.before_step(self.steps)
        # Run step
        if self.done:
            self.obs = self.env.reset()
            self.agent.observe_change(self.obs, None)
        self.obs, self.reward, self.done, _ = self.env.step(self.agent.act())
        self.agent.observe_change(self.obs, self.reward, terminal=self.done)
        # Update weights
        self.after_step(self.steps)

    def run(self):
        if self.verbose:
            step_range = tqdm(range(self.total_steps), total=self.total_steps, initial=self.step_range.start)
        try:
            for self.steps in step_range:
                self.run_step()
            self.save_results()
            if not self.keep_checkpoint:
                delete_checkpoint()
        except KeyboardInterrupt:
            pass

        return self.steps_to_reward

    def save_results(self, additional_data={}):
        results = self.state_dict()

        # Add additional data
        for k,v in additional_data.items():
            if k in results:
                print('WARNING: OVERWRITING KEY %s' % k)
            results[k] = v
        # Save results
        if self.results_file_path is None:
            self.results_file_path = utils.save_results(
                    results,
                    directory=self.directory,
                    file_name_prefix=self.agent_name)
        else:
            utils.save_results(
                    results,
                    file_path=self.results_file_path)

    def state_dict(self):
        return {
            'args': self.args,
            'agent_state': self.agent.state_dict(),
            'results_file_path': self.results_file_path,
            'rewards': self.rewards,
            'state_action_values': self.state_action_values,
            'steps_to_reward': self.steps_to_reward,
            'eval_steps': self.eval_steps,
            'steps': self.steps,
            'env': self.env.state_dict(),
            'test_env': self.test_env.state_dict(),
            'env_steps': self.env._elapsed_steps,
            'done': self.done
        }

    def load_state_dict(self, state):
        # Env
        self.env.load_state_dict(state['env'])
        self.test_env.load_state_dict(state['test_env'])
        self.env._elapsed_steps = state['env_steps']
        
        # Agent
        if self.agent_name != state['args']['agent_name']:
            raise Exception('Agent type mismatch. Expected %s, received %s.' % (self.agent_name, state['agent_name']))
        self.agent.load_state_dict(state['agent_state'])

        # Experiment progress
        self.results_file_path = state['results_file_path']
        self.rewards = state['rewards']
        self.state_action_values = state['state_action_values']
        self.steps_to_reward = state['steps_to_reward']
        self.eval_steps = state['eval_steps']
        self.steps = state['steps']
        self.step_range = range(self.steps,self.total_steps)
        self.done = state['done']

    @staticmethod
    def from_checkpoint(file_name=None):
        if file_name is None:
            file_name = checkpoint_path()
        if os.path.isfile(file_name):
            with open(file_name,'rb') as f:
                state = dill.load(f)
        else:
            raise Exception('Checkpoint does not exist: %s' % file_name)

        agent_params = state['args'].pop('agent_params')
        exp = Experiment(**state['args'],**agent_params)
        exp.load_state_dict(state)
        exp.checkpoint = state
        return exp

class ExperimentRandomControllerDropout:
    def __init__(self, directory=None, max_steps=500, test_iters=1, dropout_probability=0.5,
            verbose=False, seed=None, initial_checkpoint=None):
        self.args = locals()
        del self.args['self']

        self.directory = directory
        self.max_steps = max_steps
        self.test_iters = test_iters
        self.verbose = verbose
        self.seed = seed
        checkpoint = Experiment.from_checkpoint(initial_checkpoint)
        self.agent = checkpoint.agent
        self.agent.controller_dropout = dropout_probability

        env_name='gym_fourrooms:fourrooms-v0'
        if seed is not None:
            torch.manual_seed(seed) # Required for consistent random initialization of neural net weights

        rand = np.random.RandomState(seed)

        self.env = gym.make(env_name,goal_duration_episodes=1).unwrapped
        self.env = TimeLimit(self.env,max_steps)
        if seed is not None:
            self.env.seed(seed)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self.results_file_path = None
        self.steps_to_reward = []
        self.done = True

    def run_step(self):
        # Run episode with random controller dropout, and measure performance in number of steps to reach the goal
        test_results = self.agent.test(
                self.env, self.test_iters, render=False, processors=1)

    def run(self):
        step_range = range(self.test_iters)
        if self.verbose:
            step_range = tqdm(step_range)
        try:
            results = []
            for self.steps in step_range:
                test_results = self.agent.test(
                        self.env, 1, render=False, processors=1)
                results.append(test_results[0]['steps'])
                print(test_results[0]['steps'])
            print(np.mean(results))
        except KeyboardInterrupt:
            pass

        return self.steps_to_reward

    def save_results(self, additional_data={}):
        results = self.state_dict()

        # Add additional data
        for k,v in additional_data.items():
            if k in results:
                print('WARNING: OVERWRITING KEY %s' % k)
            results[k] = v
        # Save results
        if self.results_file_path is None:
            self.results_file_path = utils.save_results(
                    results,
                    directory=self.directory,
                    file_name_prefix=self.agent_name)
        else:
            utils.save_results(
                    results,
                    file_path=self.results_file_path)

    def state_dict(self):
        return {
            'args': self.args,
            'agent_state': self.agent.state_dict(),
            'results_file_path': self.results_file_path,
            'rewards': self.rewards,
            'state_action_values': self.state_action_values,
            'steps_to_reward': self.steps_to_reward,
            'eval_steps': self.eval_steps,
            'steps': self.steps,
            'env': self.env.state_dict(),
            'test_env': self.test_env.state_dict(),
            'env_steps': self.env._elapsed_steps,
            'done': self.done
        }

    def load_state_dict(self, state):
        # Env
        self.env.load_state_dict(state['env'])
        self.test_env.load_state_dict(state['test_env'])
        self.env._elapsed_steps = state['env_steps']
        
        # Agent
        if self.agent_name != state['args']['agent_name']:
            raise Exception('Agent type mismatch. Expected %s, received %s.' % (self.agent_name, state['agent_name']))
        self.agent.load_state_dict(state['agent_state'])

        # Experiment progress
        self.results_file_path = state['results_file_path']
        self.rewards = state['rewards']
        self.state_action_values = state['state_action_values']
        self.steps_to_reward = state['steps_to_reward']
        self.eval_steps = state['eval_steps']
        self.steps = state['steps']
        self.step_range = range(self.steps,self.total_steps)
        self.done = state['done']

    @staticmethod
    def from_checkpoint(file_name=None):
        if file_name is None:
            file_name = checkpoint_path()
        if os.path.isfile(file_name):
            with open(file_name,'rb') as f:
                state = dill.load(f)
        else:
            raise Exception('Checkpoint does not exist: %s' % file_name)

        agent_params = state['args'].pop('agent_params')
        exp = Experiment(**state['args'],**agent_params)
        exp.load_state_dict(state)
        exp.checkpoint = state
        return exp

##################################################
# Hyperparameter Search
##################################################

def run_hyperparam_search_extremes(space, proc=1):
    directory = os.path.join(utils.get_results_directory(),__name__)
    params = hyperparams.utils.list_extremes(space)
    params = utils.split_params(params[int(os.environ['SLURM_ARRAY_TASK_MIN']):(int(os.environ['SLURM_ARRAY_TASK_MAX'])+1)])
    funcs = [lambda: run_trial_with_checkpoint(**p) for p in params]
    utils.cc(funcs,proc=proc)
    return utils.get_all_results(directory)

def run_hyperparam_search(space, proc=1):
    directory = os.path.join(utils.get_results_directory(),__name__,'random')
    ddict = {'directory': directory}
    params = [{**hyperparams.utils.sample_hyperparam(space),**ddict} for _ in range(proc)]
    funcs = [lambda: run_trial_with_checkpoint(**p) for p in params]
    utils.cc(funcs,proc=proc)

def random_lin_comb(vecs):
    n_points = vecs.shape[0]
    weights = np.random.rand(n_points)
    weights /= np.sum(weights)
    return (weights.reshape(n_points,1)*vecs).sum(0)

def sample_convex_hull(results_directory, agent_name='ActorCritic', threshold=0.1, perturbance=0):
    """ Find the top {threshold}% of parameters, and sample a set of parameters
    within the convex hull formed by those points.
    """
    from scipy.spatial import ConvexHull
    scores = compute_score(results_directory,sortby='mean')[agent_name]
    params = []
    for p,s in scores[:n_points]:
        params.append(hyperparams.utils.param_to_vec(p,space[agent_name]))
    params = np.array(params)
    hull = ConvexHull(params)
    output = random_lin_comb(hull)
    output = output.tolist()
    if perturbance > 0:
        output = hyperparams.utils.perturb_vec(output,space[agent_name],perturbance)
    output = hyperparams.utils.vec_to_param(output,space[agent_name])
    return output

def bin_lsh(data, space, n_planes=4):
    random_planes = []
    for _ in range(n_planes):
        u = hyperparams.utils.sample_hyperparam(space)
        v = hyperparams.utils.sample_hyperparam(space)
        u = hyperparams.utils.param_to_vec(u,space)
        v = hyperparams.utils.param_to_vec(v,space)
        u = torch.tensor(u)
        v = torch.tensor(v)
        random_planes.append((u,v))

    bins = [[] for _ in range(1<<n_planes)]
    for p,v in data:
        # Convert params to vector
        x = hyperparams.utils.param_to_vec(p,space)
        x = torch.tensor(x)
        # Compute random projections
        bits = [torch.dot(v,x-u)>0 for u,v in random_planes]
        index = sum([b*(1<<i) for i,b in enumerate(bits)])
        # Place data in appropriate bin
        bins[index].append((x.tolist(),v))
    return bins

def hoeffding(vals,target,score_range):
    t = np.nanmean(vals) - target
    n = len(vals)
    d = score_range[1]-score_range[0]
    return np.exp(-2*n**2*t**2/d**2)

def sample_lsh(results_directory, agent_name='ActorCritic', n_planes=4, perturbance=0,
        scoring='mean', target_score=None, score_range=[0,500]):
    """ Split the data into a number of buckets through locality-sensitive
    hashing. Find the bucket with the best average score and sample parameters
    in the convex hull of parameters in that bucket.
    """
    scores = compute_score(results_directory)[agent_name]

    bins = bin_lsh(scores,space[agent_name], n_planes=n_planes)
    if scoring=='mean':
        score_bins = [np.nanmean([v['mean'] for k,v in b]) for b in bins]
        best_index = np.nanargmin(score_bins) # Lower score is better
    elif scoring=='improvement_prob':
        score_bins = [hoeffding([v['mean'] for k,v in b],target_score,score_range) for b in bins]
        best_index = np.nanargmax(score_bins) # Want highest probability of improvement
    param = random_lin_comb(np.array([k for k,v in bins[best_index]])).tolist()
    if perturbance > 0:
        param = hyperparams.utils.perturb_vec(param,space[agent_name],perturbance)
    param = hyperparams.utils.vec_to_param(param,space[agent_name])
    print('Performance:',np.nanmean([v['mean'] for k,v in bins[best_index]]))
    print('Number of points',len(bins[best_index]))
    if target_score is not None:
        print('Improvement Probability:',hoeffding([v['mean'] for k,v in bins[best_index]],target_score,score_range))
    return param

def run_bayes_opt(results_directory, agent_name='ActorCritic'):
    scores = compute_score(results_directory)[agent_name]
    x0 = []
    y0 = []
    count = 0
    for p,v in scores:
        if p['agent_name'] != agent_name:
            continue
        vec = hyperparams.utils.param_to_vec(p,space[agent_name])
        for a in vec:
            if np.abs(a) > 1.7320508075688772*100:
                count += 1
                break
        else:
            x0.append(vec)
            y0.append(v['mean'])
    print('Out of bounds:',count,len(x0))
    def func(vec):
        p = hyperparams.utils.vec_to_param(vec,space[agent_name])
        return run_trial(**p)
    res = gp_minimize(
            func,                  # the function to minimize
            hyperparams.utils.space_to_ranges(space[agent_name]),      # the bounds on each dimension of x
            acq_func="EI",      # the acquisition function
            n_calls=0,         # the number of evaluations of f
            n_random_starts=0,  # the number of random initialization points
            noise=25, # Variance of the function output
            x0=x0,y0=y0)
    vec = res.x
    val = res.fun
    print('Optimal parameters (score: %f):' % val)
    pprint.pprint(hyperparams.utils.vec_to_param(vec,space[agent_name]))

##################################################
# Plotting
##################################################

def plot(results_directory,plot_directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    data_y = defaultdict(lambda: [])
    data_x = {}
    for args,result in utils.get_all_results(results_directory):
        data_y[args['agent_name']].append(result['steps_to_reward'])
        data_x[args['agent_name']] = range(0,args['total_steps'],args['epoch'])
    for k,v in data_y.items():
        max_len = max([len(y) for y in v])
        data_y[k] = np.nanmean(np.array([y+[np.nan]*(max_len-len(y)) for y in v]),0)
        plt.plot(data_x[k],data_y[k],label=k)
    plt.xlabel('Training Steps')
    plt.ylabel('Steps to Reward')
    plt.legend(loc='best')
    plt.grid()
    plot_path = os.path.join(plot_directory,'plot.png')
    plt.savefig(plot_path)
    plt.close()
    print('Saved plot %s' % plot_path)

def plot(data, plot_directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    for k,v in data.items():
        plt.plot(v['x'],v['y'],label=k)
    plt.xlabel('Training Steps')
    plt.ylabel('Steps to Reward')
    plt.legend(loc='best')
    plt.grid(which='both')
    plot_path = os.path.join(plot_directory,'plot.png')
    plt.savefig(plot_path)
    plt.close()
    print('Saved plot %s' % plot_path)

def plot_single_param(results_directory, plot_directory, agent_name, param,
        log=False):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    scores = compute_score(results_directory)[agent_name]

    x = []
    y = []
    for k,v in scores:
        p = dict(k)
        if log:
            x.append(np.log(p[param]))
        else:
            x.append(p[param])
        y.append(v['mean'])
    plt.scatter(x,y)
    plt.xlabel(param)
    plt.ylabel('Score')
    #plt.legend(loc='best')
    plt.grid(which='both')
    plot_path = os.path.join(plot_directory,'plot.png')
    plt.savefig(plot_path)
    plt.close()
    print('Saved plot %s' % plot_path)

def plot_tsne(results_directory, plot_directory, agent_name):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import sklearn
    from sklearn.manifold import TSNE

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    scores = compute_score(results_directory)[agent_name]

    log_params = ['controller_learning_rate','subpolicy_learning_rate','q_net_learning_rate']
    ignore_params = ['agent_name','gamma','test_iters','verbose','directory']

    params = []
    values = []
    for p,s in scores:
        params.append(hyperparams.utils.param_to_vec(p,space[agent_name]))
        values.append(s['mean'])
    x_embedded = TSNE(n_components=2).fit_transform(params)
    plt.scatter([x for x,y in x_embedded], [y for x,y in x_embedded],c=values,s=10)
    plt.colorbar()
    plot_path = os.path.join(plot_directory,'tsne.png')
    plt.savefig(plot_path)
    plt.close()
    print('Saved plot %s' % plot_path)

def plot_tsne_smooth(results_directory, plot_directory, agent_name, n_planes=4):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    import sklearn
    from sklearn.manifold import TSNE

    if not os.path.isdir(plot_directory):
        os.makedirs(plot_directory)

    scores = smoothen_scores_lsh(results_directory,agent_name,n_planes=n_planes)

    params = []
    values = []
    for p,s in scores.items():
        params.append(p)
        #values.append(np.log(s)) # Log so we can better distinguish smaller values
        values.append(s) # Log so we can better distinguish smaller values
    x_embedded = TSNE(n_components=2).fit_transform(params)
    plt.scatter([x for x,y in x_embedded], [y for x,y in x_embedded],
            c=values, s=5)
    plt.colorbar()
    plot_path = os.path.join(plot_directory,'tsne_smooth.png')
    plt.savefig(plot_path,dpi=200)
    plt.close()
    print('Saved plot %s' % plot_path)

##################################################
# Results Parsing
##################################################

def flatten_params(params):
    p = params
    ap = p.pop('agent_params',{})
    if 'directory' in p:
        del p['directory']
    for k,v in ap.items():
        p[k] = v
    return dict(p.items())

def compute_score(directory,sortby='mean',
        keys=['mean']):
    results = utils.get_all_results(directory)
    scores = defaultdict(lambda: [])
    for param,values in results:
        try:
            param = flatten_params(param)
            d = values['steps_to_reward']
            if len(d) < 100:
                continue
            m = np.mean(d[50:])
            s = {
                    'data': d,
                    'mean': m
            }
            scores[param['agent_name']].append((param,s))
        except:
            pass # Invalid file
    sorted_scores = {}
    for an in scores.keys():
        sorted_scores[an] = sorted(scores[an],key=lambda x: x[1][sortby])
    return sorted_scores

def compute_series_all(directory):
    results = utils.get_all_results(directory)
    series = defaultdict(lambda: [])
    for k,v in results:
        s = v['steps_to_reward']
        if len(s) < 100:
            continue
        series[k['agent_name']].append(s)
    return series

def compute_series_euclidean(directory,agent_name='ActorCritic',params={},radius=0.1):
    params = hyperparams.utils.param_to_vec(
            params, space[agent_name])
    params = torch.tensor(params)
    results = utils.get_all_results(directory)
    series = []
    for k,v in results:
        if d['agent_name'] != agent_name:
            continue
        k = hyperparams.utils.param_to_vec(k, space[agent_name])
        k = torch.tensor(k)
        if ((k-param)**2).sum() > radius:
            continue
        series.append(v['steps_to_reward'])
    return series

def compute_series_lsh(directory, iterations=10, n_planes=4):
    scores = compute_score(directory)
    output = defaultdict(lambda: {'series': [], 'bin_size': []})
    for agent_name in scores.keys():
        for _ in tqdm(range(iterations),desc='Computing series (LSH)'):
            bins = bin_lsh(scores[agent_name], space[agent_name],
                    n_planes=n_planes)
            score_bins = [np.nanmean([v['mean'] for k,v in b]) for b in bins]
            min_index = np.nanargmin(score_bins)
            min_series = np.array([v['data'] for k,v in bins[min_index]]).mean(0)
            output[agent_name]['series'].append(min_series)
            output[agent_name]['bin_size'].append(len(bins[min_index]))
        output[agent_name]['series'] = np.array(output[agent_name]['series']).mean(0)
    return output

def smoothen_scores_lsh(results_directory, agent_name, iterations=10, n_planes=4):
    scores = compute_score(results_directory)[agent_name]

    output = defaultdict(lambda: [])
    for _ in range(iterations):
        bins = bin_lsh(scores,space[agent_name],n_planes=n_planes)
        for b in bins:
            if len(b) == 0:
                continue
            score = np.nanmean([v['mean'] for k,v in b])
            for k,_ in b:
                output[tuple(k)].append(score)
    for k,v in output.items():
        output[k] = np.mean(output[k])
    return output

def smoothen_series_lsh(results_directory, agent_name, iterations=10,n_planes=4):
    data = compute_series(results_directory)[agent_name]

    output = defaultdict(lambda: 0)
    for _ in range(iterations):
        bins = bin_lsh(scores,space[agent_name],n_planes=n_planes)
        for b in bins:
            if len(b) == 0:
                continue
            series = np.nanmean([v['mean'] for k,v in b])
            for k,_ in b:
                output[tuple(k)] += score
    for k,v in output.items():
        output[k] /= iterations
    return output

def fit_gaussian_process(directory, agent_name):
    results = utils.get_all_results(directory)
    scores = defaultdict(lambda: [])
    x = []
    y = []
    for param,values in results:
        try:
            if param['agent_name'] != agent_name:
                continue

            param = flatten_params(param)
            vec = hyperparams.utils.param_to_vec(param,space[agent_name])

            d = values['steps_to_reward']
            if len(d) < 100:
                continue
            m = np.mean(d[50:])

            x.append(vec)
            y.append(m)
        except:
            pass # Invalid file
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    kernel = RBF()
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x,y)
    breakpoint()
    return x,y

def aggregate_results(results_directory):
    pass

def compute_subpolicy_boundaries(agent, shape=[10,10], goal=None):
    def neighbours(p):
        yield p + torch.tensor([[0,1,0,0]]).float()
        yield p + torch.tensor([[0,-1,0,0]]).float()
        yield p + torch.tensor([[1,0,0,0]]).float()
        yield p + torch.tensor([[-1,0,0,0]]).float()
    augmented = isinstance(agent.controller_net, PolicyFunctionAugmentatedState)
    output = torch.zeros(shape*2)
    if goal is None:
        states = list(itertools.product(range(shape[0]),range(shape[1]),range(shape[0]),range(shape[1])))
    else:
        states = list(itertools.product(range(shape[0]),range(shape[1]),[goal[0]],[goal[1]]))
    for p in tqdm(states):
        if not augmented:
            p = torch.tensor(p).view(1,-1).float()
            a1 = torch.argmax(agent.controller_net(p))
            for n in neighbours(p):
                a2 = torch.argmax(agent.controller_net(n))
                if a1 != a2:
                    output[tuple(*p.long())] += 1
        else:
            for a in range(4):
                a = torch.tensor([[a]])
                p = torch.tensor(p).view(1,-1).float()
                a1 = torch.argmax(agent.controller_net(p,a))
                for n in neighbours(p):
                    a2 = torch.argmax(agent.controller_net(n,a))
                    if a1 != a2:
                        output[tuple(*p.long())] += 1
    return output

##################################################
# Execution
##################################################

def get_experiment_params(directory):
    params = {}
    params['ac-001'] = {
        'agent_name': 'ActorCritic',
        'gamma': 0.9,
        'controller_learning_rate': 0.001,
        'subpolicy_learning_rate': 0.001,
        'q_net_learning_rate': 0.001,
        'eps_b': 0.05,
        'polyak_rate': 0.001,
        'batch_size': 256,
        'min_replay_buffer_size': 1000,
        'steps_per_task': 10000,
        'total_steps': 100000+1,
        'epoch': 1000,
        'test_iters': 5,
        'verbose': True,
        'cnet_n_layers': 2,
        'cnet_layer_size': 3,
        'snet_n_layers': 2,
        'snet_layer_size': 3,
        'qnet_n_layers': 2,
        'qnet_layer_size': 20,
        'num_options': 5,
        'directory': directory
    }
    params['hrl_memoryless-001'] = {
            **params['ac-001'],
            'agent_name': 'HDQNAgentWithDelayAC_v2',
            'subpolicy_q_net_learning_rate': 1e-3
    }
    params['hrl_augmented-001'] = {
            **params['hrl_memoryless-001'],
            'agent_name': 'HDQNAgentWithDelayAC_v3',
            'subpolicy_q_net_learning_rate': 1e-3
    }

    """
    Looked at the decision-boundary plots fo AC, and it looks like it just uses a single subpolicy for everything.
    This means individual subpolicies have too much representational power, so we'll reduce it.
    """
    for alg in ['ac', 'hrl_memoryless', 'hrl_augmented']:
        params['%s-002'%alg] = {
                **params['%s-001'%alg],
                'snet_layer_size': 2,
        }

    # Params for debugging purposes
    params['debug'] = {
            **params['hrl_augmented-001'],
            'seed': 1,
            'epoch': 5,
            'test_iters': 5,
            'min_replay_buffer_size': 10,
            'batch_size': 5,
            'total_steps': 100
    }

    return params

def run():
    DEFAULT_DIRECTORY = os.path.join(utils.get_results_root_directory(),'hrl-4')
    #DEFAULT_DIRECTORY = os.path.join(utils.get_results_root_directory(),'dev')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=str, default=DEFAULT_DIRECTORY)
    subparsers = parser.add_subparsers(help='')
    
    parsers = {}
    commands = [
            ['plot',''],
            ['plot-lsh',''],
            ['plot2',''],
            ['tsne',''],
            ['checkpoint',''],
            ['random',''],
            ['run',''],
            ['controller-dropout',''],
            ['decision-boundary',''],
    ]
    for c,h in commands:
        parsers[c] = subparsers.add_parser(c, help=h)
        parsers[c].set_defaults(command=c)

    parsers['run'].add_argument('exp_name', type=str, choices=list(get_experiment_params(None).keys()))
    parsers['controller-dropout'].add_argument('initial_checkpoint', type=str)
    parsers['checkpoint'].add_argument('checkpoint_files', type=str, nargs='*')
    parsers['plot'].add_argument('directories', type=str, nargs='+')
    parsers['plot2'].add_argument('directory', type=str, default=None)
    parsers['random'].add_argument('exp_name', type=str, choices=list(space.keys()))
    parsers['decision-boundary'].add_argument('directory', type=str)
    parsers['decision-boundary'].add_argument('--clear-cache', dest='clear_cache', action='store_true')
    parsers['decision-boundary'].add_argument('--no-cache', dest='no_cache', action='store_true')

    args = parser.parse_args()

    # Set up environment with parsed args
    utils.set_results_directory(args.results_root)
    directory = os.path.join(utils.get_results_directory(),__name__)
    plot_directory = os.path.join(utils.get_results_directory(),'plots',__name__)
    for agent_name in space.keys():
        space[agent_name]['directory'] = directory
    experiment_params = get_experiment_params(directory)

    import sys
    print(sys.argv)
    if len(sys.argv) >= 2:
        if args.command == 'plot':
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt

            print(args.directories)
            for directory in args.directories:
                exp_name = os.path.split(os.path.normpath(directory))[-1]
                x = None
                y = []
                count = 0
                for r in utils.get_all_results(directory):
                    if len(r['steps_to_reward']) == 101:
                        x = r['eval_steps']
                        y.append(r['steps_to_reward'])
                        count += 1
                y = np.array(y).mean(axis=0)
                plt.plot(x,y,label='%s (%d)'%(exp_name,count))
            plt.legend(loc='best')
            plt.grid(which='both')
            if not os.path.isdir(plot_directory):
                os.makedirs(plot_directory)
            plot_path = os.path.join(plot_directory, 'plot.png')
            plt.savefig(plot_path)
            print('plot saved at', plot_path)
        elif args.command == 'plot-lsh':
            """
            Plot the time series of the best runs found by the LSH algorithm
            """
            series = compute_series_lsh(directory,iterations=100,n_planes=8)
            data = defaultdict(lambda: {})
            for an,s in series.items():
                #data['all']['x'] = range(0,p['total_steps'],p['epoch'])
                data[an]['x'] = range(0,100000,1000)
                data[an]['y'] = s['series']
                print(an,np.mean(s['series'][50:]),np.mean(s['bin_size']))
            plot(data,plot_directory)
        elif args.command == 'plot2':
            results_dir = directory
            # Old plotting function. Used to generate plot from 2020-01-07
            def map_func(params):
                p = params
                ap = p.pop('agent_params',{})
                if 'directory' in p:
                    del p['directory']
                for k,v in ap.items():
                    p[k] = v
                return frozenset(p.items())
            def compute_series(directory,params={}):
                def reduce(r,acc):
                    acc.append(r['steps_to_reward'])
                    return acc
                series = utils.get_all_results_map_reduce(
                        directory, map_func, reduce, lambda: [])
                for k,v in series.items():
                    max_len = max([len(x) for x in v])
                    series[k] = np.nanmean(np.array(list(itertools.zip_longest(*v,fillvalue=np.nan))),axis=1)
                return series
            def compute_score(directory,params={},sortby='mean',
                    keys=['mean','ucb1','count']):
                def reduce(r,acc):
                    if len(r['steps_to_reward']) == 100:
                        acc.append(np.mean(r['steps_to_reward'][50:]))
                    return acc
                scores = utils.get_all_results_map_reduce(
                        directory, map_func, reduce, lambda: [])
                n = defaultdict(lambda: 0)
                for k,v in scores.items():
                    an = dict(k)['agent_name']
                    n[an] += len([x for x in v if x==x])
                for k,v in scores.items():
                    if len(v) == 0:
                        scores[k] = {
                                #'data': [],
                                'mean': np.nan,
                                'ucb1': -np.inf,
                                'count': len(v)
                        }
                    else:
                        m = np.nanmean(v)
                        d = [x for x in v if x==x]
                        c = len(d)
                        t = n[dict(k)['agent_name']]
                        scores[k] = {
                                #'data': d,
                                'mean': m,
                                'ucb1': m-500*np.sqrt(2*np.log(t)/c),
                                'count': c
                        }
                sorted_scores = sorted(scores.items(),key=lambda x: x[1][sortby])
                return sorted_scores
            series = compute_series(results_dir)
            scores = compute_score(results_dir)
            scores_by_agent = defaultdict(lambda: [])
            for p,s in scores:
                p = dict(p)
                scores_by_agent[p['agent_name']].append((p,s))
            agent_names = scores_by_agent.keys()
            data = defaultdict(lambda: {'x': None, 'y': None})
            for an in agent_names:
                p = scores_by_agent[an][0][0]
                y = series[map_func(p)]
                s = scores_by_agent[an][0][1]
                print(s, an)
                print(p)
                label = '%s (%d)' % (an,scores_by_agent[an][0][1]['count'])
                data[label]['x'] = range(0,p['total_steps'],p['epoch'])
                data[label]['y'] = y
            plot(data,plot_directory)
        elif args.command == 'plot3':

            pass

        elif args.command == 'tsne':
            """
            Visualize distribution of performances with t-sne
            """
            plot_tsne(directory, plot_directory, 'ActorCritic')
            plot_tsne_smooth(directory, plot_directory, 'ActorCritic',n_planes=6)
            plot_tsne_smooth(directory, plot_directory, 'HDQNAgentWithDelayAC_v2',n_planes=6)
            plot_tsne_smooth(directory, plot_directory, 'HDQNAgentWithDelayAC_v3',n_planes=6)

        elif args.command == 'checkpoints' or args.command == 'checkpoint':
            """
            Start a run from a checkpoint
            """
            if len(args.checkpoint_files) > 0:
                fns = args.checkpoint_files
            else:
                fns = list_checkpoints()
            for fn in fns:
                print('Running from checkpoint',fn)
                exp = Experiment.from_checkpoint(fn)
                exp.run()

        elif args.command == 'random':
            """
            Run a trial with random parameters
            """
            run_hyperparam_search(space[args.exp_name])

        elif args.command == 'run':
            exp_name = args.exp_name

            params = experiment_params[exp_name]
            params['directory'] = os.path.join(directory,exp_name)
            run_trial_with_checkpoint(**params)

        elif args.command == 'controller-dropout':
            initial_checkpoint = args.initial_checkpoint
            params = {
                    'max_steps': 500,
                    'test_iters': 10,
                    'dropout_probability': 0,
                    'seed': 1,
                    'initial_checkpoint': initial_checkpoint,
                    'directory': directory
            }

            exp = ExperimentRandomControllerDropout(**params)
            exp.run()

        elif args.command == 'decision-boundary':
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            import shelve
            import uuid

            cache = not args.no_cache
            clear_cache = args.clear_cache
            map_shape = [13,13]

            # Directories/Paths
            results_dir = os.path.normpath(args.directory) # Directory whose checkpoints are to be processed
            output_dir = os.path.join(args.results_root, args.command)
            processed_results_mapping = os.path.join(output_dir, 'mapping.pkl')

            # Ensure all directories exist
            if not os.path.isdir(plot_directory):
                os.makedirs(plot_directory)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            # Get path of the decision boundary data for the provided result directory
            with shelve.open(processed_results_mapping) as mapping:
                if results_dir in mapping:
                    results_path = mapping[results_dir]
                else:
                    results_path = '%s.pkl' % uuid.uuid1()
                    mapping[results_dir] = results_path

            # Load precomputed decision boundary data
            def update_boundaries(output, results_dir):
                for checkpoint_path in tqdm(list(utils.get_all_result_paths(results_dir))):
                    if checkpoint_path in output:
                        continue
                    try:
                        tqdm.write(checkpoint_path)
                        exp = Experiment.from_checkpoint(checkpoint_path)
                        output[checkpoint_path] = compute_subpolicy_boundaries(exp.agent, map_shape)
                    except KeyboardInterrupt:
                        break
                    except:
                        pass
                return output
            boundaries_by_goal = {}
            if cache:
                with shelve.open(results_path) as db_results:
                    if clear_cache:
                        db_results.clear()
                    update_boundaries(db_results, results_dir)
                    mean_boundaries = torch.stack(
                            [x.sum(dim=3).sum(dim=2)/x.sum() for x in db_results.values() if x.sum() > 0]
                    ).mean(dim=0).numpy()
                    for x,y in itertools.product(range(map_shape[0]), range(map_shape[1])):
                        boundaries_by_goal[(x,y)] = torch.stack(
                                [v[:,:,x,y]/v[:,:,x,y].sum() for v in db_results.values() if v[:,:,x,y].sum() > 0]
                        ).mean(dim=0).numpy()
            else:
                db_results = update_boundaries({}, results_dir)
                mean_boundaries = torch.stack([x for x in db_results.values() if x is not None]).mean(dim=0).numpy()
                for x,y in itertools.product(range(map_shape[0]), range(map_shape[1])):
                    boundaries_by_goal[(x,y)] = torch.stack(
                            [v[:,:,x,y]/v[:,:,x,y].sum() for v in db_results.values() if v[:,:,x,y].sum() > 0]
                    ).mean(dim=0).numpy()

            exp_name = os.path.split(os.path.normpath(results_dir))[-1]

            plt.imshow(mean_boundaries)
            plt.colorbar()
            plt.title(exp_name)
            plot_path = os.path.join(plot_directory,'decision-boundaries-%s.png'%exp_name)
            plt.savefig(plot_path)
            plt.close()
            print('Figure saved to %s' % plot_path)

            fig, axes = plt.subplots(*map_shape)
            for x,y in itertools.product(range(map_shape[0]), range(map_shape[1])):
                axes[y,x].imshow(boundaries_by_goal[(x,y)])
            plot_path = os.path.join(plot_directory,'decision-boundaries-g-%s.png'%exp_name)
            plt.savefig(plot_path)
            plt.close()
            print('Figure saved to %s' % plot_path)

        else:
            print('Invalid command')
    else:
        utils.set_results_directory(
                os.path.join(utils.get_results_root_directory(),'dev'))

        checkpoint_path = '/network/tmp1/huanghow/hrl-4/checkpoints/524717.checkpoint.pkl'
        checkpoint_path = '/network/tmp1/huanghow/hrl-4/experiments.hrl.transfer_nrooms/ac-001/ActorCritic-0.pkl'
        exp = Experiment.from_checkpoint(checkpoint_path)

        boundaries = compute_subpolicy_boundaries(exp.agent, [13,13])
        print(boundaries)

        #run_hyperparam_search(space['ActorCritic'])
        #run_hyperparam_search(space['HDQNAgentWithDelayAC_v2'])
        #run_hyperparam_search_extremes(space['HDQNAgentWithDelayAC_v2'])
        #run_hyperparam_search(space['HDQNAgentWithDelayAC_v3'])
        #run_hyperparam_search_extremes(space['HDQNAgentWithDelayAC_v2'])

        #param = sample_convex_hull(directory)
        #param = sample_lsh(directory, 'HDQNAgentWithDelayAC_v2', n_planes=8, perturbance=0.0)
        #param = sample_lsh(directory, 'HDQNAgentWithDelayAC_v2', n_planes=8, perturbance=0.05, scoring='improvement_prob', target_score=182.58163722924036)
        #param = sample_lsh(directory, 'ActorCritic', n_planes=8, perturbance=0.01)
        #param = hyperparams.utils.sample_hyperparam(space['HDQNAgentWithDelayAC_v2'])
        #run_trial_with_checkpoint(**param)

        #run_bayes_opt(directory,'ActorCritic')
        #run_bayes_opt(directory,'HDQNAgentWithDelayAC_v2')

        #s = smoothen_scores_lsh(directory, 'HDQNAgentWithDelayAC_v2')
        #pprint.pprint(sorted(s.values()))

        #count = 0
        #for v,save in utils.modify_all_results(directory):
        #    if v is None or 'steps_to_reward' not in v[1] or len(v[1]['steps_to_reward']) < 100:
        #        #save(None)
        #        count += 1
        #print(count)
