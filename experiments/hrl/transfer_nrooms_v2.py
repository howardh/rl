import os
import gym
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from agent.hrl_agent_v5 import HRLAgent_v5
from agent.bsuite_agent import A2CAgent, DQNAgent

import experiment
import experiment.plotter
from experiment import Experiment, ExperimentRunner, load_checkpoint
from experiment.logger import Logger

ENV_MAP = """
xxxxxxxxxxxxxxxxx
x               x
x               x
x               x
xxxxxxxxxxxxxxxxx
"""

def init_train_test_env(goal_repeat_allowed, num_training_tasks, split_train_test, random_key):
    env_name='gym_fourrooms:fourrooms-v0'

    env = gym.make(
            env_name,
            fail_prob=0,
            goal_duration_episodes=1,
            goal_repeat_allowed=goal_repeat_allowed,
            #env_map=ENV_MAP
    ).unwrapped
    test_env = gym.make(
            env_name,
            fail_prob=0,
            goal_duration_episodes=1,
            goal_repeat_allowed=goal_repeat_allowed,
            #env_map=ENV_MAP
    ).unwrapped

    available_tasks = env.coords

    num_training_tasks = min(num_training_tasks, len(available_tasks))
    num_testing_tasks = len(available_tasks)-num_training_tasks

    training_task_indices = np.random.choice(
            len(available_tasks),
            num_training_tasks,
            replace=False
    ) # TODO: make deterministic as a function of seed or random key
    training_tasks = [available_tasks[i] for i in training_task_indices]

    if split_train_test:
        testing_tasks = [c for i,c in enumerate(available_tasks) if i not in training_task_indices]
    else:
        testing_tasks = training_tasks

    env.available_goals = training_tasks
    test_env.available_goals = testing_tasks

    return env, test_env

def actor_network(observation):
    seq = hk.Sequential([
        hk.Linear(4), jax.nn.relu,
        hk.Linear(15), jax.nn.relu,
        hk.Linear(15), jax.nn.relu,
        hk.Linear(4)
    ])
    logits = seq(observation)
    return logits

def critic_network(observation):
    seq = hk.Sequential([
        hk.Linear(4), jax.nn.relu,
        hk.Linear(15), jax.nn.relu,
        hk.Linear(15), jax.nn.relu,
        hk.Linear(4)
    ])
    values = seq(observation)
    return values

class HRLExperiment(Experiment):
    def setup(self,config):
        self.config = config
        self.plot_directory = config.get('plot_directory')
        if self.plot_directory is not None:
            os.makedirs(self.plot_directory, exist_ok=True)

        self.env, self.test_env = init_train_test_env(
                goal_repeat_allowed=config.get('goal_repeat_allowed'),
                num_training_tasks=config.get('num_training_tasks'),
                split_train_test=config.get('split_train_test'),
                random_key=None # TODO
        )
        #self.agent = HRLAgent_v5(
        #    action_space=self.env.action_space,
        #    observation_space=self.env.observation_space,
        #    discount_factor=config.get('discount_factor'),
        #    actor_lr=config.get('actor_learning_rate'),
        #    critic_lr=config.get('critic_learning_rate'),
        #    polyak_rate=config.get('polyak_rate'),
        #    actor=None,
        #    critic=None
        #)
        self.agent = A2CAgent(
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            discount_factor=config.get('discount_factor'),
            learning_rate=config.get('learning_rate'),
            rng=hk.PRNGSequence(0)
        )
        #self.agent = DQNAgent(
        #    action_space=self.env.action_space,
        #    observation_space=self.env.observation_space,
        #    discount_factor=config.get('discount_factor'),
        #    learning_rate=config.get('actor_learning_rate'),
        #    #polyak_rate=config.get('polyak_rate'),
        #    #actor_model=actor_network,
        #    #critic_model=critic_network,
        #    rng=hk.PRNGSequence(0)
        #)

        self.done = True
        self.logger = Logger()

    def run_step(self,iteration):
        # Run step
        if self.done:
            self.obs = self.env.reset()
            self.agent.observe_change(self.obs, None)
        self.obs, self.reward, self.done, _ = self.env.step(self.agent.act())
        self.agent.observe_change(self.obs, self.reward, terminal=self.done)
        # Update weights
        self.agent.train()
    def before_epoch(self,iteration): # Test agent
        if self.config.get('split_train_test') == False:
            self.test_env.reset_goal(self.env.goal)
        test_results = self.agent.test(
                self.test_env, self.config.get('test_iters'),
                max_steps=self.config.get('test_max_steps'),
                render=False, processors=1)

        # Log results
        self.logger.log(
                iteration=iteration,
                rewards=np.mean([r['total_rewards'] for r in test_results]),
                #testing_state_action_values=np.mean([r['state_action_values'] for r in test_results]),
                steps_to_reward=np.mean([r['steps'] for r in test_results]),
                #training_state_values_1=np.mean(self.agent.state_values_1),
                #training_state_values_2=np.mean(self.agent.state_values_2),
                **{k: np.mean(v) for k,v in self.agent.debug.items()}
        )
        self.agent.debug.clear()

        # Plot
        if self.plot_directory is not None:
            filename = os.path.join(self.plot_directory,'plot.png')
            curves = [
                {
                    'key': 'rewards',
                    'label': 'raw',
                },{
                    'key': 'rewards',
                    'smooth_fn': experiment.plotter.ComposeTransforms(
                        experiment.plotter.EMASmoothing(0.1),
                    ),
                    'label': 'smoothed',
                },{
                    'key': 'rewards',
                    'smooth_fn': experiment.plotter.ComposeTransforms(
                        experiment.plotter.LinearInterpResample(300),
                        experiment.plotter.EMASmoothing(0.01),
                    ),
                    'label': 'resampled+smoothed',
                }
            ]
            experiment.plotter.plot(self.logger, curves, filename=filename, min_points=3)

        # Display progress
        tqdm.write('steps %d \t Reward: %f \t Steps: %f' % (
            iteration, self.logger[-1]['rewards'], self.logger[-1]['steps_to_reward']
        ))
    def state_dict(self):
        return {
            'logger': self.logger.state_dict(),
            'done': self.done,
            'env': self.env.state_dict(),
            'test_env': self.test_env.state_dict(),
            'agent': self.agent.state_dict(),
        }
    def load_state_dict(self,state):
        self.logger.load_state_dict(state.get('logger'))
        self.done = state.get('done')
        self.env.load_state_dict(state.get('env'))
        self.test_env.load_state_dict(state.get('test_env'))
        self.agent.load_state_dict(state.get('agent'))

def run():
    import typer
    app = typer.Typer()

    DEFAULT_CONFIG = {
            'goal_repeat_allowed': True,
            'num_training_tasks': 10,
            'split_train_test': False,
            'discount_factor': 0.99,
            'learning_rate': 1e-4,
            'test_iters': 5,
            'test_max_steps': 500,
            'plot_directory': './plots',
    }

    @app.command()
    def run():
        config = {
                **DEFAULT_CONFIG
        }
        exp = ExperimentRunner(HRLExperiment,config=config,
                epoch=10,
                checkpoint_frequency=50,
                verbose=True,
        )
        exp.run()

    @app.command()
    def checkpoint(filename):
        exp = load_checkpoint(HRLExperiment, filename)
        exp.run()

    app()
