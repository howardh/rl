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

from skopt.learning.gaussian_process.kernels import Matern

from rl import utils
from rl.agent.hrl_agent_v5 import HRLAgent_v5
from rl.agent.bsuite_agent import A2CAgent, DQNAgent

import experiment
import experiment.plotter
from experiment import Experiment, ExperimentRunner, load_checkpoint
from experiment.logger import Logger
from experiment.hyperparam import Uniform, IntUniform, LogUniform, LogIntUniform, GridSearch, RandomSearch, BayesianOptimizationSearch

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

class BaseExperiment(Experiment):
    def setup(self,config,output_directory=None):
        self.config = config
        self.plot_directory = output_directory
        if self.plot_directory is not None:
            os.makedirs(self.plot_directory, exist_ok=True)

        self.env, self.test_env = init_train_test_env(
                goal_repeat_allowed=config.get('goal_repeat_allowed'),
                num_training_tasks=config.get('num_training_tasks'),
                split_train_test=config.get('split_train_test'),
                random_key=None # TODO
        )

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
            'config': self.config,
            'plot_directory': self.plot_directory,
            'logger': self.logger.state_dict(),
            'done': self.done,
            'env': self.env.state_dict(),
            'test_env': self.test_env.state_dict(),
            'agent': self.agent.state_dict(),
        }
    def load_state_dict(self,state):
        self.setup(state['config'], state['plot_directory'])
        self.logger.load_state_dict(state.get('logger'))
        self.done = state.get('done')
        self.env.load_state_dict(state.get('env'))
        self.test_env.load_state_dict(state.get('test_env'))
        self.agent.load_state_dict(state.get('agent'))

class HRLExperiment(BaseExperiment):
    def setup(self,config,output_directory=None):
        super().setup(config, output_directory)

        self.agent = HRLAgent_v5(
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            discount_factor=config.get('discount_factor'),
            actor_lr=config.get('actor_learning_rate'),
            critic_lr=config.get('critic_learning_rate'),
            polyak_rate=config.get('polyak_rate'),
            actor=None,
            critic=None
        )

class DQNExperiment(BaseExperiment):
    def setup(self,config,output_directory=None):
        super().setup(config, output_directory)
        self.agent = DQNAgent(
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            discount_factor=config['discount_factor'],
            learning_rate=config['learning_rate'],
            #polyak_rate=config.get('polyak_rate'),
            #actor_model=actor_network,
            #critic_model=critic_network,
            rng=hk.PRNGSequence(0)
        )

class A2CExperiment(BaseExperiment):
    def setup(self,config,output_directory=None):
        super().setup(config, output_directory)
        self.agent = A2CAgent(
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            discount_factor=config.get('discount_factor'),
            learning_rate=config.get('learning_rate'),
            td_lambda=config.get('td_lambda'),
            num_layers=config.get('num_layers'),
            layer_size=config.get('layer_size'),
            rng=hk.PRNGSequence(0)
        )

def run():
    import typer
    app = typer.Typer()

    HRL_DEFAULT_CONFIG = {
            'goal_repeat_allowed': True,
            'num_training_tasks': 10,
            'split_train_test': False,
            'discount_factor': 0.99,
            'learning_rate': 1e-4,
            'test_iters': 5,
            'test_max_steps': 500,
    }
    results_root_dir = utils.get_results_root_directory()
    #bo_kernel = Matern(length_scale=10,nu=2.5,length_scale_bounds='fixed')
    bo_kernel = Matern()

    def get_exp_params():
        params = {}
        params['dqn-001'] = {
            'cls': DQNExperiment,
            'search_space': {
                'goal_repeat_allowed': True,
                'num_training_tasks': 10,
                'split_train_test': False,
                'discount_factor': 0.99,
                'learning_rate': LogUniform(1e-5,1e-1,3),
                'min_replay_size': 1000,
                'epsilon': Uniform(0,0.5),
                'batch_size': 32,
                'replay_capacity': 10000,
                'network_structure': [64,64],
                'test_iters': 5,
                'test_max_steps': 500,
            },
        }
        params['a2c-001'] = {
            'cls': A2CExperiment,
            'search_space': {
                'goal_repeat_allowed': True,
                'num_training_tasks': 10,
                'split_train_test': False,
                'discount_factor': 0.99,
                'learning_rate': LogUniform(1e-5,1e-1,3),
                'td_lambda': Uniform(0,1),
                #'network_structure': [64,64],
                'num_layers': IntUniform(1,3),
                'layer_size': LogIntUniform(1,128),
                'test_iters': 5,
                'test_max_steps': 500,
            },
        }
        params['hrl-001'] = {
            'cls': HRLExperiment,
            'search_space': {
                'goal_repeat_allowed': True,
                'num_training_tasks': 10,
                'split_train_test': False,
                'discount_factor': 0.99,
                'actor_learning_rate': LogUniform(1e-5,1e-1,3),
                'critic_learning_rate': LogUniform(1e-5,1e-1,3),
                'polyak_rate': LogUniform(1e-3,1),
                #'network_structure': [64,64],
                'actor_num_layers': IntUniform(1,3),
                'actor_layer_size': LogIntUniform(1,128),
                'critic_num_layers': IntUniform(1,3),
                'critic_layer_size': LogIntUniform(1,128),
                'test_iters': 5,
                'test_max_steps': 500,
            },
        }
        return params

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

    @app.command()
    def gridsearch(exp_name, output_directory:str = None, debug:bool = typer.Option(False,"--debug")):
        params = get_exp_params()[exp_name]

        if debug:
            search = GridSearch(params['cls'],
                    name='Debug-GridSearch-%s'%exp_name,
                    search_space={
                        **params['search_space'],
                        'test_iters': 5,
                        'test_max_steps': 50,
                    },
                    output_directory=output_directory,
                    epoch=10,
                    checkpoint_frequency=100,
                    max_iterations=50,
                    verbose=True,
            )
        else:
            search = GridSearch(params['cls'],
                    name='GridSearch-%s'%exp_name,
                    search_space=params['search_space'],
                    output_directory=output_directory,
                    epoch=1000,
                    checkpoint_frequency=10000,
                    max_iterations=100000,
                    verbose=True,
            )

        if output_directory is None:
            search.run()
        else:
            #search.load_checkpoint
            pass # Load gridsearch results

        print('Search complete. To obtain the results, run\n\n\tpython main.py hyperparam-search-results %s'%os.path.join(search.directory,'Experiments'))

    @app.command()
    def random_search(exp_name, output_directory:str = None, debug:bool = typer.Option(False,"--debug")):
        params = get_exp_params()[exp_name]

        if debug:
            search = RandomSearch(params['cls'],
                    name='Debug-RandomSearch-%s'%exp_name,
                    search_space={
                        **params['search_space'],
                        'test_iters': 5,
                        'test_max_steps': 50,
                    },
                    output_directory=output_directory,
                    epoch=10,
                    checkpoint_frequency=100,
                    max_iterations=50,
                    verbose=True,
            )
        else:
            search = RandomSearch(params['cls'],
                    name='RandomSearch-%s'%exp_name,
                    search_space=params['search_space'],
                    output_directory=output_directory,
                    epoch=1000,
                    checkpoint_frequency=10000,
                    max_iterations=100000,
                    verbose=True,
            )

        if output_directory is None:
            search.run()
        else:
            #search.load_checkpoint
            pass

        print('Search complete. To obtain the results, run\n\n\tpython main.py hyperparam-search-results %s'%os.path.join(search.directory,'Experiments'))

    @app.command()
    def bo_search(exp_name, output_directory:str = None, debug:bool = typer.Option(False,"--debug"), budget:int = 3):
        params = get_exp_params()[exp_name]

        if debug:
            search = BayesianOptimizationSearch(params['cls'],
                    name='Debug-BOSearch-%s'%exp_name,
                    search_space={
                        **params['search_space'],
                        'test_iters': 5,
                        'test_max_steps': 50,
                    },
                    score_fn=lambda exp: exp.logger.mean('steps_to_reward'),
                    kernel=bo_kernel,
                    root_directory=results_root_dir,
                    output_directory=output_directory,
                    epoch=10,
                    checkpoint_frequency=100,
                    max_iterations=50,
                    verbose=True,
                    search_budget=budget
            )
        else:
            search = BayesianOptimizationSearch(params['cls'],
                    name='BOSearch-%s'%exp_name,
                    search_space=params['search_space'],
                    score_fn=lambda exp: exp.logger.mean('steps_to_reward'),
                    kernel=bo_kernel,
                    root_directory=results_root_dir,
                    output_directory=output_directory,
                    epoch=1000,
                    checkpoint_frequency=10000,
                    max_iterations=100000,
                    verbose=True,
                    search_budget=budget
            )

        search.run()
        search.plot_gp()

        print('-'*50)
        print('Search complete.')
        cmd = 'python main.py hyperparam-search-results {path}'.format(path=os.path.join(search.directory,'Experiments'))
        print('To obtain the results, run\n\t%s'%cmd)
        cmd = 'python main.py bo-search {exp_name} --output-directory {output_dir}{debug}'.format(output_dir=search.directory, exp_name=exp_name, debug=' --debug' if debug else '')
        print('To run additional search, run\n\t%s'%cmd)

    @app.command()
    def hyperparam_search_results(directory:str, exp_name:str):
        """ Return the best configuration and score for a hyperparameter search whose experiment results are stored in the provided directory. """
        import dill
        from pprint import pprint
        from experiment.hyperparam.search import SimpleAnalysis, GroupedAnalysis, GaussianProcessAnalysis

        params = get_exp_params()[exp_name]

        #analysis = SimpleAnalysis(A2CExperiment, directory=directory,
        #        score_fn=lambda exp: exp.logger.mean('steps_to_reward'))
        #analysis = GroupedAnalysis(A2CExperiment, directory=directory,
        #        score_fn=lambda exp: exp.logger.mean('steps_to_reward'))
        analysis = GaussianProcessAnalysis(params['cls'],
                kernel=bo_kernel,
                directory=directory,
                score_fn=lambda exp: exp.logger.mean('steps_to_reward'),
                search_space=params['search_space'])
        print(analysis.get_best_score())
        pprint(analysis.get_best_config())
        analysis.plot()

        #results = []
        #for exp_dir in os.listdir(directory):
        #    # Load results
        #    checkpoint_filename = os.path.join(directory, exp_dir, 'checkpoint.pkl')
        #    with open(checkpoint_filename,'rb') as f:
        #        checkpoint = dill.load(f)
        #    # Extract config and logs
        #    logger = Logger()
        #    logger.load_state_dict(checkpoint['exp']['logger'])
        #    config = checkpoint['args']['config']
        #    # Save data
        #    results.append((config,logger.mean('steps_to_reward')))
        ## Sort by score
        #sorted_results = sorted(results, key=lambda x: x[1])
        ## Output best result
        #best_config, best_score = sorted_results[0]
        #pprint(best_config)
        #print('best score:',best_score)

    app()
