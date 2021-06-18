import os
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import dill
import gym.spaces
import gym
import gym.envs

from experiment import Experiment, make_experiment_runner
from rl.agent.smdp import hrl
from rl.experiments.hrl2 import disjoint

class DropoutExperiment(Experiment):
    def setup(self, config, output_directory):
        self.output_directory = output_directory
        self.agent_deploy_state_filename = config['agent']
        with open(self.agent_deploy_state_filename,'rb') as f:
            state = dill.load(f)
        self.agent = hrl.make_agent_from_deploy_state(state)
        self.agent_deploy_state_filename = config['agent']

        self.dropout_probs = [0.,.2,.4,.6,.8,1.]

        self.env = self._init_env()

        self.results = defaultdict(lambda: [])
    def _init_env(self):
        available_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
        if 'Hopper-v1' in available_envs:
            env_name = 'Hopper-v1'
        else:
            env_name = 'Hopper-v2'
        env = disjoint.make_env(env_name,1_000)
        return env
    def run_step(self, iteration):
        agent = self.agent
        env = self.env
        done = False
        total_reward = 0

        agent.dropout_prob = self.dropout_probs[iteration%len(self.dropout_probs)]

        obs = env.reset()
        agent.observe(obs, testing=True)
        while not done:
            obs, reward, done, _ = env.step(agent.act(testing=True))
            agent.observe(obs, reward, done, testing=True)
            total_reward += reward
        self.results[agent.dropout_prob].append({
            'total_reward': total_reward
        })
    def plot(self):
        labels = sorted(self.results.keys())
        values = [[v['total_reward'] for v in self.results[l]] for l in labels]
        plt.violinplot(
                values,
                vert = False,
                showmeans = True,
        )
        plt.yticks(ticks=list(range(1,len(labels)+1)),labels=labels)
        plt.ylabel('Dropout Probability')
        plt.xlabel('Total Reward')
        plt.title('Dropout on Hopper-v3')
        plt.grid()

        plot_filename = os.path.join(self.output_directory, 'plot.png')
        plt.savefig(plot_filename)
        print('Plot saved at %s' % plot_filename)
    def state_dict(self):
        return {
                'results': self.results
        }
    def load_state_dict(self, state):
        self.results = state['results']

def make_app():
    import typer
    app = typer.Typer()

    @app.command()
    def run(filename : str,
            results_directory : Optional[str] = None,
            debug : bool = typer.Option(False, '--debug')):

        exp_name = 'dropout'

        if debug:
            pass
        exp = make_experiment_runner(
                DropoutExperiment,
                experiment_name=exp_name,
                verbose=True,
                checkpoint_frequency=5,
                max_iterations=30*5,
                results_directory=results_directory,
                config={
                    'agent': filename
                })
        exp.run()
        exp.exp.plot()

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
