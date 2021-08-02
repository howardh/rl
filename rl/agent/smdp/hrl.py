import os
from typing import Any, DefaultDict, TypeVar, Sequence, Union, NamedTuple, Mapping, Optional
from copy import copy

import torch
import dill
import gym.spaces
import numpy as np
from collections import defaultdict

from experiment.logger import Logger
from rl.agent.agent import Agent, DeployableAgent
from rl.agent.augmented_obs_stack import AugmentedObservationStack
import rl.agent.smdp.sac as sac
import rl.agent.smdp.dqn as dqn

ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')
class StateDictType(NamedTuple):
    obs_stack: Mapping[Any, Mapping]

class HRLAgent(Agent[np.ndarray,ActionType]):
    def __init__(self,
            action_space : gym.spaces.Space,
            observation_space : gym.spaces.Space,
            agent : DeployableAgent[np.ndarray, int],
            children : Sequence[DeployableAgent[np.ndarray,ActionType]],
            children_discount : Union[Sequence[float],float],
            delay : int = 0):
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_children = len(children)

        self.obs_stack = defaultdict(
                lambda: AugmentedObservationStack(stack_len=self.delay+1,action_len=0))
        self.children_rewards = defaultdict(
                lambda: [[] for _ in children])

        self._steps = 0

        self.agent = agent
        self.children = children
        if type(children_discount) is float:
            self.children_discount = [children_discount] * self.num_children
        elif isinstance(children_discount, Sequence):
            self.children_discount = children_discount
        else:
            raise TypeError('Expected children_discount to be a list or a float. Received %s.' % type(children_discount))
        self.delay = delay

        # Dropout
        self.dropout_prob = 0.
        self.last_parent_action : DefaultDict[Any, Optional[int]] = defaultdict(lambda: None)

        # Logs
        self.logger = Logger(key_name='step')

    def observe(self, obs, reward=None, terminal=False, testing=False, time=1, discount=None, env_key=None):
        if env_key is None:
            env_key = testing
        if not testing:
            self._steps += 1

        # Logging
        if testing and reward is not None:
            self.logger.append(step=self._steps, reward=reward, terminal=terminal)

        # Give the parent agent a delayed observation (memoryless)
        self.obs_stack[env_key].append_obs(obs,reward)

        delayed_obs, delayed_reward = self._get_delayed_obs_reward(env_key)
        if delayed_obs is not None:
            self.agent.observe(
                    obs=delayed_obs,
                    reward=delayed_reward,
                    terminal=terminal if self.delay == 0 else False,
                    testing=testing,
                    time=time,
                    discount=discount,
                    env_key=env_key)

        # Save relevant observation data for children
        for cr in self.children_rewards[env_key]:
            cr.append(reward)

        # If terminal, then iterate through the remaining observations and pass them to the agents.
        if terminal:
            # Children receive the last observation with all the rewards it didn't receive yet.
            for child_index in range(len(self.children)):
                discount = self.children_discount[child_index]
                child_obs = self.obs_stack[env_key].get(0,0)
                if child_obs is None:
                    raise Exception('This should never happen.')
                self.children[child_index].observe(
                        obs=child_obs,
                        reward=self._compute_child_reward(
                            child_index=child_index,env_key=env_key),
                        terminal=True,
                        testing=testing,
                        time=len(self.children_rewards[env_key][child_index]),
                        discount=discount,
                        env_key=env_key)
            # Parent agent receives all observations that it did not yet see because of the delay
            for i in range(self.delay-1,-1,-1):
                action = self.agent.act(testing=testing,env_key=env_key)
                self.obs_stack[env_key].append_action(action)
                self.obs_stack[env_key].append_obs(np.empty([0]))
                delayed_obs, delayed_reward = self._get_delayed_obs_reward(env_key)
                if delayed_obs is not None:
                    self.agent.observe(
                            obs=delayed_obs,
                            reward=delayed_reward,
                            terminal = i!=0,
                            testing=testing,
                            time=time, # TODO: ??? Is this right?
                            discount=discount,
                            env_key=env_key)
            # Reset obs stack and rewards in preparation for the next episode
            self.obs_stack[env_key].clear()
            for cr in self.children_rewards[env_key]:
                cr.clear()
            # Reset dropout variables
            self.last_parent_action[env_key] = None
    def _get_delayed_obs_reward(self, env_key):
        delayed_obs = self.obs_stack[env_key].get(self.delay,0)
        delayed_reward = self.obs_stack[env_key].get_reward(self.delay,0)
        return delayed_obs, delayed_reward

    def act(self, testing=False, env_key=None) -> ActionType:
        if env_key is None:
            env_key = testing

        parent_action = self.last_parent_action[env_key]
        if np.random.rand() < self.dropout_prob or parent_action is None:
            if self._parent_can_act(env_key):
                parent_action = self.agent.act(testing=testing, env_key=env_key)
            else:
                parent_action = np.random.choice(self.num_children)
        self.last_parent_action[env_key] = parent_action

        active_child = self.children[parent_action]

        # Send observation to active child
        discount = self.children_discount[parent_action]
        obs = self.obs_stack[env_key].get(0,0)
        if obs is None:
            raise Exception('This should never happen')
        active_child.observe(obs=obs,
                reward=self._compute_child_reward(
                    child_index=parent_action,env_key=env_key),
                terminal=False,
                testing=testing,
                time=len(self.children_rewards[env_key][parent_action]),
                discount=discount,
                env_key=env_key)

        # Reset rewards for the active child
        self.children_rewards[env_key][parent_action].clear()

        # Get action from active child
        action = active_child.act(testing=testing, env_key=env_key)

        # Save action
        self.obs_stack[env_key].append_action(action)

        # Logging
        if testing:
            self.logger.append(step=self._steps, parent_action_testing=parent_action)
        else:
            self.logger.log(step=self._steps, parent_action_training=parent_action)

        return action
    def _parent_can_act(self, env_key) -> bool:
        """ Return True if the parent has enough observations to act, otherwise return False. """
        delayed_obs, _ = self._get_delayed_obs_reward(env_key)
        return delayed_obs is not None

    def _compute_child_reward(self, child_index, env_key):
        discount = self.children_discount[child_index]
        child_rewards = self.children_rewards[env_key][child_index]
        if child_rewards[0] is None: # First observation since the episode started.
            return None # The agent hasn't done anything before this point, so we just treat this observation as the starting state with no reward.
        return (np.array(child_rewards)*(discount**np.arange(len(child_rewards)))).sum()

    def state_dict(self):
        return {
                'obs_stack': {k: v.state_dict() for k,v in self.obs_stack.items()},
                'children_rewards': self.children_rewards,
                'steps': self._steps,
                'agent': self.agent.state_dict(),
                'children': [c.state_dict() for c in self.children],
                'delay': self.delay,
                'logger': self.logger.state_dict(),
        }
    def load_state_dict(self, state):
        for k,v in state['obs_stack'].items():
            self.obs_stack[k].load_state_dict(v)
        for k,v in state['children_rewards'].items():
            self.children_rewards[k] = v
        self._steps = state['steps']
        self.agent.load_state_dict(state['agent'])
        for child,child_state in zip(self.children,state['children']):
            child.load_state_dict(child_state)
        self.delay = state['delay']
        self.logger.load_state_dict(state['logger'])

    def state_dict_deploy(self):
        return {
                'action_space': self.action_space,
                'observation_space': self.observation_space,
                'agent': self.agent.state_dict_deploy(),
                'children': [c.state_dict_deploy() for c in self.children],
                'delay': self.delay,
        }
    def load_state_dict_deploy(self, state):
        self.agent.load_state_dict_deploy(state['agent'])
        for child,child_state in zip(self.children,state['children']):
            child.load_state_dict_deploy(child_state)
        self.delay = state['delay']

def make_agent(
        action_space : gym.spaces.Box,
        observation_space : gym.spaces.Box,
        parent_params,
        children_params,
        device):
    # Children
    children = []
    for child_param in children_params:
        child_param = copy(child_param)
        pi_net  = sac.PolicyNetwork(
                observation_space.shape[0],
                action_space.shape[0],
                structure = child_param.pop('pi_net_structure')
        ).to(device)
        sac_agent = sac.SACAgent(
            action_space=action_space,
            observation_space=observation_space,
            reward_scale=5,
            **child_param,
            q_net_1 = sac.QNetwork(observation_space.shape[0], action_space.shape[0]).to(device),
            q_net_2 = sac.QNetwork(observation_space.shape[0], action_space.shape[0]).to(device),
            v_net   = sac.VNetwork(observation_space.shape[0]).to(device),
            pi_net  = pi_net,
            device=device,
        )
        children.append(sac_agent)
    # Parent
    parent_params = copy(parent_params)
    q_net_structure = parent_params.pop('q_net_structure')
    q_net = dqn.QNetworkFCNN([observation_space.shape[0],*q_net_structure,len(children)]).to(device)
    dqn_agent = dqn.DQNAgent(
        action_space=gym.spaces.Discrete(len(children)),
        observation_space=observation_space,
        **parent_params,
        q_net=q_net,
        device=device,
    )
    # Hierarchy
    return HRLAgent(
        action_space=action_space,
        observation_space=observation_space,
        agent = dqn_agent,
        children = children,
        children_discount = [p['discount_factor'] for p in children_params],
    )

def make_agent_from_deploy_state(state : Union[str,Mapping], device : torch.device = torch.device('cpu')):
    if isinstance(state, str): # If it's a string, then it's the filename to the dilled state
        filename = state
        if not os.path.isfile(filename):
            raise Exception('No file found at %s' % filename)
        with open(filename, 'rb') as f:
            state = dill.load(f)
    if not isinstance(state, Mapping):
        raise Exception('Invalid state')
    parent_agent = dqn.make_agent_from_deploy_state(state['agent'], device)
    children_agents = [sac.make_agent_from_deploy_state(s,device) for s in state['children']]
    return HRLAgent(
        action_space=state['action_space'],
        observation_space=state['observation_space'],
        agent = parent_agent,
        children = children_agents,
        children_discount = 0.99,
    )

if __name__ == "__main__":
    import os
    import itertools
    import pprint

    from tqdm import tqdm
    import torch
    import torch.cuda
    import numpy as np
    import gym
    import gym.envs
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    
    #import rl.agent.smdp.rand as rand
    #import rl.agent.smdp.constant as constant

    def make_env(env_name):
        env = gym.make(env_name)
        if isinstance(env,gym.wrappers.TimeLimit):
            env = env.env
        return env
    
    def train(envs, agent, training_steps=1_000_000, test_frequency=1000, render=False, plot_filename='plot.png', log_plot_filename='plot-log.png'):
        test_results = {}
        env = envs[0]
        env_test = envs[1]

        done = True
        for i in tqdm(range(training_steps), desc='training'):
            if i % test_frequency == 0:
                video_file_name = os.path.join('output','video-%d.avi'%i)
                test_results[i] = [test(env_test, agent, render=(i==0 and render), video_file_name=video_file_name) for i in tqdm(range(5), desc='testing')]
                avg = np.mean([x['total_reward'] for x in test_results[i]])
                tqdm.write('Iteration {i}\t Average reward: {avg}'.format(i=i,avg=avg))
                tqdm.write(pprint.pformat(test_results[i], indent=4))
                plot(test_results, plot_filename)
                plot(test_results, log_plot_filename, log_scale=True)
                plot_actions(test_results, 'plot-actions.png')
                plot_actions_test(test_results, 'plot-actions-test.png')
            if done:
                obs = env.reset()
                agent.observe(obs, testing=False)
            obs, reward, done, _ = env.step(agent.act(testing=False))
            agent.observe(obs, reward, done, testing=False)
        env.close()

        return test_results

    def test(env, agent, render=False, video_file_name='output.avi'):
        total_reward = 0
        total_steps = 0

        obs = env.reset()
        agent.observe(obs, testing=True)
        if render:
            frame = env.render(mode='rgb_array')
            video = cv2.VideoWriter(video_file_name, 0, 60, (frame.shape[0],frame.shape[1])) # type: ignore
            video.write(frame)
        for total_steps in tqdm(itertools.count(), desc='test episode'):
            obs, reward, done, _ = env.step(agent.act(testing=True))
            if render:
                frame = env.render(mode='rgb_array')
                video.write(frame) # type: ignore
            total_reward += reward
            agent.observe(obs, reward, done, testing=True)
            if done:
                break
        env.close()
        if render:
            video.release() # type: ignore

        def compute_action_counts_test():
            parent_action = agent.logger[-1]['parent_action_testing']
            count = [0] * agent.num_children
            for a in parent_action:
                count[a] += 1
            return count

        def compute_action_counts_train():
            count = [0] * agent.num_children
            for x,_ in zip(reversed(agent.logger),range(1000)):
                if 'parent_action_training' not in x:
                    continue
                a = x['parent_action_training']
                count[a] += 1
            return count

        return {
            'total_steps': total_steps,
            'total_reward': total_reward,
            'parent_actions_count_test': compute_action_counts_test(),
            'parent_actions_count_train': compute_action_counts_train(),
        }

    def save_checkpoint():
        pass

    def plot(test_results, filename, log_scale=False):
        keys = sorted(test_results.keys())
        y = [np.mean([x['total_reward'] for x in test_results[i]]) for i in keys]
        plt.plot(keys,y)
        plt.ylabel('Total Reward')
        plt.xlabel('Steps')
        plt.grid()
        if log_scale:
            plt.yscale('log')
        plt.savefig(filename)
        plt.close()

    def plot_actions(test_results, filename):
        keys = sorted(test_results.keys())
        y = np.array([
            [x['parent_actions_count_train'] for x in test_results[i]] for i in keys
        ])
        y = y.mean(1)
        y = y/y.sum(1, keepdims=True)
        if y.shape[0] < 3:
            return
        y = y.transpose()
        plt.stackplot(keys, *y, labels=range(y.shape[0]))
        plt.ylabel('Training Subpolicy Choice Distribution')
        plt.xlabel('Steps')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(filename)
        plt.close()

    def plot_actions_test(test_results, filename):
        keys = sorted(test_results.keys())
        y = np.array([
            [x['parent_actions_count_test'] for x in test_results[i]] for i in keys
        ])
        y = y.mean(1)
        y = y/y.sum(1, keepdims=True)
        if y.shape[0] < 3:
            return
        y = y.transpose()
        plt.stackplot(keys, *y, labels=range(y.shape[0]))
        plt.ylabel('Training Subpolicy Choice Distribution')
        plt.xlabel('Steps')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig(filename)
        plt.close()

    if torch.cuda.is_available():
        print('GPU found')
        device = torch.device('cuda')
    else:
        print('No GPU found. Running on CPU.')
        device = torch.device('cpu')

    available_envs = [env_spec.id for env_spec in gym.envs.registry.all()]
    if 'Hopper-v1' in available_envs:
        env_name = 'Hopper-v1'
    else:
        env_name = 'Hopper-v2'
    #env_name = 'HopperBulletEnv-v0'
    env = [make_env(env_name), make_env(env_name)]

    sac_agent = sac.SACAgent(
        action_space=env[0].action_space,
        observation_space=env[0].observation_space,
        reward_scale=5,
        q_net_1 = sac.QNetwork(env[0].observation_space.shape[0], env[0].action_space.shape[0]).to(device),
        q_net_2 = sac.QNetwork(env[0].observation_space.shape[0], env[0].action_space.shape[0]).to(device),
        v_net   = sac.VNetwork(env[0].observation_space.shape[0]).to(device),
        pi_net  = sac.PolicyNetwork(env[0].observation_space.shape[0], env[0].action_space.shape[0]).to(device),
        device=device,
    )
    sac_agent2 = sac.SACAgent(
        action_space=env[0].action_space,
        observation_space=env[0].observation_space,
        reward_scale=5,
        q_net_1 = sac.QNetwork(env[0].observation_space.shape[0], env[0].action_space.shape[0]).to(device),
        q_net_2 = sac.QNetwork(env[0].observation_space.shape[0], env[0].action_space.shape[0]).to(device),
        v_net   = sac.VNetwork(env[0].observation_space.shape[0]).to(device),
        pi_net  = sac.PolicyNetwork(env[0].observation_space.shape[0], env[0].action_space.shape[0]).to(device),
        device=device,
    )
    dqn_agent = dqn.DQNAgent(
        action_space=gym.spaces.Discrete(2),
        observation_space=env[0].observation_space,
        discount_factor=0.99,
        #learning_rate=1e-4,
        #update_frequency=1,
        #target_update_frequency=1_000,
        #polyak_rate=1,
        warmup_steps=10_000,
        replay_buffer_size=100_000,
        q_net=dqn.QNetworkFCNN([env[0].observation_space.shape[0],10,2]).to(device),
        device=device,
    )
    #random_agent = rand.RandomAgent(env[0].action_space)
    agent = HRLAgent(
        action_space=env[0].action_space,
        observation_space=env[0].observation_space,
        agent = dqn_agent,
        #agent = constant.ConstantAgent(action=0),
        #agent = rand.RandomAgent(gym.spaces.Discrete(2)),
        children = [
            sac_agent,
            sac_agent2,
            #rand.RandomAgent(env[0].action_space),
        ],
        children_discount = [0.99, 0.99]
    )

    results = train(env,agent,plot_filename='plot.png',training_steps=4_000_000)
    test(env[1], agent)

