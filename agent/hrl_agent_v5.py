import haiku as hk
import jax.numpy as jnp
from tqdm import tqdm
import gym
import copy
from collections import defaultdict

from agent.agent import Agent
from agent.hdqn_agent import create_augmented_obs_transform_one_hot_action, AugmentedObservationStack

from . import ReplayBuffer

def get_obs(obs_stack, delay, action_mem):
    """ Return the observation the agent needs to act on. This can be fed
    directly to the policy network.  """

    obs = obs_stack.get(delay-action_mem,action_mem)
    if obs is not None:
        obs = jnp.asarray(obs)
    return obs

class HRLAgent_v5(Agent):
    def __init__(self, action_space, observation_space, discount_factor, actor_lr=1e-4, critic_lr=1e-3, polyak_rate=0.001, delay=1, action_mem=1, actor=None, critic=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.discount_factor = discount_factor
        self.polyak_rate = polyak_rate
        self.delay = delay
        self.action_mem = action_mem

        self.prev_obs = None
        self.replay_buffer = ReplayBuffer(50000)

        obs_stack_transform = create_augmented_obs_transform_one_hot_action(4)
        self.obs_stack = [
            AugmentedObservationStack(
                transform=obs_stack_transform,
                stack_len=self.delay+2,
                action_len=self.action_mem)
            for _ in range(2)
        ] # 0 = training stack, 1 = testing stack

        self.debug = defaultdict(lambda: [])

    def get_obs_sizes(self):
        """ Return an array with the size of the inputs for policies of each level of the hierarchy (highest to lowest). """
        if len(self.observation_space.high.shape) > 1:
            raise NotImplementedError('Cannot handle multidimensional observations')
        return [
            self.observation_space.high.shape[0]+self.action_space.n*self.action_mem,
            self.observation_space.high.shape[0]
        ]

    def observe_step(self, obs0, action0, reward1, obs1, terminal=False):
        obs0 = jnp.asarray(obs0)
        obs1 = jnp.asarray(obs1)
        self.replay_buffer.add_transition(obs0, action0, reward1, obs1, terminal)

    def train(self,batch_size,iterations):
        # TODO
        pass

    def act(self, observation, testing=False):
        obs_aug,obs,mask = self.get_current_obs(testing)

    def get_current_obs(self,testing=False):
        obs_stack = self.obs_stack[testing]

        obs_aug = obs_stack.get(self.delay-self.action_mem,self.action_mem)
        obs = obs_stack.get(0,0)
        mask = jnp.asarray([[obs_aug is not None, obs is not None]])

        def to_array(x):
            if x is None:
                size = self.get_obs_sizes()[0] # Only augmented obs can be None
                return jnp.zeros([size])
            return jnp.asarray(x).squeeze()

        return to_array(obs_aug),to_array(obs),mask
