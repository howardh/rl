import itertools
import haiku as hk
import jax
import jax.numpy as jnp
from tqdm import tqdm
import gym
import copy
from collections import defaultdict

import optax
import bsuite
import bsuite.baselines.jax.actor_critic.agent
import bsuite.baselines.jax.dqn.agent
import dm_env

from rl.agent.agent import Agent
from . import ReplayBufferJax

class BsuiteAgent(Agent):
    def __init__(self, action_space, observation_space, discount_factor):
        self.action_space = action_space
        self.observation_space = observation_space
        self.discount_factor = discount_factor

        self.obs = [[],[]]
        self.timesteps = [[],[]]
        self.last_action = None
        self.debug = defaultdict(lambda: [])

    def observe_change(self, obs, reward=None, terminal=False, testing=False):
        obs = jnp.asarray(obs, dtype=jnp.float32)

        # Create TimeStep
        step_type = dm_env.StepType.FIRST
        if reward is not None:
            step_type = dm_env.StepType.MID
            reward = float(reward)
        if terminal:
            step_type = dm_env.StepType.LAST
        timestep = dm_env.TimeStep(
                step_type=step_type,
                reward=reward,
                discount=self.discount_factor,
                observation=obs
        )

        if reward is None: # reward is None if it is the first observation of the episode
            self.obs[testing] = (None, obs)
            self.timesteps[testing] = (None, timestep)
        else:
            self.obs[testing] = (self.obs[testing][1], obs)
            self.timesteps[testing] = (self.timesteps[testing][1], timestep)

        if not testing and self.obs[testing][0] is not None:
            self.agent.update(
                    self.timesteps[testing][0],
                    self.last_action,
                    self.timesteps[testing][1],
            )

    def train(self):
        pass

    def act(self, testing=False):
        timestep = self.timesteps[testing][1]
        action = self.agent.select_action(timestep)
        if not testing:
            self.last_action = action
        return action

    def test_once(self, env, max_steps=float('inf'), render=False):
        """
        Run an episode on the environment by following the target behaviour policy (Probably using a greedy deterministic policy).

        env
            Environment on which to run the test
        """
        reward_sum = 0
        obs = env.reset()
        self.observe_change(obs,testing=True)
        for steps in itertools.count():
            if steps > max_steps:
                break
            action = self.act(testing=True)
            obs, reward, done, _ = env.step(action)
            self.observe_change(obs,reward,done,testing=True)
            reward_sum += reward
            if render:
                env.render()
            if done:
                break
        return {'total_rewards': reward_sum, 'steps': steps}

    def state_dict(self):
        return {
                'discount_factor': self.discount_factor,
                'obs': self.obs,
                'timesteps': self.timesteps,
                'last_action': self.last_action,
                'debug': self.debug,
                'agent_state': self.agent._state,
        }
    def load_state_dict(self, state):
        state = {k:v for k,v in state.items()} # Copy so we can pop without modifying the original
        self.agent._state = state.pop('agent_state')
        for k,v in state.items():
            self.__setattr__(k,v)

class A2CAgent(BsuiteAgent):
    def __init__(self, action_space, observation_space, discount_factor, learning_rate=1e-4,
            sequence_length=32, td_lambda=0.9,
            network_structure=None, num_layers=3, layer_size=128,
            rng=None):
        super().__init__(action_space, observation_space, discount_factor)

        if network_structure is None:
            network_structure = [layer_size]*num_layers

        obs_spec = dm_env.specs.Array(shape=observation_space.low.shape, dtype=jnp.float32)
        action_spec = dm_env.specs.DiscreteArray(num_values=action_space.n)
        def network(inputs):
            flat_inputs = hk.Flatten()(inputs)
            torso = hk.nets.MLP(network_structure)
            policy_head = hk.Linear(action_spec.num_values)
            value_head = hk.Linear(1)
            embedding = torso(flat_inputs)
            logits = policy_head(embedding)
            value = value_head(embedding)
            return logits, jnp.squeeze(value, axis=-1)
        self.agent = bsuite.baselines.jax.actor_critic.agent.ActorCritic(
            obs_spec=obs_spec,
            action_spec=action_spec,
            network=network,
            optimizer=optax.adam(learning_rate),
            rng=rng,
            sequence_length=sequence_length,
            discount=discount_factor,
            td_lambda=td_lambda,
        )
    def state_dict(self):
        return {
                **super().state_dict(),
                'agent_rng': self.agent._rng.internal_state,
        }
    def load_state_dict(self, state):
        state = {k:v for k,v in state.items()} # Copy so we can pop without modifying the original
        self.agent._rng.replace_internal_state(state.pop('agent_rng'))
        super().load_state_dict(state)

class DQNAgent(BsuiteAgent):
    """ Wrapper around the Bsuite Jax DQN agent.
    Note: Not deterministic w.r.t. rng.
    """
    def __init__(self, action_space, observation_space, discount_factor,
            min_replay_size=10, epsilon=0.05, batch_size=32, replay_capacity=1000,
            network_structure=None, num_layers=2, layer_size=64,
            learning_rate=1e-4, rng=None):
        super().__init__(action_space, observation_space, discount_factor)
        self.rng = rng

        if network_structure is None:
            network_structure = [layer_size]*num_layers

        obs_spec = dm_env.specs.Array(shape=observation_space.low.shape, dtype=jnp.float32)
        action_spec = dm_env.specs.DiscreteArray(num_values=action_space.n)
        def network(inputs: jnp.ndarray) -> jnp.ndarray:
            flat_inputs = hk.Flatten()(inputs)
            mlp = hk.nets.MLP(network_structure+[action_spec.num_values])
            action_values = mlp(flat_inputs)
            return action_values
        self.agent = bsuite.baselines.jax.dqn.agent.DQN(
            obs_spec=obs_spec,
            action_spec=action_spec,
            network=network,
            optimizer=optax.adam(learning_rate),
            rng=self.rng,
            discount=discount_factor,
            min_replay_size=min_replay_size,
            batch_size=batch_size,
            epsilon=epsilon,
            replay_capacity=replay_capacity,
            sgd_period=1,
            target_update_period=4
        )

    def state_dict(self):
        return {
                **super().state_dict(),
                'rng': self.rng.internal_state,
        }
    def load_state_dict(self, state):
        state = {k:v for k,v in state.items()} # Copy so we can pop without modifying the original
        self.rng.replace_internal_state(state.pop('rng'))
        super().load_state_dict(state)
