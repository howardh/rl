"""
FrozenLake-v0

See https://github.com/openai/gym/wiki/FrozenLake-v0

SFFF
FHFH
FFFH
HFFG

Actions: 

+--+--+--+--+
|00|01|02|03|
+--+--+--+--+
|04|05|06|07|
+--+--+--+--+
|08|09|10|11|
+--+--+--+--+
|12|13|14|15|
+--+--+--+--+

"""

import itertools
from typing import Callable, Any
from enum import Enum

import gym
import gym.spaces
import numpy as np

class actions(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

##################################################
# Optimal policy and values
##################################################

optimal_policy = {
    0:  actions.LEFT.value,
    1:  actions.UP.value,
    2:  actions.UP.value,
    3:  actions.UP.value,
    4:  actions.LEFT.value,
    5:  None,
    6:  actions.LEFT.value, # Or right
    7:  None,
    8:  actions.UP.value,
    9:  actions.DOWN.value,
    10: actions.LEFT.value,
    11: None,
    12: None,
    13: actions.RIGHT.value,
    14: actions.DOWN.value,
    15: None,
}
optimal_policy_state_action_value = [
    [0.8235293990274958, 0.8235293975689129, 0.8235293975689129, 0.8235293972521844],
    [0.5490195976819288, 0.5490195966462433, 0.5490195951876605, 0.8235293947579163],
    [0.7254901803116807, 0.7254901797742794, 0.7254901787385941, 0.823529391726247],
    [0.549019593614574, 0.549019593614574, 0.5490195930771729, 0.8235293901531604],
    [0.8235293999546268, 0.549019600384371, 0.5490196000676426, 0.54901959945724],
    [None,None,None,None],
    [0.5294117555985274, 0.2549019585225398, 0.5294117555985274, 0.27450979707598755],
    [None,None,None,None],
    [0.549019600384371, 0.5490196018544196, 0.5490196012440169, 0.8235294017414038],
    [0.5686274457352289, 0.8235294042577687, 0.5490196037603818, 0.5294117590199265],
    [0.764705875732252, 0.588235290608232, 0.4901960743752193, 0.4509803864810528],
    [None,None,None,None],
    [None,None,None,None],
    [0.5686274465948749, 0.6078431344890414, 0.8823529358460741, 0.588235290608232],
    [0.862745093011581, 0.9411764678223746, 0.9019607811070722, 0.8823529370937152],
    [None,None,None,None],
] # optimal_policy_state_action_value[state][action]
optimal_policy_state_value = [
    0.823529370422352,
    0.8235293565641773,
    0.8235293467240036,
    0.8235293416180891,
    0.8235293734316279,
    None,
    0.5294117351452877,
    None,
    0.8235293792311369,
    0.8235293873987377,
    0.764705860863551,
    None,
    None,
    0.882352923875104,
    0.9411764616108227,
    None,
] # optimal_policy_state_value[state]

##################################################
# Gym Wrappers
##################################################

def _obs_to_onehot(obs : int) -> np.ndarray:
    output = np.zeros([16])
    output[obs] = 1
    return output
def _onehot_to_obs(obs : np.ndarray):
    return obs.squeeze().argmax().item()

class OnehotObs(gym.Wrapper):
    def __init__(self, env, obs_type=np.float32):
        self.env = env
        obs_size = env.observation_space.n
        self.observation_space = gym.spaces.Box(0, 1,
                [obs_size],
                obs_type)
        self.action_space = env.action_space
    def reset(self):
        obs = self.env.reset()
        obs = _obs_to_onehot(obs)
        return obs
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = _obs_to_onehot(obs)
        return obs, reward, done, info

class AtariObs(gym.Wrapper):
    def __init__(self, env, shape=[84,84], num_frames=4):
        self.env = env
        self.observation_space = gym.spaces.Box(0, 255,
                (num_frames, shape[0], shape[1]),
                np.uint8)
        self.action_space = env.action_space
    def reset(self):
        obs = self.env.reset()
        obs = self._to_atari(obs)
        return obs
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._to_onehot(obs)
        return obs, reward, done, info
    def _to_atari(self, val : int):
        val = val
        pass # TODO

##################################################
# Utils
##################################################

def compute_state_value_difference(v : Callable[[Any], float], input_type='int') -> float:
    """ Compute the mean absolute difference between the values predicted by the given function, and the true state value of an optimal policy.
    """
    diffs = []
    for s in range(16):
        if optimal_policy_state_value[s] is None:
            continue

        true_val = optimal_policy_state_value[s]

        if input_type == 'int':
            s = s
        elif input_type == 'one_hot':
            s = _obs_to_onehot(s)
        elif input_type == 'atari':
            raise NotImplementedError()
        else:
            raise Exception('Invalid input type: %s' % input_type)

        d = np.abs(true_val-v(s))
        diffs.append(d)
    return np.mean(diffs).item()

if __name__=='__main__':
    env = gym.make('FrozenLake-v0')

    ##################################################
    # Compute Transition Matrix
    ##################################################

    a = 0.1
    q = [ [10 for _ in range(4)] for _ in range(16) ] # q[state][action]
    t = [ [[0 for _ in range(16)] for _ in range(4)] for _ in range(16) ] # t[state][action][next_state]
    q_diff = []
    
    done = True
    obs = env.reset()
    for _ in range(1_000_000):
        if done:
            obs = env.reset()
        if np.random.rand() < 0.1:
            action = env.action_space.sample()
        else:
            action = np.argmax(q[obs])
        next_obs, reward, done, _ = env.step(action)

        t[obs][action][next_obs] += 1
        if done:
            q_target = reward
        else:
            q_target = reward + max(q[next_obs])
        q_diff.append(np.abs(q[obs][action]-q_target))
        q[obs][action] = (1-a)*q[obs][action] + a*q_target

        obs = next_obs

        if len(q_diff) > 100:
            print(np.mean(q_diff))
            if np.mean(q_diff) < 0.01:
                break
            q_diff = []

    # Transition probabilities from data
    transition_probs = np.rint(np.nan_to_num(np.array(t)/np.array(t).sum(2,keepdims=True))*3)/3

    # Check that they're proper probability distributions
    if set(transition_probs.sum(2).flatten().tolist()) != {0,1}:
        raise Exception('Failed to create transition probability matrix.')

    ##################################################
    # Compute V (state value)
    ##################################################

    # Transition matrix
    p = np.zeros([16,16])
    for s in range(16):
        a = optimal_policy[s]
        if a is None:
            continue
        p[s,:] = transition_probs[s,a,:]

    # Rewards
    r = np.zeros([16])
    r[15] = 1

    # Compute state value
    v = np.zeros([16])
    while True:
        targ = r+p@v
        diff = np.abs(v-targ).sum()
        print(diff)
        if diff < 1e-8:
            break
        v = targ

    print('State values:', v)

    ##################################################
    # Compute Q (state-action value) and optimal policy
    ##################################################

    sa_to_int = lambda s,a: s*4+a
    q = np.zeros([16*4])
    pi = [0 for _ in range(16)]
    r = np.zeros([16*4])
    for a in range(4):
        r[sa_to_int(15,a)] = 1
    while True:
        # Update transition probability matrix
        p = np.zeros([16*4,16*4])
        for s0,a0,s1 in itertools.product(range(16),range(4),range(16)):
            a1 = pi[s1]
            p[sa_to_int(s0,a0),sa_to_int(s1,a1)] = transition_probs[s0][a0][s1]
        # Update Q function
        while True:
            targ = r+p@q
            diff = np.abs(q-targ).sum()
            print(diff)
            if diff < 1e-6:
                break
            q = targ
        # Update policy
        new_pi = [
                np.argmax([q[sa_to_int(s,a)] for a in range(4)])
                for s in range(16)
        ]
        # Check if policy changed
        if new_pi == pi:
            break
        pi = new_pi
    while True:
        targ = r+p@q
        diff = np.abs(q-targ).sum()
        print(diff)
        if diff < 1e-8:
            break
        q = targ

    # Check that it matches the optimal policy
    for s in range(16):
        if optimal_policy[s] is not None:
            if optimal_policy[s] == pi[s]:
                print('Policy at state %d verified.' % s)
            else:
                print('Policy at state %d mismatch.' % s)

    print('State-action values:', q)
    print('Policy:', pi)
