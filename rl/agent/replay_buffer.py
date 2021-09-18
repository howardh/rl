from typing import List, Optional
from collections import deque
import warnings

import torch
import torch.utils.data

from gym.wrappers.frame_stack import LazyFrames

class FIFOBuffer(torch.utils.data.Dataset):
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.index = 0

    def add(self,value):
        if len(self.buffer) < self.max_size:
            self.buffer.append(value)
        else:
            self.buffer[self.index] = value
            self.index = (self.index+1)%self.max_size

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def state_dict(self):
        return {
                'buffer': self.buffer,
                'index': self.index,
                'prev_transition': self.prev_transition
        }

    def load_state_dict(self, state):
        self.buffer = state['buffer']
        self.index = state['index']
        self.prev_transition = state['prev_transition']

class ReplayBuffer(FIFOBuffer):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.prev_transition = None

    def add_transition(self, obs0, action0, reward, obs, terminal=False, misc=None):
        if misc is None:
            transition = (obs0, action0, reward, obs, terminal)
        else:
            transition = (obs0, action0, reward, obs, terminal, *misc)
        self.add(transition)

    def step(self, obs, reward, action, terminal=False):
        if self.prev_transition is not None:
            obs0, _, action0 = self.prev_transition
            self.add_transition(obs0, action0, reward, obs, terminal)
        if terminal:
            self.prev_transition = None
        else:
            self.prev_transition = (obs, reward, action)

    def state_dict(self):
        return {
                **super().state_dict(),
                'prev_transition': self.prev_transition
        }

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.prev_transition = state['prev_transition']

class AtariReplayBuffer(ReplayBuffer):
    """ A replay buffer designed to minimize memory usage in Atari environments. """
    def __init__(self, max_size, framestack=4):
        super().__init__(max_size)
        self._boundary_frames = {}
        self._new_episode = True
        self._framestack = framestack

    def __getitem__(self, index):
        obs_stack = []
        for i in range(self._framestack+1):
            index2 = (index-i+self.max_size)%self.max_size
            obs_stack.insert(0,self.buffer[index2][2])
            if index2 in self._boundary_frames:
                obs_stack = self._boundary_frames[index2] + obs_stack
                break
        obs0 = torch.stack([torch.tensor(o) for o in obs_stack[-self._framestack-1:-1]])
        obs1 = torch.stack([torch.tensor(o) for o in obs_stack[-self._framestack:]])
        action0, reward, _, terminal, *misc = self.buffer[index]
        return (obs0, action0, reward, obs1, terminal, *misc)

    def add_transition(self, obs0 : LazyFrames, action0, reward : float, obs : LazyFrames, terminal : bool = False, misc : Optional[List] = None):
        if misc is None:
            data = (action0, reward, obs._frames[-1], terminal)
        else:
            data = (action0, reward, obs._frames[-1], terminal, *misc)

        if self._new_episode:
            self._boundary_frames[self.index] = obs0._frames[:]
            self._new_episode = False

        if terminal:
            self._new_episode = True

        if len(self.buffer) < self.max_size:
            self.buffer.append(data)
            self.index = (self.index+1)%self.max_size
        else:
            next_index = (self.index+1)%self.max_size
            if self.index in self._boundary_frames:
                if next_index in self._boundary_frames:
                    del self._boundary_frames[self.index]
                else:
                    self._boundary_frames[next_index] = self._boundary_frames[self.index][1:]+[self.buffer[self.index][2]]
                    del self._boundary_frames[self.index]
            self.buffer[self.index] = data
            self.index = next_index

    def state_dict(self):
        return {
                **super().state_dict(),
                'boundary_frames': self._boundary_frames,
                'new_episode': self._new_episode,
        }

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._boundary_frames = state['boundary_frames']
        self._new_episode = True # XXX: I don't know how to restore an Atari environment state, so we're just going to start a new episode.
        warnings.warn('The replay buffer written with the assumption that the Atari environment is reset when reloading from a checkpoint.')

class ReplayBufferStackedObs(ReplayBuffer):
    def __init__(self, max_size, num_obs):
        super().__init__(max_size)
        self.obs_stack = deque(maxlen=num_obs)

    def add_transition(self, obs0, action0, reward, obs, terminal=False):
        self.obs_stack.append(obs0)

        num_obs = len(self.obs_stack)
        num_missing = self.obs_stack.maxlen-num_obs
        obs_mask = torch.tensor([False]*num_missing + [True]*num_obs).float()
        if len(self.obs_stack) != self.obs_stack.maxlen:
            obs_stack = [torch.zeros_like(obs0)]*num_missing + list(self.obs_stack)
        else:
            obs_stack = list(self.obs_stack)

        transition = (obs_stack, action0, reward, obs, terminal, obs_mask)
        self.add(transition)
        if terminal:
            self.obs_stack.clear()

    def state_dict(self):
        d = super().state_dict()
        d['obs_stack'] = tuple(self.obs_stack)
        return d

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for o in state['obs_stack']:
            self.obs_stack.append(o)

class ReplayBufferStackedObsAction(ReplayBuffer):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.transition_stack = deque(maxlen=2)
    def add_transition(self, obs0, action0, obs1, action1, reward2, obs2, terminal=False):
        obs_mask = torch.tensor([obs0 is not None, obs1 is not None]).float()
        transition = (obs0,action0,obs1,action1,reward2,obs2,terminal,obs_mask)
        self.add(transition)
    def step(self, obs2, reward2, action2, terminal=False):
        if len(self.transition_stack) == 2:
            obs0,       _, action0 = self.transition_stack[0]
            obs1, reward1, action1 = self.transition_stack[1]
            self.add_transition(
                    obs0, action0,
                    obs1, action1,
                    reward1, obs2, terminal)
        if terminal:
            self.transition_stack.clear()
        else:
            self.transition_stack.append((obs2, reward2, action2))
    def state_dict(self):
        d = super().state_dict()
        d['transition_stack'] = tuple(self.transition_stack)
        return d
    def load_state_dict(self, state):
        super().load_state_dict(state)
        for o in state['transition_stack']:
            self.transition_stack.append(o)
