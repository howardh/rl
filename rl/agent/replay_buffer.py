import torch
import torch.utils.data
from collections import deque

import jax
import jax.numpy as jnp

class FIFOBuffer(torch.utils.data.Dataset):
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.index = 0
        self.prev_transition = None

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
    def add_transition(self, obs0, action0, reward, obs, terminal=False):
        transition = (obs0, action0, reward, obs, terminal)
        self.add(transition)

    def step(self, obs, reward, action, terminal=False):
        if self.prev_transition is not None:
            obs0, _, action0 = self.prev_transition
            self.add_transition(obs0, action0, reward, obs, terminal)
        if terminal:
            self.prev_transition = None
        else:
            self.prev_transition = (obs, reward, action)

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

# Jax version
# This is identical to the above `ReplayBuffer`, except for the added `sample` function
class ReplayBufferJax:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.index = 0
        self.prev_transition = None

    def _add_to_buffer(self,transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.index] = transition
            self.index = (self.index+1)%self.max_size

    def add_transition(self, obs0, action0, reward, obs, terminal=False):
        transition = (obs0, action0, reward, obs, terminal)
        self._add_to_buffer(transition)

    def step(self, obs, reward, action, terminal=False):
        if self.prev_transition is not None:
            obs0, _, action0 = self.prev_transition
            self.add_transition(obs0, action0, reward, obs, terminal)
        if terminal:
            self.prev_transition = None
        else:
            self.prev_transition = (obs, reward, action)

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

    def sample(self, random_key, batch_size):
        choices = jax.random.choice(
                key=random_key,
                a=len(self.buffer),
                shape=[batch_size],
                replace=False
        )
        batch = default_collate([self.buffer[c] for c in choices])
        return batch

# Below copied from the pytorch implementation of default_collate
# https://github.com/pytorch/pytorch/blob/9920ae665b6e3e7b3a5c27ab8650972463999cd2/torch/utils/data/_utils/collate.py#L42
import re
import collections.abc
string_classes = (str, bytes)
int_classes = int
container_abcs = collections.abc
np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, jnp.ndarray):
        out = None
        return jnp.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([jnp.asarray(b) for b in batch])
        elif elem.shape == ():  # scalars
            return jnp.asarray(batch)
    elif isinstance(elem, float):
        return jnp.array(batch, dtype=jnp.float64)
    elif isinstance(elem, int_classes):
        return jnp.array(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
