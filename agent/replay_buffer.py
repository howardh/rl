import torch
import torch.utils.data
from collections import deque

class ReplayBuffer(torch.utils.data.Dataset):
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
            obs0, reward0, action0 = self.prev_transition
            self.add_transition(obs0, action0, reward, obs, terminal)
        if terminal:
            self.prev_transition = None
        else:
            self.prev_transition = (obs, reward, action)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

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
            obs_stack = [torch.empty_like(obs0)]*num_missing + list(self.obs_stack)
        else:
            obs_stack = list(self.obs_stack)

        transition = (obs_stack, action0, reward, obs, terminal, obs_mask)
        self._add_to_buffer(transition)
        if terminal:
            self.obs_stack.clear()

class ReplayBufferStackedObsAction(ReplayBuffer):
    def __init__(self, max_size):
        super().__init__(max_size)
        self.transition_stack = deque(maxlen=2)
    def add_transition(self, obs0, action0, obs1, action1, reward2, obs2, terminal=False):
        obs_mask = torch.tensor([obs0 is not None, obs1 is not None]).float()
        transition = (obs0,action0,obs1,action1,reward2,obs2,terminal,obs_mask)
        self._add_to_buffer(transition)
    def step(self, obs2, reward2, action2, terminal=False):
        if len(self.transition_stack) == 2:
            obs0, reward0, action0 = self.transition_stack[0]
            obs1, reward1, action1 = self.transition_stack[1]
            self.add_transition(
                    obs0, action0,
                    obs1, action1,
                    reward1, obs2, terminal)
        if terminal:
            self.transition_stack.clear()
        else:
            self.transition_stack.append((obs, reward, action))
