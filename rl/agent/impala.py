import os
from abc import abstractmethod
import cProfile
import copy
from typing import List, Mapping, Tuple, Iterable
import threading
import pprint
import time
import timeit

import gym
import gym.spaces
import torch
from torch import multiprocessing as mp
from torch.utils.data.dataloader import default_collate

from experiment.logger import Logger
from frankenstein.buffer.shared import SharedBuffer
from frankenstein.value.v_trace import v_trace_return
from frankenstein.loss.policy_gradient import advantage_policy_gradient_loss

from rl.experiments.training._utils import make_env


class RecurrentModel(torch.nn.Module):
    @abstractmethod
    def init_hidden(self, batch_size): ...


class Resnet(torch.nn.Module):
    """
    See figure 3 in https://arxiv.org/pdf/1802.01561.pdf
    and table 7 in https://arxiv.org/pdf/1809.04474.pdf
    """
    def __init__(self, num_actions) -> None:
        super().__init__()
        self.seqs = torch.nn.ModuleList()
        num_channels = [4, 16,32, 32]
        for in_channels, out_channels in zip(num_channels, num_channels[1:]):
            conv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1),
                    torch.nn.MaxPool2d(kernel_size = 3, stride = 2),
            )
            res = [torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels = out_channels,
                        out_channels = out_channels,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels = out_channels,
                        out_channels = out_channels,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1),
            ) for _ in range(2)]
            self.seqs.append(torch.nn.ModuleList([conv, *res]))

        self.fc = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(32 * 9 * 9, 256),
                torch.nn.ReLU(),
        )
        self.action = torch.nn.Linear(256, num_actions)
        self.value = torch.nn.Linear(256, 1)
    
    def forward(self, x, hidden):
        hidden = hidden # Unused
        x = x.float() / 255.0
        for conv, res0, res1 in self.seqs: # type: ignore
            x = conv(x)
            x = res0(x) + x
            x = res1(x) + x
        x = self.fc(x)
        return {
            'action': self.action(x),
            'state_value': self.value(x),
            'hidden': (),
        }

    def init_hidden(self, batch_size=1):
        batch_size = batch_size
        return ()


def bin_envs(environments):
    """ Organize the environments by their observation/action spaces. """
    spaces = []
    def find_matching_space_index(env):
        for i, (obs_space, act_space) in enumerate(spaces):
            if env.observation_space == obs_space and env.action_space == act_space:
                return i
        return None
    
    output = []
    for env in environments:
        # Check if the spaces match any of the existing spaces.
        space_index = find_matching_space_index(env)
        if space_index is None:
            spaces.append((env.observation_space, env.action_space))
            space_index = len(spaces) - 1
        output.append(space_index)
    return output


def copy_tensor(src, dest, dest_indices=...):
    """ Copy the observation from src to dest. """
    if isinstance(dest, torch.Tensor):
        dest.__setitem__(dest_indices, src)
    elif isinstance(dest, Mapping):
        for k in src.keys():
            copy_tensor(src[k], dest[k])
    elif isinstance(dest, Iterable):
        for s,d in zip(src, dest):
            copy_tensor(s, d)
    else:
        raise NotImplementedError(f"Unknown data type: {type(src)}")


def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, Mapping):
        return {k: to_tensor(v, device) for k,v in x.items()}
    elif isinstance(x, Tuple):
        return tuple(to_tensor(v, device) for v in x)
    else:
        try:
            return torch.tensor(x, device=device)
        except:
            raise NotImplementedError(f"Unknown data type: {type(x)}")


def unsqueeze(x, dim):
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(dim)
    elif isinstance(x, Mapping):
        return {k: unsqueeze(v, dim) for k,v in x.items()}
    elif isinstance(x, Tuple):
        return tuple(unsqueeze(v, dim) for v in x)
    else:
        raise NotImplementedError(f"Unknown data type: {type(x)}")


def transpose(x, dim0, dim1):
    if isinstance(x, torch.Tensor):
        return x.transpose(dim0, dim1)
    elif isinstance(x, Mapping):
        return {k: transpose(v, dim0, dim1) for k,v in x.items()}
    elif isinstance(x, Tuple):
        return tuple(transpose(v, dim0, dim1) for v in x)
    elif isinstance(x, List):
        return list(transpose(v, dim0, dim1) for v in x)
    else:
        raise NotImplementedError(f"Unknown data type: {type(x)}")


def get_slice(x, indices):
    if isinstance(x, torch.Tensor):
        return x.__getitem__(indices)
    elif isinstance(x, Mapping):
        return {k: get_slice(v, indices) for k,v in x.items()}
    elif isinstance(x, Tuple):
        return tuple(get_slice(v, indices) for v in x)
    else:
        raise NotImplementedError(f"Unknown data type: {type(x)}")


class ImpalaTrainer:
    """
    Implementation of Impala. See https://arxiv.org/abs/1802.01561.

    Hyperparameters for Atari are in table G.1

    See original implementation here: https://github.com/deepmind/scalable_agent
    """
    def __init__(self,
            env_configs : List[Mapping],
            buffer_obs_space: Mapping,
            buffer_act_space: Mapping,
            total_steps : int,
            net : RecurrentModel,
            discount_factor : float = 0.99,
            learning_rate : float = 6e-4,
            lr_scheduler : dict = {
                'type': 'LinearLR',
                'parameters': {
                    'start_factor': 1.0,
                    'end_factor': 0,
                    'total_iters': 200_000_000/32/20, # Total steps / batch size / rollout length
                }
            },
            reward_clip : Tuple[float,float] = (-1,1),
            vf_loss_coeff : float = 0.5,
            entropy_loss_coeff : float = 0.01,
            max_grad_norm : float = 40.0,
            max_rollout_length : int = 20,
            batch_size : int = 32,
            num_train_loop_workers : int = 2,
            num_buffers_per_env : int = 2,
            device : torch.device = torch.device('cpu'),
            logger : Logger = None,
        ):
        self._env_configs = env_configs
        self._buffer_obs_space = buffer_obs_space
        self._buffer_act_space = buffer_act_space

        self.total_steps = total_steps

        self.discount_factor = discount_factor
        self.reward_clip = reward_clip
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.max_rollout_length = max_rollout_length
        self.batch_size = batch_size
        self.num_train_loop_workers = num_train_loop_workers
        self.num_buffers_per_env = num_buffers_per_env
        self.device = device

        # Validate input
        if max_rollout_length < 1:
            raise ValueError(f'Rollout length must be >= 1. Received {max_rollout_length}.')

        self._steps = 0 # Number of steps experienced

        self.net = copy.deepcopy(net).to(torch.device('cpu')) # Make a copy of the model to give to the actor processes. We need to make a copy because multiprocessing + working doesn't support CUDA models, but we want the training to be done with CUDA.
        self.net.share_memory()

        self.learner_net = net.to(self.device)

        #self.optimizer = torch.optim.Adam(self.learner_net.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.RMSprop(self.learner_net.parameters(), lr=learning_rate, momentum=0, eps=0.01)
        self.scheduler = self._init_lr_scheduler(self.optimizer, config=lr_scheduler)

        # Logging
        if logger is None:
            self.logger = Logger(key_name='step', allow_implicit_key=True)
            self.logger.log(step=0)
        else:
            self.logger = logger

    def _init_lr_scheduler(self, optimizer, config):
        if config is None:
            return None
        valid_schedulers = {
                cls.__name__: cls
                for cls in [
                    torch.optim.lr_scheduler.LinearLR
                ]
        }
        if config['type'] not in valid_schedulers:
            raise Exception(f'Invalid lr_scheduler type: "{config["type"]}". Valid types are: {list(valid_schedulers.keys())}.')
        return valid_schedulers[config['type']](optimizer, **config['parameters'])

    def _action_dist(self, action_space, output):
        if isinstance(action_space, gym.spaces.Box):
            return torch.distributions.Normal(output['action_mean'], output['action_std'])
        elif isinstance(action_space, gym.spaces.Discrete):
            return torch.distributions.Categorical(logits=output['action'])
        else:
            raise NotImplementedError(f'Unsupported action space of type {type(action_space)}')

    def _actor_loop(self, env, task_id, net, buffer, seed=None):
        if seed is None:
            seed = os.getpid() + int(time.time())
            thread_id = threading.current_thread().ident
            if thread_id is not None:
                seed += thread_id
        torch.random.manual_seed(seed)

        device = torch.device('cpu')
        with torch.no_grad():
            hidden = net.init_hidden()

            episode_step_count = 0
            episode_return = 0

            obs = None
            reward = None
            done = True
            action = None

            while True:
                if done:
                    obs, reward, done = env.reset(), 0, False
                    hidden = net.init_hidden()
                    episode_step_count = 0
                    episode_return = 0
                else:
                    obs, reward, done, _ = env.step(action)

                episode_step_count += 1
                episode_return += reward

                output = net(unsqueeze(to_tensor(obs, device), 0), hidden)
                action_dist = self._action_dist(env.action_space, output)
                action = action_dist.sample().item()
                hidden = output['hidden']

                buffer.append_obs(
                        obs = obs,
                        reward = reward,
                        terminal = done,
                        misc = {
                            'episode_step_count': episode_step_count,
                            'episode_return': episode_return,
                            'state_value': output['state_value'],
                            'task': task_id,
                            'hidden': output['hidden'],
                        }
                )

                if not done:
                    buffer.append_action(action)

    def _train_loop(self, shared_net, net, buffer, optimizer, batch_size, action_space, profiler, lock=threading.Lock(), lock2=threading.Lock()):
        while self._steps < self.total_steps:
            profiler.enable()
            batch = buffer.get_batch(batch_size, device=self.device)

            with lock:
                # batch.misc['hidden'] has shape
                # (seq_len, batch_size, num_blocks, hidden_size)
                # We need a hidden state with shape (num_blocks, batch_size, hidden_size)
                hidden = tuple(h[0,...].transpose(0,1) for h in batch.misc['hidden'])
                output = []
                for t in range(self.max_rollout_length+1):
                    # Get the next output
                    o = net(
                            get_slice(batch.obs, [t,...]),
                            hidden
                    )
                    output.append(o)
                    hidden = o['hidden']
                output = default_collate(output)

                action_dist = self._action_dist(action_space, output)
                entropy = action_dist.entropy()
                log_action_probs = action_dist.log_prob(batch.action.squeeze(2)).unsqueeze(2)

                # Misc computations
                with torch.no_grad():
                    v_trace = v_trace_return(
                        log_action_probs = log_action_probs[:-1],
                        old_log_action_probs = batch.misc['action_log_prob'][:-1],
                        state_values = output['state_value'][:-1],
                        next_state_values = output['state_value'][1:],
                        rewards = batch.reward[1:].clip(*self.reward_clip) if self.reward_clip is not None else batch.reward[1:],
                        terminals = batch.terminal[1:],
                        discount = self.discount_factor,
                        max_c = 1,
                        max_rho = 1,
                    )
                    advantages = v_trace-output['state_value'][:-1]

                # Value loss
                v_loss = (output['state_value'][:-1] - v_trace).pow(2).mean()

                # Policy loss
                pg_loss = advantage_policy_gradient_loss(
                    log_action_probs = log_action_probs[:-1],
                    advantages = advantages,
                    terminals = batch.terminal[:-1],
                )
                pg_loss = pg_loss.mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = self.vf_loss_coeff*v_loss + pg_loss + self.entropy_loss_coeff*entropy_loss

                # Train the network
                optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
                optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                shared_net.load_state_dict(net.state_dict())

                profiler.disable()

            with lock2:
                # Logging
                returns = batch.misc['episode_return'][batch.terminal]
                task_ids = batch.misc['task'][batch.terminal]
                labels = [self._env_configs[i].get('label', self._env_configs[i]['env_name']) for i in task_ids]
                returns_by_label = {l:[] for l in labels}
                for l,r in zip(labels, returns):
                    returns_by_label[l].append(r.item())
                self.logger.log(
                    step = self._steps,
                    losses = {
                        'v_loss': v_loss.item(),
                        'pg_loss': pg_loss.item(),
                        'entropy_loss': entropy_loss.item(),
                        'loss': loss.item(),
                    },
                    returns = {
                        label: torch.tensor(returns).mean().item()
                        for label, returns in returns_by_label.items()
                    },
                    v_trace = v_trace.mean().item(),
                    advantages = advantages.mean().item(),
                    entropy = entropy.mean().item(),
                    state_value = output['state_value'][:-1].mean().item(),
                )
                if len(returns_by_label) > 0:
                    pprint.pprint(returns_by_label)

                self._steps += self.max_rollout_length * batch_size

                # Free the buffer
                batch.release()

    def train(self):
        envs = [make_env(**conf) for conf in self._env_configs]
        bins = bin_envs(envs)
        if max(bins) > 0:
            raise NotImplementedError(f'Environments must have the same observation and action spaces.')

        num_buffers = len(self._env_configs)*self.num_buffers_per_env
        ctx = mp.get_context('fork')
        dummy_hidden = self.net.init_hidden(1)
        assert isinstance(dummy_hidden, Iterable)
        assert envs[0].observation_space is not None
        buffer = SharedBuffer(
                num_buffers = num_buffers,
                rollout_length = self.max_rollout_length,
                mp_context = ctx,
                observation_space = self._buffer_obs_space,
                action_space = self._buffer_act_space,
                misc_space = {
                    'type': 'dict',
                    'data': {
                        'episode_step_count': {
                            'type': 'box',
                            'shape': (1,),
                            'dtype': torch.int,
                        },
                        'episode_return': {
                            'type': 'box',
                            'shape': (1,),
                            'dtype': torch.float32,
                        },
                        'task': {
                            'type': 'box',
                            'shape': (1,),
                            'dtype': torch.uint8,
                        },
                        'state_value': {
                            'type': 'box',
                            'shape': (1,),
                            'dtype': torch.float32,
                        },
                        'action_log_prob': {
                            'type': 'box',
                            'shape': (1,),
                            'dtype': torch.float32,
                        },
                        'hidden': {
                            'type': 'tuple',
                            'data': [{
                                    'type': 'box',
                                    'shape': h[:,0,...].shape, # (num_blocks, batch_size, hidden_size)
                                    'dtype': torch.float32,
                            } for h in dummy_hidden] # type: ignore
                        }
                    }
                }
        )

        #self._actor_loop(envs[0], 0, self.net, buffer)
        actor_processes = []
        for i in range(len(self._env_configs)-1):
            process = ctx.Process(
                target=self._actor_loop,
                args=(
                    envs[i],
                    i,
                    self.net,
                    buffer
                )
            )
            process.start()
            actor_processes.append(process)

        # Start training processes/threads
        profilers = [cProfile.Profile() for _ in range(self.num_train_loop_workers)]
        #self._train_loop(
        #        shared_net = self.net,
        #        net = self.learner_net,
        #        buffer = buffer,
        #        optimizer = self.optimizer,
        #        batch_size = 2,
        #        action_space = envs[0].action_space,
        #        profiler = profilers[0],
        #)

        timer = timeit.default_timer
        start_time = timer()

        train_threads = []
        for i in range(self.num_train_loop_workers):
            thread = threading.Thread(
                target=self._train_loop,
                args=(
                    self.net,
                    self.learner_net,
                    buffer,
                    self.optimizer,
                    self.batch_size,
                    envs[0].action_space,
                    profilers[i],
                )
            )
            thread.start()
            train_threads.append(thread)

        try:
            while self._steps < self.total_steps:
                time.sleep(5)
                sps = self._steps / (timer() - start_time)
                print(f'{self._steps:,} steps -- {sps:.2f} steps/sec')
        except KeyboardInterrupt:
            pass

        for thread in train_threads:
            thread.join()
        for process in actor_processes:
            process.join(timeout=1)

        breakpoint()

    def state_dict(self):
        return {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            '_steps': self._steps,
        }

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._steps = state_dict['_steps']


def train_mp2():
    from rl.experiments.pathways.models import ModularPolicy2
    from rl.experiments.training._utils import make_env

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    env_name = 'ALE/Pong-v5'
    env_config = {
        'env_name': env_name,
        'atari': True,
        'episode_stack': 1,
        'dict_obs': True,
        'config': {
            'frameskip': 1,
            #'mode': 0,
            #'difficulty': 0,
            #'repeat_action_probability': 0.25,
            #'full_action_space': False,
        },
        'atari_config': {
            #'repeat_action_probability': 0.25,
            #'terminal_on_life_loss': False,
        }
    }

    envs = [make_env(**env_config) for _ in range(2)]

    observation_space = envs[0].observation_space
    assert observation_space is not None
    action_space = envs[0].action_space
    assert action_space is not None

    net = ModularPolicy2(
            inputs = {
                'obs': {
                    'type': 'GreyscaleImageInput',
                    'config': {
                        'in_channels': observation_space['obs'].shape[0],
                        'scale': 1/255.0,
                    }
                },
                'reward': {
                    'type': 'ScalarInput',
                },
                'done': {
                    'type': 'ScalarInput',
                },
                #'action': {
                #    'type': 'DiscreteInput',
                #    'config': {
                #        'input_size': num_actions,
                #    }
                #},
            },
            outputs = {
                'action': {
                    'type': 'LinearOutput',
                    'config': {
                        'output_size': action_space.n,
                    }
                },
                'state_value': {
                    'type': 'LinearOutput',
                    'config': {
                        'output_size': 1,
                    }
                },
            },
            input_size = 32,
            key_size = 32,
            value_size = 32,
            ff_size = 64,
            num_heads = 4,
            recurrence_type='RecurrentAttention9',
    )
    buffer_obs_space = {
        'type': 'dict',
        'data': {
            'obs': {
                'type': 'box',
                'shape': observation_space['obs'].shape,
                'dtype': torch.uint8,
            },
            'reward': {
                'type': 'box',
                'shape': (1,),
                'dtype': torch.uint8,
            },
            'done': {
                'type': 'box',
                'shape': (1,),
                'dtype': torch.bool,
            },
            'action': {
                'type': 'box',
                'shape': (1,),
                'dtype': torch.uint8,
            },
        }
    }
    buffer_act_space = {
        'type': 'discrete'
    }
    trainer = ImpalaTrainer(
            env_configs = [env_config]*8,
            buffer_obs_space = buffer_obs_space,
            buffer_act_space = buffer_act_space,
            net = net, # type: ignore
            num_train_loop_workers = 2,
            num_buffers_per_env = 2,
            batch_size = 8,
            device = device,
    )
    trainer.logger.init_wandb(wandb_params={
        'project': 'impala-test'
    })
    trainer.train()


def train_resnet():
    from rl.experiments.training._utils import make_env

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    env_name = 'ALE/Pong-v5'
    env_config = {
        'env_name': env_name,
        'atari': True,
        #'episode_stack': 1,
        #'dict_obs': False,
        'config': {
            'frameskip': 1,
            #'mode': 0,
            #'difficulty': 0,
            #'repeat_action_probability': 0.25,
            'repeat_action_probability': 0,
            #'full_action_space': False,
        },
        'atari_config': {
            #'repeat_action_probability': 0.25,
            'terminal_on_life_loss': True,
        }
    }

    envs = [make_env(**env_config) for _ in range(2)]

    observation_space = envs[0].observation_space
    assert observation_space is not None
    action_space = envs[0].action_space
    assert action_space is not None

    net = Resnet(
            num_actions = action_space.n,
    )
    buffer_obs_space = {
        'type': 'box',
        'shape': observation_space.shape,
        'dtype': torch.uint8,
    }
    buffer_act_space = {
        'type': 'discrete'
    }
    trainer = ImpalaTrainer(
            env_configs = [env_config]*24,
            buffer_obs_space = buffer_obs_space,
            buffer_act_space = buffer_act_space,
            total_steps = 200_000_000,
            net = net, # type: ignore
            num_train_loop_workers = 2,
            num_buffers_per_env = 4,
            device = device,
    )
    trainer.logger.init_wandb(wandb_params={
        'project': 'impala-test'
    })
    trainer.train()


if __name__=='__main__':
    torch.set_num_threads(1) # It hangs on certain operations if I don't have this. See https://github.com/pytorch/pytorch/issues/58962

    #train_mp2()
    train_resnet()
