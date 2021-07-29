import copy
import itertools
from collections import defaultdict
from typing import Mapping, Sequence, Union, Generic, TypeVar, Optional, Tuple

import dill
import numpy as np
import torch
import torch.nn
import torch.utils.data
import torch.distributions
import torch.optim
import gym.spaces

from rl.agent.agent import DeployableAgent
from rl.agent import ReplayBuffer

class QNetwork(torch.nn.Module):
    def __init__(self, num_features, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_features+num_actions,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=1),
        )
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = self.fc(x)
        return x
    def state_dict(self):
        return {
                'num_actions': self.num_actions,
                'num_features': self.num_features,
                'model_state_dict': super().state_dict(),
        }
    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict['model_state_dict'])
class VNetwork(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_features,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=1),
        )
    def forward(self, obs):
        x = obs
        x = self.fc(x)
        return x
    def state_dict(self):
        return {
                'num_features': self.num_features,
                'model_state_dict': super().state_dict(),
        }
    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict['model_state_dict'])
class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_features : int, num_actions : int, structure : Sequence[int] = [256,256]):
        super().__init__()
        self.num_actions = num_actions
        self.num_features = num_features
        self.structure = structure

        layers = []
        for in_size, out_size in zip([num_features,*structure],structure):
            layers.append(torch.nn.Linear(in_features=in_size,out_features=out_size))
            layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Sequential(*layers)
        self.fc_mean = torch.nn.Linear(in_features=structure[-1],out_features=num_actions)
        self.fc_log_std = torch.nn.Linear(in_features=structure[-1],out_features=num_actions)
    def forward(self, obs):
        x = obs
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        return mean,log_std
    def state_dict(self):
        return {
                'num_actions': self.num_actions,
                'num_features': self.num_features,
                'structure': self.structure,
                'model_state_dict': super().state_dict(),
        }
    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict['model_state_dict'])

ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')
class ObservationStack(Generic[ObsType,ActionType]):
    def __init__(self) -> None:
        self.prev : Optional[Tuple[ObsType,Optional[float],bool]] = None
        self.curr : Optional[Tuple[ObsType,Optional[float],bool]] = None
        self.prev_a : Optional[ActionType] = None
        self.curr_a : Optional[ActionType] = None
    def append_obs(self, obs : ObsType, reward : Optional[float], terminal : bool):
        if self.curr is not None and self.curr[2]: # If the last observation was terminal
            self.prev = None
            self.curr = None
            self.prev_a = None
            self.curr_a = None
        else:
            self.prev = self.curr
            self.prev_a = self.curr_a
        self.curr = (obs, reward, terminal)
        self.curr_a = None
    def append_action(self, action : ActionType):
        if self.curr_a is not None:
            raise Exception('`append_action` was called twice in a row without a new observation. Make sure to call `append_obs` each time time an observation is received.')
        self.curr_a = action
    def get_transition(self) -> Optional[Tuple[ObsType,ActionType,float,ObsType,bool]]:
        """ Return the most recent transition tuple.

        Returns:
            Tuple: (`s0`, `a0`, `r1`, `s1`, `t1`)

                - `s0`: Starting state
                - `a0`: Action taken at state `s0`
                - `r1`: Reward obtained from taking action `a0` at state `s0` and transitioning to `s1`
                - `s1`: State reached by taking action `a0` at state `s0`
                - `t1`: `True` if this transition is terminal, and `False` otherwise.
            If no transition is available, return None.
        """
        if self.prev is None or self.curr is None:
            return None
        if self.prev_a is None:
            return None
        s0, _, _ = self.prev
        a0 = self.prev_a
        s1, r1, t1 = self.curr
        if r1 is None:
            return None
        return s0, a0, r1, s1, t1
    def get_obs(self) -> ObsType:
        if self.curr is None:
            raise Exception('No observation available.')
        return self.curr[0]
    def state_dict(self):
        return {
                'prev': self.prev,
                'curr': self.curr,
                'prev_a': self.prev_a,
                'curr_a': self.curr_a
        }
    def load_state_dict(self,state):
        self.prev = state['prev']
        self.curr = state['curr']
        self.prev_a = state['prev_a']
        self.curr_a = state['curr_a']
ObsType = TypeVar('ObsType')
ActionType = TypeVar('ActionType')

class SACAgent(DeployableAgent):
    """
    Implementation of SAC based on [Haarnoja 2018](https://arxiv.org/pdf/1801.01290.pdf), but with added support for semi-markov decision processes (i.e. observations include the number of time steps since the last observation), and variable discount rates.
    """
    def __init__(self,
            action_space : gym.spaces.Box,
            observation_space : gym.spaces.Box,
            discount_factor : float = 0.99,       # Haarnoja 2018 Table 1 - Shared - discount
            learning_rate : float = 3e-4,         # Haarnoja 2018 Table 1 - Shared - learning rate
            batch_size : int = 256,               # Haarnoja 2018 Table 1 - Shared - number of samples per minibatch
            warmup_steps : int = 1000,            # https://github.com/haarnoja/sac - `n_initial_exploration_steps`
            replay_buffer_size : int = 1_000_000, # Haarnoja 2018 Table 1 - Shared - replay buffer size
            target_update_frequency : int = 1,
            polyak_rate : float = 0.005,          # Haarnoja 2018 Table 1 - SAC - target smoothing coefficient
            reward_scale : float = 5,             # Haarnoja 2018 Table 2 (task dependent)
            device : torch.device = torch.device('cpu'),
            q_net_1 : torch.nn.Module = None,
            q_net_2 : torch.nn.Module = None,
            v_net : torch.nn.Module = None,
            pi_net : torch.nn.Module = None):
        """
        Args:
            action_space: Gym action space.
            observation_space: Gym observation space.
            warmup_steps (int): Number of steps to take using an exploration policy. This value isn't specified in the paper, but is found in Haarnoja's code under the variable name `n_initial_exploration_steps`. The exploration policy used is a uniform distribution in the range [-1,1]. See [code](https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/policies/uniform_policy.py#L24) for details.
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_frequency = target_update_frequency
        self.polyak_rate = polyak_rate
        self.reward_scale = reward_scale
        self.device = device

        # State (training)
        self.obs_stack = defaultdict(lambda: ObservationStack())
        self._steps = 0 # Number of steps experienced
        self._training_steps = 0 # Number of training steps experienced

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        if q_net_1 is None:
            raise Exception('Models must all be provided. `q_net_1` is missing.')
        if q_net_2 is None:
            raise Exception('Models must all be provided. `q_net_2` is missing.')
        if v_net is None:
            raise Exception('Models must all be provided. `v_net` is missing.')
        if pi_net is None:
            raise Exception('Models must all be provided. `pi_net` is missing.')
        self.q_net_1 = q_net_1
        self.q_net_2 = q_net_2
        self.v_net = v_net
        self.v_net_target = copy.deepcopy(self.v_net)
        self.pi_net = pi_net

        self.optimizer_q_1 = torch.optim.Adam(self.q_net_1.parameters(), lr=learning_rate)
        self.optimizer_q_2 = torch.optim.Adam(self.q_net_2.parameters(), lr=learning_rate)
        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=learning_rate)
        self.optimizer_pi = torch.optim.Adam(self.pi_net.parameters(), lr=learning_rate)

    def observe(self, obs, reward=None, terminal=False, testing=False, time=1, discount=None, env_key=None):
        if env_key is None:
            env_key = testing
        if discount is None:
            discount = self.discount_factor

        self.obs_stack[env_key].append_obs(obs, reward, terminal)

        # Add to replay buffer
        x = self.obs_stack[env_key].get_transition()
        if x is not None and not testing:
            self._steps += 1

            obs0, action0, reward1, obs1, terminal = x
            obs0 = torch.Tensor(obs0)
            obs1 = torch.Tensor(obs1)
            transition = (obs0, action0, reward1, obs1, terminal, time, discount)
            self.replay_buffer.add(transition)

            # Train each time something is added to the buffer
            self._train()

    def _sample_action(self,
            s : torch.Tensor,
            include_log_prob : bool = False
            ) -> Mapping[str,torch.Tensor]: # python 3.8 is required for `TypedDict`, so we use `Mapping` instead.
        output = {}

        # Sample an action
        mean,log_std = self.pi_net(s)
        output['mean'] = mean
        output['log_std'] = log_std
        #standard_norm = torch.distributions.Normal(0,1)
        #z = standard_norm.sample(mean.shape)
        #action = mean+z*torch.exp(log_std)
        dist = torch.distributions.Normal(mean,torch.exp(log_std))
        action = dist.rsample()

        high = self.action_space.high
        low = self.action_space.low
        scale = torch.tensor((high-low)/2, device=self.device)
        bias = torch.tensor((high+low)/2, device=self.device)
        scaled_action = torch.tanh(action)*scale+bias
        output['action'] = scaled_action.float() # `float()` converts it to the float32 (or whatever the default float precision is)

        #weights = [p.abs().mean().item() for p in self.pi_net.parameters()]
        #tqdm.write(str(weights))
        if torch.isnan(scaled_action).any():
            breakpoint()

        # Compute log prob
        if include_log_prob:
            #output['log_prob'] = normal_log_prob(mean,log_std,action)
            log_prob = dist.log_prob(action).sum(dim=1,keepdim=True)
            log_prob -= torch.log(scale*(1-scaled_action**2) +  1e-6).sum(dim=1,keepdim=True) # See second equation of appendix C "Enforcing Action Bounds".
            output['log_prob'] = log_prob

        return output

    def _train(self, iterations=1):
        if len(self.replay_buffer) < self.batch_size*iterations:
            return
        if len(self.replay_buffer) < self.warmup_steps:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        for _,(s0,a0,r1,s1,term,time,gamma) in zip(range(iterations),dataloader):
            # Fix data types
            s0 = s0.to(self.device)
            a0 = a0.float().to(self.device)
            r1 = r1.float().to(self.device).view(-1,1)
            s1 = s1.to(self.device)
            term = term.float().to(self.device).view(-1,1)
            time = time.to(self.device).view(-1,1)
            gamma = gamma.float().to(self.device).view(-1,1)
            g_t = torch.pow(gamma,time)

            r1 *= self.reward_scale

            # Sample an action and compute its value
            sampled_a0 = self._sample_action(s0, include_log_prob=True)
            q1 = self.q_net_1(s0,sampled_a0['action'])
            q2 = self.q_net_2(s0,sampled_a0['action'])
            q = torch.min(q1,q2)

            ### Update state value function (equation 5/6) ###
            #x = self.v_net(s0)-q+sampled_a0['log_prob']
            #x = x.detach()
            #loss = (self.v_net(s0)*x).mean()
            v_pred = self.v_net(s0)
            v_target = (q-sampled_a0['log_prob']).detach()
            assert v_pred.shape == v_target.shape
            loss = 1/2*(v_pred-v_target)**2
            loss = loss.mean()
            self.optimizer_v.zero_grad()
            loss.backward()
            self.optimizer_v.step()

            ### Update state-action value function (equation 7/9) ###
            q_target = (r1 + g_t*self.v_net_target(s1)*(1-term)).detach()
            for q_net, optimizer in [(self.q_net_1, self.optimizer_q_1), (self.q_net_2, self.optimizer_q_2)]:
                #q0 = q_net(s0,a0)
                #vt = self.v_net_target(s1)
                #x = (q0-r1-g_t*vt).detach()
                q_pred = q_net(s0,a0)
                assert q_pred.shape == q_target.shape
                loss = 1/2*(q_pred-q_target)**2
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Sample an action and compute its value (resampling because the Q networks have changed)
            sampled_a0 = self._sample_action(s0, include_log_prob=True)
            q1 = self.q_net_1(s0,sampled_a0['action'])
            q2 = self.q_net_2(s0,sampled_a0['action'])
            q = torch.min(q1,q2)

            ### Update policy (equation 12/13) ###
            self.optimizer_pi.zero_grad()
            loss = sampled_a0['log_prob'] - q # This is equation 12, but OpenAI's pseudocode uses the negative of this.
            loss = loss.mean()
            loss.backward()
            self.optimizer_pi.step()

        self._training_steps += 1

        self._update_target()

    def _update_target(self):
        if self._training_steps % self.target_update_frequency != 0:
            return
        tau = self.polyak_rate
        for p1,p2 in zip(self.v_net_target.parameters(), self.v_net.parameters()):
            p1.data = (1-tau)*p1+tau*p2

    def act(self, testing=False, env_key=None):
        """ Return a random action according to an epsilon-greedy policy. """
        if env_key is None:
            env_key = testing

        obs = self.obs_stack[env_key].get_obs()
        obs = torch.tensor(obs).unsqueeze(0).float().to(self.device)

        if testing:
            action = self._sample_action(obs)['mean'] # Is this right? Does the testing code use a sample or the mean?
            action = action.flatten().detach().cpu().numpy()
        else:
            if self._steps < self.warmup_steps:
                action = np.random.uniform(-1., 1., self.action_space.shape)
            else:
                action = self._sample_action(obs)['action']
                action = action.flatten().detach().cpu().numpy()

        self.obs_stack[env_key].append_action(action)

        return action

    def state_dict(self):
        return {
                # Models
                'q_net_1': self.q_net_1.state_dict(),
                'q_net_2': self.q_net_2.state_dict(),
                'v_net': self.v_net.state_dict(),
                'v_net_target': self.v_net_target.state_dict(),
                'pi_net': self.pi_net.state_dict(),
                # Optimizers
                'optimizer_q_1': self.optimizer_q_1.state_dict(),
                'optimizer_q_2': self.optimizer_q_2.state_dict(),
                'optimizer_v': self.optimizer_v.state_dict(),
                'optimizer_pi': self.optimizer_pi.state_dict(),
                # Misc
                'obs_stack': {k:os.state_dict() for k,os in self.obs_stack.items()},
                'steps': self._steps,
                'training_steps': self._training_steps,
        }

    def load_state_dict(self, state):
        # Models
        self.q_net_1.load_state_dict(state['q_net_1'])
        self.q_net_2.load_state_dict(state['q_net_2'])
        self.v_net.load_state_dict(state['v_net'])
        self.v_net_target.load_state_dict(state['v_net_target'])
        self.pi_net.load_state_dict(state['pi_net'])
        # Optimizers
        self.optimizer_q_1.load_state_dict(state['optimizer_q_1'])
        self.optimizer_q_2.load_state_dict(state['optimizer_q_2'])
        self.optimizer_v.load_state_dict(state['optimizer_v'])
        self.optimizer_pi.load_state_dict(state['optimizer_pi'])
        # Misc
        for k,os_state in state['obs_stack'].items():
            self.obs_stack[k].load_state_dict(os_state)
        self._steps = state['steps']
        self._training_steps = state['training_steps']

    def state_dict_deploy(self):
        return {
            'action_space': self.action_space,
            'observation_space': self.observation_space,

            'q_net_1': self.q_net_1.state_dict(),
            'q_net_2': self.q_net_2.state_dict(),
            'v_net': self.v_net.state_dict(),
            'v_net_target': self.v_net_target.state_dict(),
            'pi_net': self.pi_net.state_dict(),

            'q_net_1_class': self.q_net_1.__class__,
            'q_net_2_class': self.q_net_2.__class__,
            'v_net_class': self.v_net.__class__,
            'pi_net_class': self.pi_net.__class__,
        }
    def load_state_dict_deploy(self, state):
        self.q_net_1.load_state_dict(state['q_net_1'])
        self.q_net_2.load_state_dict(state['q_net_2'])
        self.v_net.load_state_dict(state['v_net'])
        self.v_net_target.load_state_dict(state['v_net_target'])
        self.pi_net.load_state_dict(state['pi_net'])

def make_agent_from_deploy_state(state : Union[str,Mapping], device : torch.device = torch.device('cpu')):
    if isinstance(state, str): # If it's a string, then it's the filename to the dilled state
        filename = state
        if not os.path.isfile(filename):
            raise Exception('No file found at %s' % filename)
        with open(filename, 'rb') as f:
            state = dill.load(f)
    if not isinstance(state,Mapping):
        raise ValueError('State is expected to be a dictionary. Found a %s.' % type(state))

    # Q Network
    if state['q_net_1_class'] is not state['q_net_2_class']:
        raise NotImplementedError('The two Q networks use different models. Unable to handle this case.')
    q_net_cls = state['q_net_1_class']
    if q_net_cls is QNetwork:
        q_net_1 = QNetwork(num_features=state['q_net_1']['num_features'],num_actions=state['q_net_1']['num_actions']).to(device)
        q_net_2 = QNetwork(num_features=state['q_net_2']['num_features'],num_actions=state['q_net_2']['num_actions']).to(device)
    else:
        raise Exception('Unable to initialize policy network of type %s' % q_net_cls)

    # Value Network
    v_net_cls = state['v_net_class']
    if v_net_cls is VNetwork:
        v_net = VNetwork(num_features=state['v_net']['num_features']).to(device)
    else:
        raise Exception('Unable to initialize policy network of type %s' % v_net_cls)

    # Policy Network
    pi_net_cls = state['pi_net_class']
    if pi_net_cls is PolicyNetwork:
        pi_net = PolicyNetwork(num_features=state['pi_net']['num_features'],num_actions=state['pi_net']['num_actions'],structure=state['pi_net']['structure']).to(device)
    else:
        raise Exception('Unable to initialize policy network of type %s' % pi_net_cls)

    agent = SACAgent(
            action_space=state['action_space'],
            observation_space=state['observation_space'],
            q_net_1=q_net_1,
            q_net_2=q_net_2,
            v_net=v_net,
            pi_net=pi_net,
            device=device,
    )
    agent.load_state_dict_deploy(state)
    return agent

if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import torch.cuda
    import numpy as np
    import pprint
    import gym
    import gym.envs
    #import pybullet_envs
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    def make_env(env_name):
        env = gym.make(env_name)
        if isinstance(env,gym.wrappers.TimeLimit):
            env = env.env
        return env
    
    def train(envs, agent, training_steps=1_000_000, test_frequency=1000, render=False):
        test_results = {}
        env = envs[0]
        env_test = envs[1]
        plot_x = []
        plot_y = []

        done = True
        for i in tqdm(range(training_steps), desc='training'):
            if i % test_frequency == 0:
                video_file_name = os.path.join('output','video-%d.avi'%i)
                test_results[i] = [test(env_test, agent, render=(i==0 and render), video_file_name=video_file_name) for i in tqdm(range(5), desc='testing')]
                avg = np.mean([x['total_reward'] for x in test_results[i]])
                tqdm.write('Iteration {i}\t Average reward: {avg}'.format(i=i,avg=avg))
                tqdm.write(pprint.pformat(test_results[i], indent=4))
                # Plot (rewards)
                plot_file_name = os.path.join('output','plot.png')
                plot_x.append(i)
                plot_y.append(avg)
                plt.plot(plot_x,plot_y)
                plt.xlabel('Training Steps')
                plt.ylabel('Average Reward')
                plt.grid()
                plt.savefig(plot_file_name)
                plt.close()
                tqdm.write('Plot saved at %s' % os.path.abspath(plot_file_name))
                ## Plot (state value)
                #plot_file_name = os.path.join('output','plot-state-value.png')
                #x,y = agent.debug_logger['state_value']
                #plt.plot(x,[np.mean(v) for v in y])
                #plt.xlabel('Training Steps')
                #plt.ylabel('Average State Values')
                #plt.grid()
                #plt.savefig(plot_file_name)
                #plt.close()
                #tqdm.write('Plot saved at %s' % os.path.abspath(plot_file_name))
                ## Plot (state-action value)
                #plot_file_name = os.path.join('output','plot-state-action-value.png')
                #x,y = agent.debug_logger['state_action_value_1']
                #plt.plot(x,[np.mean(v) for v in y])
                #x,y = agent.debug_logger['state_action_value_2']
                #plt.plot(x,[np.mean(v) for v in y])
                #plt.xlabel('Training Steps')
                #plt.ylabel('Average State Action Values')
                #plt.grid()
                #plt.savefig(plot_file_name)
                #plt.close()
                #tqdm.write('Plot saved at %s' % os.path.abspath(plot_file_name))

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

        return {
            'total_steps': total_steps,
            'total_reward': total_reward
        }

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

    agent = SACAgent(
        action_space=env[0].action_space,
        observation_space=env[0].observation_space,
        reward_scale=5,
        q_net_1 = QNetwork(env[0].observation_space.shape[0], env[0].action_space.shape[0]).to(device),
        q_net_2 = QNetwork(env[0].observation_space.shape[0], env[0].action_space.shape[0]).to(device),
        v_net   = VNetwork(env[0].observation_space.shape[0]).to(device),
        pi_net  = PolicyNetwork(env[0].observation_space.shape[0], env[0].action_space.shape[0]).to(device),
        device=device,
    )

    results = train(env,agent)
    test(env[1], agent)

