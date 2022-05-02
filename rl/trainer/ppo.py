from typing import Generator, Optional, Dict

from tqdm import tqdm
from torchtyping import TensorType
import gym
import gym.spaces
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

from frankenstein.buffer.vec_history import VecHistoryBuffer
from frankenstein.advantage.gae import generalized_advantage_estimate 
from frankenstein.loss.policy_gradient import clipped_advantage_policy_gradient_loss
from experiment.logger import Logger

from rl.experiments.training.vectorized import make_vec_env
from rl.agent.smdp.a2c import compute_ppo_losses_recurrent
from rl.agent.impala import to_tensor
from rl.experiments.training._utils import zip2


def compute_ppo_losses_recurrent(
        history : VecHistoryBuffer,
        model : torch.nn.Module,
        initial_hidden : TensorType,
        discount : float,
        gae_lambda : float,
        norm_adv : bool,
        clip_vf_loss : Optional[float],
        entropy_loss_coeff : float,
        vf_loss_coeff : float,
        target_kl : Optional[float],
        num_epochs : int) -> Generator[Dict[str,TensorType],None,None]:
    """
    Compute the losses for PPO.
    """
    obs = history.obs
    action = history.action
    reward = history.reward
    terminal = history.terminal

    n = len(history.obs_history)

    with torch.no_grad():
        net_output = []
        hidden = initial_hidden
        for o,t in zip2(obs,terminal):
            hidden = tuple([
                torch.where(t, init_h, h)
                for init_h,h in zip(initial_hidden,hidden)
            ])
            no = model(o,hidden)
            hidden = no['hidden']
            net_output.append(no)
        net_output = default_collate(net_output)
        state_values_old = net_output['value'].squeeze(2)
        action_dist = torch.distributions.Categorical(logits=net_output['action'][:n-1])
        log_action_probs_old = action_dist.log_prob(action)

        # Advantage
        advantages = generalized_advantage_estimate(
                state_values = state_values_old[:n-1,:],
                next_state_values = state_values_old[1:,:],
                rewards = reward[1:,:],
                terminals = terminal[1:,:],
                discount = discount,
                gae_lambda = gae_lambda,
        )
        returns = advantages + state_values_old[:n-1,:]

        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(num_epochs):
        net_output = []
        hidden = initial_hidden
        for o,t in zip2(obs,terminal):
            hidden = tuple([
                torch.where(t, init_h, h)
                for init_h,h in zip(initial_hidden,hidden)
            ])
            no = model(o,hidden)
            hidden = no['hidden']
            net_output.append(no)
        net_output = default_collate(net_output)

        assert 'value' in net_output
        assert 'action' in net_output
        state_values = net_output['value'].squeeze()
        action_dist = torch.distributions.Categorical(logits=net_output['action'][:n-1])
        log_action_probs = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        with torch.no_grad():
            logratio = log_action_probs - log_action_probs_old
            ratio = logratio.exp()
            approx_kl = ((ratio - 1) - logratio).mean()

        # Policy loss
        pg_loss = clipped_advantage_policy_gradient_loss(
                log_action_probs = log_action_probs,
                old_log_action_probs = log_action_probs_old,
                advantages = advantages,
                terminals = terminal[:n-1],
                epsilon=0.1
        ).mean()

        # Value loss
        if clip_vf_loss is not None:
            v_loss_unclipped = (state_values[:n-1] - returns) ** 2
            v_clipped = state_values_old[:n-1] + torch.clamp(
                state_values[:n-1] - state_values_old[:n-1],
                -clip_vf_loss,
                clip_vf_loss,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((state_values[:n-1] - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - entropy_loss_coeff * entropy_loss + v_loss * vf_loss_coeff

        yield {
                'loss': loss,
                'loss_pi': pg_loss,
                'loss_vf': v_loss,
                'loss_entropy': -entropy_loss,
                'approx_kl': approx_kl,
        }

        if target_kl is not None:
            if approx_kl > target_kl:
                break


def get_env_batch_size(env):
    if isinstance(env, gym.vector.AsyncVectorEnv) or isinstance(env, gym.vector.SyncVectorEnv):
        return env.num_envs
    elif type(env).__name__ == 'AtariGymEnvPool':
        return env.all_env_ids.shape[0]
    else:
        raise NotImplementedError()


class PPOTrainer():
    def __init__(self,
            env,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            lr_scheduler = None,
            discount: float = 0.99,
            reward_scale: float = 1,
            reward_clip: float = None,
            rollout_length: int = 32,
            num_steps: int = 1_000,
            vf_loss_coeff : float = 0.5,
            entropy_loss_coeff : float = 0.01,
            max_grad_norm : float = 0.5,
            gae_lambda: float = 0.95,
            num_epochs : int = 5,
            clip_vf_loss : float = 0.1,
            norm_adv : bool = True,
            target_kl : float = None,
            device: torch.device = torch.device('cpu'),
            wandb: dict = None,
            ) -> None:
        config = locals()
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = get_env_batch_size(env)

        self.model.to(device)

        self.num_steps = num_steps
        self._steps = 0

        self.discount = discount
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.rollout_length = rollout_length
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.clip_vf_loss = clip_vf_loss
        self.norm_adv = norm_adv
        self.target_kl = target_kl
        self.device = device

        self.logger = Logger(key_name='step', allow_implicit_key=True)
        self.logger.log(step=0)
        if wandb is not None:
            self.logger.init_wandb({'config': config, **wandb})

    def train_steps(self):
        history = VecHistoryBuffer(
                num_envs = self.batch_size,
                max_len=self.rollout_length+1,
                device=self.device)
        agent = AgentVec(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                model=self.model,
                batch_size=self.batch_size,
        )

        obs = self.env.reset()
        history.append_obs(obs)
        agent.observe(obs)
        episode_reward = np.zeros(self.batch_size)
        episode_steps = np.zeros(self.batch_size)
        while True:
            # Check if we've trained for long enough
            tqdm.write(str(self._steps))
            if self._steps >= self.num_steps:
                break

            # Gather data
            for i in range(self.rollout_length):
                action = agent.act()
                obs, reward, done, info = self.env.step(action)

                history.append_action(action)
                episode_reward += reward
                episode_steps += 1

                reward *= self.reward_scale
                if self.reward_clip is not None:
                    reward = np.clip(reward, *self.reward_clip)

                history.append_obs(obs, reward, done)

                if done.any():
                    if 'lives' in info:
                        done = info['lives'] == 0
                if done.any():
                    self.logger.log(
                            reward = episode_reward[done].mean().item(),
                            episode_length = episode_steps[done].mean().item(),
                            step = self._steps + i*self.batch_size,
                    )
                    episode_reward[done] = 0
                    episode_steps[done] = 0

            # Train
            losses = compute_ppo_losses_recurrent(
                    history=history,
                    model=self.model,
                    initial_hidden=self.model.init_hidden(self.batch_size), # type: ignore
                    discount=self.discount,
                    gae_lambda=self.gae_lambda,
                    norm_adv=self.norm_adv,
                    clip_vf_loss=self.clip_vf_loss,
                    entropy_loss_coeff=self.entropy_loss_coeff,
                    vf_loss_coeff=self.vf_loss_coeff,
                    target_kl=self.target_kl,
                    num_epochs = self.num_epochs,
            )
            for x in losses:
                self.optimizer.zero_grad()
                x['loss'].backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Clear data
            history.clear()
            # Log
            self._steps += self.rollout_length*self.batch_size
            self.logger.log(
                    last_loss_pi=x['loss_pi'].item(), # type: ignore
                    last_loss_v=x['loss_vf'].item(), # type: ignore
                    last_loss_entropy=x['loss_entropy'].item(), # type: ignore
                    last_loss_total=x['loss'].item(), # type: ignore
                    #last_approx_kl=approx_kl.item(), # type: ignore
                    #learning_rate=self.scheduler.get_lr()[0], # type: ignore
                    step=self._steps,
            )
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            yield

    def train(self):
        for _ in tqdm(self.train_steps()):
            pass

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            '_steps': self._steps,
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self._steps = state['_steps']


def make_action_fn(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return lambda x: int(x)
    elif isinstance(action_space, gym.spaces.Box):
        return lambda x: x
    else:
        raise NotImplementedError


class AgentVec():
    def __init__(self, observation_space, action_space, model, batch_size=None):
        self.observation_space = observation_space
        self.action_space = action_space

        self.model = model
        self.batch_size = batch_size

        self.hidden = model.init_hidden(batch_size)
        self.prev_obs = None

    @property
    def device(self):
        return self.model.parameters().__next__().device

    def observe(self, obs, done=None):
        self.prev_obs = obs
        if done is not None and done.any():
            self.hidden = torch.where(
                    done,
                    self.model.init_hidden(self.batch_size),
                    self.hidden)

    def act_dist(self):
        obs = to_tensor(self.prev_obs, device=self.device)
        output = self.model(obs, self.hidden)
        self.hidden = output['hidden']

        # Get action
        if isinstance(self.action_space, gym.spaces.Discrete):
            assert 'action' in output
            action_probs_unnormalized = output['action']
            action_probs = action_probs_unnormalized.softmax(1)#.squeeze()
            assert (torch.abs(action_probs.sum(1)-1) < 1e-6).all()
            action_dist = torch.distributions.Categorical(action_probs)
            return action_dist
        elif isinstance(self.action_space, gym.spaces.Box):
            assert 'action_mean' in output
            assert 'action_std' in output
            action_dist = torch.distributions.Normal(output['action_mean'],output['action_std'])
            return action_dist
        else:
            raise NotImplementedError('Unsupported action space of type %s' % type(self.action_space))

    def act(self):
        action_dist = self.act_dist()
        action = action_dist.sample().cpu().numpy()
        return action

    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'hidden': self.hidden,
            'prev_obs': self.prev_obs,
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'])
        self.hidden = state['hidden']
        self.prev_obs = state['prev_obs']


def main2():
    from rl.agent.smdp.a2c import layer_init

    class Model(torch.nn.Module):
        def __init__(self, num_actions):
            super().__init__()
            self.num_actions = num_actions
            self.conv = torch.nn.Sequential(
                layer_init(torch.nn.Conv2d(
                    in_channels=4,out_channels=32,kernel_size=8,stride=4)),
                torch.nn.ReLU(),
                layer_init(torch.nn.Conv2d(
                    in_channels=32,out_channels=64,kernel_size=4,stride=2)),
                torch.nn.ReLU(),
                layer_init(torch.nn.Conv2d(
                    in_channels=64,out_channels=64,kernel_size=3,stride=1)),
                torch.nn.ReLU(),
            )
            self.fc = torch.nn.Sequential(
                layer_init(torch.nn.Linear(in_features=64*7*7,out_features=512)),
                torch.nn.ReLU(),
            )
            self.v = layer_init(torch.nn.Linear(in_features=512,out_features=1),std=1)
            self.pi = layer_init(torch.nn.Linear(in_features=512,out_features=num_actions),std=0.01)
        def forward(self, x, h):
            x = x.float()/255.0
            x = self.conv(x)
            x = x.view(-1,64*7*7)
            x = self.fc(x)
            v = self.v(x)
            pi = self.pi(x)
            return {
                    'value': v,
                    'action': pi, # Unnormalized action probabilities
                    'hidden': h,
            }
        def init_hidden(self, batch_size):
            batch_size = batch_size
            return ()

    if torch.cuda.is_available():
        print('Using GPU')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')

    num_envs = 8
    env_name = 'Breakout-v5'
    env_config = {
        'env_type': 'envpool',
        'env_configs': {
            'env_name': env_name,
            'atari': True,
            'atari_config': {
                'num_envs': num_envs,
                'stack_num': 4,
                'repeat_action_probability': 0.25,
                'episodic_life': True,
                #'reward_clip': True,
            }
        }
    }
    env = make_vec_env(**env_config)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    model = Model(env.action_space.n)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=1e-7,
            total_iters=1_000_000)
    trainer = PPOTrainer(
        env = env,
        model = model,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
        discount = 0.99,
        num_steps = 10_000_000,
        reward_clip=(-1,1),
        rollout_length = 128,
        max_grad_norm = 0.5,
        gae_lambda = 0.95,
        device = device,
        wandb = {'project': f'PPOTrainer-{env_name}'},
    )
    trainer.train()


def main():
    from rl.experiments.pathways.models import ModularPolicy5
    num_envs = 8
    env_name = 'Breakout-v5'
    env_config = {
        'env_type': 'envpool',
        'env_configs': {
            'env_name': env_name,
            'atari': True,
            'atari_config': {
                'num_envs': num_envs,
                'stack_num': 1,
                'repeat_action_probability': 0.25,
                'episodic_life': True,
                #'reward_clip': True,
            }
        }
    }
    env = make_vec_env(**env_config)
    model = ModularPolicy5()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = PPOTrainer(
        env = env,
        model = model,
        optimizer = optimizer,
        wandb = {'project': f'PPOTrainer-{env_name}'},
    )
    trainer.train()


if __name__=='__main__':
    main2()
