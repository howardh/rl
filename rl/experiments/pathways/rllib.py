from typing import List, Union, Dict

import numpy as np
import gym
import gym.spaces
import torch
from torch.utils.data.dataloader import default_collate
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
#from ray.rllib.agents.a3c import A2CTrainer
from ray.tune.registry import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension

from rl.experiments.pathways.models import ConvPolicy
from rl.experiments.training._utils import make_env

class CustomModel(RecurrentNetwork, torch.nn.Module):
    def __init__(self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str):
        torch.nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        assert isinstance(obs_space, gym.spaces.Box)
        assert isinstance(action_space, gym.spaces.Discrete)
        assert obs_space.shape is not None
        self.model = ConvPolicy(
            num_actions=action_space.n,
            input_size=512,
            key_size=512,
            value_size=512,
            num_heads=8,
            ff_size = 1024,
            in_channels=obs_space.shape[0],
            recurrence_type='RecurrentAttention8',
            num_blocks=3,
        )
        self._last_value = None
        self._last_action = None
        self._last_hidden = None
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Union[TensorType, List[TensorType]]:
        """Adds time dimension to batch before sending inputs to forward_rnn().
        You should implement forward_rnn() in your subclass."""
        flat_inputs = input_dict["obs_flat"].float()
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            #flat_inputs,
            input_dict['obs'].float(),
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state
    def forward_rnn(self,
            inputs: TensorType,
            state: List[TensorType],
            seq_lens: TensorType
            ) -> Union[TensorType, List[TensorType]]:
        seq_lens = seq_lens # XXX: Just getting rid of the complaints from pyright
        h = [s.squeeze(2).transpose(1,0) for s in state] # FIXME: Ray is adding an extra dimension and swapping the dimensions somewhere. Not sure how to fix this properly.
        outputs = []
        for i in range(seq_lens.max().item()):
            x = inputs[:,i]
            y = self.model(x, h)
            outputs.append(y)
            assert 'hidden' in y
            h = y['hidden']
        outputs = default_collate(outputs)
        self._last_action = outputs['action']
        self._last_value = outputs['value']
        self._last_hidden = outputs['hidden']
        next_state = [s.transpose(1,0).unsqueeze(2) for s in h] # Reshape to be the same shape as `state`
        return self._last_action, next_state
    def value_function(self) -> TensorType:
        assert self._last_value is not None
        return self._last_value.reshape(-1)
    def get_initial_state(self) -> List[np.ndarray]:
        return [h.cpu().numpy() for h in self.model.init_hidden()]

if __name__ == "__main__":
    ModelCatalog.register_custom_model("conv_model", CustomModel)

    env_config = {
        'env_name': 'ALE/Breakout-v5',
        'atari': True,
        'frame_stack': 4,
        'episode_stack': 5,
        'action_shuffle': False,
        'config': {
            'frameskip': 1,
            'mode': 0,
            'difficulty': 0,
            'repeat_action_probability': 0.25,
            'full_action_space': False,
        }
    }
    trainer_config = {
        'env': 'test_env2',
        #'env': 'ALE/Breakout-v5',
        #'env': 'pathways_env',
        'env_config': env_config,
        'lr': 2.5e-4,
        'framework': 'torch',
        'preprocessor_pref': None,
        #'num_workers': 1,
        'num_gpus': 1,
        'model': {
            #'framestack': False,
            #'dim': 84
            'custom_model': 'conv_model',
        }
    }
    trainer_config_default_ppo = { # https://github.com/ray-project/rl-experiments/blob/master/atari-ppo/2020-01-21/atari-ppo.yaml
        'env': 'ALE/Breakout-v5',
        #'use_pytorch': True,   # <- switch on/off torch
        'framework': 'torch',
        'lambda': 0.95,
        'kl_coeff': 0.5,
        'clip_rewards': True,
        'clip_param': 0.1,
        'vf_clip_param': 10.0,
        'entropy_coeff': 0.01,
        'train_batch_size': 5000,
        #'sample_batch_size': 100, # Renamed to rollout_fragment_length (See https://github.com/ray-project/ray/commit/dd7072057863f4064f0ffaba463accf4a458be1d)
        'rollout_fragment_length': 100,
        'sgd_minibatch_size': 500,
        'num_sgd_iter': 10,
        #'num_workers': 10,
        'num_envs_per_worker': 5,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'vf_share_layers': True,
        'num_gpus': 1,
    }
    trainer_config_recurrent_ppo = {
        'env': 'ALE/Breakout-v5',
        'framework': 'torch',
        'lambda': 0.95,
        'kl_coeff': 0.5,
        'clip_rewards': True,
        'clip_param': 0.1,
        'vf_clip_param': 10.0,
        'entropy_coeff': 0.01,
        'train_batch_size': 5000,
        'rollout_fragment_length': 100,
        'sgd_minibatch_size': 500,
        'num_sgd_iter': 10,
        #'num_workers': 10,
        'num_envs_per_worker': 5,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'vf_share_layers': True,
        'num_gpus': 1,
        'model': {
            # TODO
        }
    }

    register_env('pathways_env', make_env)
    register_env('test_env', lambda config: gym.make('ALE/Breakout-v5', **config))
    register_env('test_env2', lambda config: make_env(**config))

    #trainer = PPOTrainer(env='test_env2', config=trainer_config)
    #trainer.train()

    ray.init(num_cpus=8, num_gpus=1)
    tune.run(
        PPOTrainer,
        stop={'timesteps_total': 25_000_000},
        #config=trainer_config,
        config=trainer_config_default_ppo,
        checkpoint_at_end=True,
        checkpoint_freq=50_000,
        #restore= ''
        callbacks=[WandbLoggerCallback(
            project='Pathways-rllib',
            #api_key_file="/path/to/file",
            log_config=True)]
    )
