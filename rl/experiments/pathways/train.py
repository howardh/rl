import os
from pathlib import Path
from typing import Optional
from pprint import pprint
import itertools

import torch
import numpy as np
import gym.spaces
import dill
from tqdm import tqdm
import experiment.logger
from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger

from rl.agent.smdp.a2c import A2CAgentRecurrentVec, PolicyValueNetworkRecurrent
from rl.experiments.training.vectorized import TrainExperiment, make_vec_env
from rl.experiments.training._utils import ExperimentConfigs
from rl.experiments.pathways.models import ConvPolicy
from rl.experiments.pathways.models import ModularPolicy, ModularPolicy2
#from rl.experiments.training._utils import make_env


class AttnRecAgent(A2CAgentRecurrentVec):
    def __init__(self, recurrence_type='RecurrentAttention', model_type='ModularPolicy', num_recurrence_blocks=1, **kwargs):
        self._model_type = model_type
        self._recurrence_type = recurrence_type
        self._num_recurrence_blocks = num_recurrence_blocks
        super().__init__(**kwargs)

    def _init_default_net(self, observation_space, action_space, device) -> PolicyValueNetworkRecurrent:
        if isinstance(observation_space, gym.spaces.Box):
            assert observation_space.shape is not None
            if len(observation_space.shape) == 1: # Mujoco
                raise Exception('Unsupported observation space or action space.')
            if len(observation_space.shape) == 3: # Atari
                return ConvPolicy(
                        num_actions=action_space.n,
                        input_size=512,
                        key_size=512,
                        value_size=512,
                        num_heads=8,
                        ff_size = 1024,
                        in_channels=observation_space.shape[0],
                        recurrence_type=self._recurrence_type,
                        num_blocks=self._num_recurrence_blocks,
                ).to(device)
        if isinstance(observation_space, gym.spaces.Dict):
            if 'obs' in observation_space.keys() and len(observation_space['obs'].shape) == 3:
                # Atari
                return self._init_atari_net(observation_space, action_space, device)
            elif 'obs (image)' in observation_space.keys():
                # Minigrid
                return self._init_minigrid_net(observation_space, action_space, device)
        raise Exception('Unsupported observation space or action space.')
    def _init_atari_net(self, observation_space, action_space, device):
        if self._model_type == 'ModularPolicy':
            return ModularPolicy(
                    inputs={
                        'obs': {
                            'type': 'GreyscaleImageInput',
                            'config': {
                                'in_channels': observation_space['obs'].shape[0]
                            },
                        },
                        'reward': {
                            'type': 'ScalarInput',
                        },
                    },
                    num_actions=action_space.n,
                    input_size=512,
                    key_size=512,
                    value_size=512,
                    num_heads=8,
                    ff_size = 1024,
                    recurrence_type=self._recurrence_type,
                    num_blocks=self._num_recurrence_blocks,
            ).to(device)
        elif self._model_type == 'ModularPolicy2':
            return ModularPolicy2(
                    inputs = {
                        'obs': {
                            'type': 'GreyscaleImageInput',
                            'config': {
                                'in_channels': observation_space['obs'].shape[0]
                            },
                        },
                        'reward': {
                            'type': 'ScalarInput',
                        },
                        'action': {
                            'type': 'DiscreteInput' if isinstance(action_space, gym.spaces.Discrete) else 'LinearInput',
                            'config': {
                                'input_size': action_space.n
                            },
                        },
                    },
                    outputs = {
                        'value': {
                            'type': 'LinearOutput',
                            'config': {
                                'output_size': 1,
                            }
                        },
                        'action': {
                            'type': 'LinearOutput',
                            'config': {
                                'output_size': action_space.n,
                            }
                        },
                    },
                    input_size=512,
                    key_size=512,
                    value_size=512,
                    num_heads=8,
                    ff_size = 1024,
                    recurrence_type=self._recurrence_type,
                    num_blocks=self._num_recurrence_blocks,
            ).to(device)
        raise NotImplementedError()
    def _init_minigrid_net(self, observation_space, action_space, device):
        observation_space = observation_space # Unused variable
        if self._model_type == 'ModularPolicy2':
            return ModularPolicy2(
                    inputs = {
                        'obs (image)': {
                            'type': 'ImageInput56',
                            'config': {
                                'in_channels': observation_space['obs (image)'].shape[0]
                            },
                        },
                        'reward': {
                            'type': 'ScalarInput',
                        },
                        'action': {
                            'type': 'DiscreteInput',
                            'config': {
                                'input_size': action_space.n
                            },
                        },
                    },
                    outputs = {
                        'value': {
                            'type': 'LinearOutput',
                            'config': {
                                'output_size': 1,
                            }
                        },
                        'action': {
                            'type': 'LinearOutput',
                            'config': {
                                'output_size': action_space.n,
                            }
                        },
                    },
                    input_size=512,
                    key_size=512,
                    value_size=512,
                    num_heads=8,
                    ff_size = 1024,
                    recurrence_type=self._recurrence_type,
                    num_blocks=self._num_recurrence_blocks,
            ).to(device)
        raise NotImplementedError()

    def state_dict_deploy(self):
        return {
                **super().state_dict_deploy(),
                '_model_type': self._model_type,
                'recurrence_type': self._recurrence_type,
                'num_recurrence_blocks': self._num_recurrence_blocks,
        }
    @staticmethod
    def from_deploy_state(state):
        if isinstance(state, str):
            with open(state, 'rb') as f:
                state = dill.load(f)
            return AttnRecAgent.from_deploy_state(state)
        elif isinstance(state, dict):
            agent = AttnRecAgent(
                    action_space = state['action_space'],
                    observation_space = state['observation_space'],
                    model_type=state['_model_type'],
                    recurrence_type=state.pop('recurrence_type'),
                    num_recurrence_blocks=state.pop('num_recurrence_blocks'),
                    obs_scale=state.pop('obs_scale'),
                    num_test_envs=16, # TODO
            )
            agent.load_state_dict_deploy(state)
            return agent


def get_params():
    from rl.experiments.pathways.train import AttnRecAgent as Agent # Need to import for pickling purposes

    params = ExperimentConfigs()

    def init_train_params():
        num_envs = 16
        env_name = 'Pong-v5'
        env_config = {
            'env_type': 'envpool',
            'env_configs': {
                'env_name': env_name,
                'atari': True,
                'atari_config': {
                    'num_envs': num_envs,
                    'stack_num': 1,
                    'repeat_action_probability': 0.25,
                }
            }
        }

        params.add('exp-001', {
            'agent': {
                'type': Agent,
                'parameters': {
                    'target_update_frequency': 32_000,
                    'num_train_envs': num_envs,
                    'num_test_envs': 1,
                    'max_rollout_length': 128,
                    'hidden_reset_min_prob': 0,
                    'hidden_reset_max_prob': 0,
                    'recurrence_type': 'RecurrentAttention',
                    'num_recurrence_blocks': 1,
                },
            },
            'env_test': env_config,
            'env_train': env_config,
            'test_frequency': None,
            'save_model_frequency': None,
            'verbose': True,
        })

        params.add_change('exp-002', {
            'agent': {
                'parameters': {
                    'recurrence_type': 'RecurrentAttention2',
                },
            },
        })

        params.add_change('exp-003', {
            'agent': {
                'parameters': {
                    'recurrence_type': 'RecurrentAttention3',
                },
            },
        })

        params.add_change('exp-004', {
            'agent': {
                'parameters': {
                    'recurrence_type': 'RecurrentAttention4',
                },
            },
        })

        params.add_change('exp-005', {
            'agent': {
                'parameters': {
                    'recurrence_type': 'RecurrentAttention5',
                },
            },
        })

        params.add_change('exp-006', {
            'agent': {
                'parameters': {
                    'recurrence_type': 'RecurrentAttention6',
                },
            },
        })

        params.add_change('exp-007', {
            'agent': {
                'parameters': {
                    'recurrence_type': 'RecurrentAttention7',
                },
            },
        })

        params.add_change('exp-008', {
            'agent': {
                'parameters': {
                    'recurrence_type': 'RecurrentAttention8',
                },
            },
        })

        # This architecture is working. Now reduce rollout length to see if it still works. It takes a long time to train with the current setup.
        params.add_change('exp-009', {
            'agent': {
                'parameters': {
                    'max_rollout_length': 64,
                },
            },
        }) # This works well

        params.add_change('exp-010', {
            'agent': {
                'parameters': {
                    'max_rollout_length': 32,
                },
            },
        })

        params.add_change('exp-011', {
            'agent': {
                'parameters': {
                    'max_rollout_length': 16,
                },
            },
        })

        params.add_change('exp-012', {
            'agent': {
                'parameters': {
                    'max_rollout_length': 8,
                },
            },
        })

    def init_meta_rl_params():
        # Params for meta-RL experiments
        num_envs = 16
        env_name = 'ALE/Pong-v5'
        env_config = {
            'env_type': 'gym_async',
            'env_configs': [{
                'env_name': env_name,
                'atari': True,
                'frame_stack': 1,
                'episode_stack': 5,
                'action_shuffle': False,
                'config': {
                    'frameskip': 1,
                    'mode': 0,
                    'difficulty': 0,
                    'repeat_action_probability': 0.25,
                    'full_action_space': False,
                }
            }] * num_envs
        }

        # Test episode stacking first to make sure it works.
        params.add('exp-meta-001', {
            'agent': {
                'type': Agent,
                'parameters': {
                    'target_update_frequency': 32_000,
                    'num_train_envs': num_envs,
                    'num_test_envs': 1,
                    'max_rollout_length': 128,
                    'hidden_reset_min_prob': 0,
                    'hidden_reset_max_prob': 0,
                    'recurrence_type': 'RecurrentAttention8',
                    'num_recurrence_blocks': 1,
                },
            },
            'env_test': env_config,
            'env_train': env_config,
            'test_frequency': None,
            'save_model_frequency': None,
            'verbose': True,
        })

        # Test a shorter rollout length
        params.add_change('exp-meta-002', {
            'agent': {
                'parameters': {
                    'max_rollout_length': 16,
                },
            },
        })

        # Add action shuffle
        env_config_diff = {
            'env_configs': [{
                'action_shuffle': True,
            }] * num_envs
        }
        params.add_change('exp-meta-003', {
            'env_test': env_config_diff,
            'env_train': env_config_diff,
        }) # It's learning something, but plateaus with the state value estimate hovering around -1.

        # Increase model size
        params.add_change('exp-meta-004', {
            'agent': {
                'parameters': {
                    'num_recurrence_blocks': 2,
                },
            },
        })

        env_config_diff = {
            'env_configs': [{
                'action_shuffle': [2,3,4,5],
            }] * num_envs
        }
        params.add_change('exp-meta-005', {
            'env_test': env_config_diff,
            'env_train': env_config_diff,
        })

        # Increase model size further
        params.add_change('exp-meta-006', {
            'agent': {
                'parameters': {
                    'num_recurrence_blocks': 4,
                },
            },
        })

        # Disable action shuffle for now. Use dict observations.
        env_config_diff = {
            'env_configs': [{
                'action_shuffle': False,
                'dict_obs': True,
                'episode_stack': 2, # Reduced from 5
            }] * num_envs
        }
        params.add_change('exp-meta-007', {
            'agent': {
                'parameters': {
                    'num_recurrence_blocks': 1,
                    'obs_scale': {
                        'obs': 1/255,
                    }
                },
            },
            'env_test': env_config_diff,
            'env_train': env_config_diff,
        })

        # Try with action shuffle again
        env_config_diff = {
            'env_configs': [{
                'action_shuffle': [2,3,4,5],
            }] * num_envs
        }
        params.add_change('exp-meta-008', {
            'env_test': env_config_diff,
            'env_train': env_config_diff,
        })

        # The previous version likely doesn't work because it's outputting a distribution over actions, so the agent has no idea what action was action taken from that distribution and has no way to know what action led to any given transition.
        # Added the action as an input and testing without action shuffle to make sure this works.
        env_config_diff = {
            'env_configs': [{
                'action_shuffle': False,
            }] * num_envs
        }
        params.add_change('exp-meta-009', {
            'agent': {
                'parameters': {
                    'model_type': 'ModularPolicy2',
                    'recurrence_type': 'RecurrentAttention9',
                },
            },
            'env_test': env_config_diff,
            'env_train': env_config_diff,
        }) # This model works

        # Try again with action shuffle
        env_config_diff = {
            'env_configs': [{
                'action_shuffle': [2,3,4,5],
            }] * num_envs
        }
        params.add_change('exp-meta-010', {
            'env_test': env_config_diff,
            'env_train': env_config_diff,
        })

    def init_seaquest_params():
        # Look for a set of parameters that work well for seaquest.
        num_envs = 16
        env_name = 'Seaquest-v5'
        env_config = {
            'env_type': 'envpool',
            'env_configs': {
                'env_name': env_name,
                'atari': True,
                'atari_config': {
                    'num_envs': num_envs,
                    'stack_num': 1,
                    'repeat_action_probability': 0.25,
                }
            }
        }

        params.add('exp-seaquest-001', {
            'agent': {
                'type': Agent,
                'parameters': {
                    'target_update_frequency': 32_000,
                    'num_train_envs': num_envs,
                    'num_test_envs': 1,
                    'max_rollout_length': 16,
                    'hidden_reset_min_prob': 0,
                    'hidden_reset_max_prob': 0,
                    'recurrence_type': 'RecurrentAttention8',
                    'num_recurrence_blocks': 1,
                },
            },
            'env_test': env_config,
            'env_train': env_config,
            'test_frequency': None,
            'save_model_frequency': None,
            'verbose': True,
        })

    def init_breakout_params():
        # Look for a set of parameters that work well for seaquest.
        num_envs = 16
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
                }
            }
        }

        params.add('exp-breakout-001', {
            'agent': {
                'type': Agent,
                'parameters': {
                    'target_update_frequency': 32_000,
                    'num_train_envs': num_envs,
                    'num_test_envs': 1,
                    'max_rollout_length': 16,
                    'hidden_reset_min_prob': 0,
                    'hidden_reset_max_prob': 0,
                    'recurrence_type': 'RecurrentAttention8',
                    'num_recurrence_blocks': 1,
                },
            },
            'env_test': env_config,
            'env_train': env_config,
            'test_frequency': None,
            'save_model_frequency': None,
            'verbose': True,
        }) # This is hitting the ~300 return that DQN gets, but only briefly.

        # Test with frame stacking. The A3C paper's LSTM experiments keep the frame stacking and they get returns in the 700s (See table S3 https://arxiv.org/pdf/1602.01783.pdf).
        params.add_change('exp-breakout-002', {
            'env_test': {
                'env_configs': {
                    'atari_config': {
                        'stack_num': 4,
                    }
                }
            },
            'env_train': {
                'env_configs': {
                    'atari_config': {
                        'stack_num': 4,
                    }
                }
            },
        })

        # Add hidden state forcing on the target network
        params.add_change('exp-breakout-003', {
            'agent': {
                'parameters': {
                    'target_net_hidden_state_forcing': True,
                },
            },
        })

    def init_atlantis_params():
        # Look for a set of parameters that work well for seaquest.
        num_envs = 16
        env_name = 'Atlantis-v5'
        env_config = {
            'env_type': 'envpool',
            'env_configs': {
                'env_name': env_name,
                'atari': True,
                'atari_config': {
                    'num_envs': num_envs,
                    'stack_num': 1,
                    'repeat_action_probability': 0.25,
                }
            }
        }

        params.add('exp-atlantis-001', {
            'agent': {
                'type': Agent,
                'parameters': {
                    'target_update_frequency': 32_000,
                    'num_train_envs': num_envs,
                    'num_test_envs': 1,
                    'max_rollout_length': 16,
                    'hidden_reset_min_prob': 0,
                    'hidden_reset_max_prob': 0,
                    'recurrence_type': 'RecurrentAttention8',
                    'num_recurrence_blocks': 1,
                },
            },
            'env_test': env_config,
            'env_train': env_config,
            'test_frequency': None,
            'save_model_frequency': None,
            'verbose': True,
        })

    def init_minigrid_params():
        num_envs = 16
        #env_name = 'MiniGrid-Empty-5x5-v0'
        env_name = 'MiniGrid-Empty-16x16-v0'
        env_config = {
            'env_type': 'gym_async',
            'env_configs': [{
                'env_name': env_name,
                'minigrid': True,
                'minigrid_config': {},
                'episode_stack': 1,
                'dict_obs': True,
                'action_shuffle': False,
                'config': {}
            }] * num_envs
        }

        params.add('exp-minigrid-001', {
            'agent': {
                'type': Agent,
                'parameters': {
                    'target_update_frequency': 8_000,
                    'num_train_envs': num_envs,
                    'num_test_envs': 1,
                    'obs_scale': {
                        'obs (image)': 1.0 / 255.0,
                    },
                    'max_rollout_length': 16,
                    'hidden_reset_min_prob': 0,
                    'hidden_reset_max_prob': 0,
                    'model_type': 'ModularPolicy2',
                    'recurrence_type': 'RecurrentAttention9',
                    'num_recurrence_blocks': 1,
                },
            },
            'env_test': env_config,
            'env_train': env_config,
            'test_frequency': None,
            'save_model_frequency': None,
            'verbose': True,
        })

        #env_name = 'MiniGrid-Empty-16x16-v0'
        #env_config_diff = {
        #    'env_configs': [{
        #        'env_name': env_name,
        #    }] * num_envs
        #}
        #params.add_change('exp-minigrid-002', {
        #    'env_test': env_config_diff,
        #    'env_train': env_config_diff,
        #})

    def init_minigrid_bandit_params():
        num_envs = 16
        env_name = 'MiniGrid-NRoomBanditsSmall-v0'
        env_config = {
            'env_type': 'gym_async',
            'env_configs': [{
                'env_name': env_name,
                'minigrid': True,
                'minigrid_config': {},
                'episode_stack': 5,
                'dict_obs': True,
                'action_shuffle': False,
                'config': {}
            }] * num_envs
        }

        params.add('exp-mgb-001', {
            'agent': {
                'type': Agent,
                'parameters': {
                    'target_update_frequency': 8_000,
                    'num_train_envs': num_envs,
                    'num_test_envs': 1,
                    'obs_scale': {
                        'obs (image)': 1.0 / 255.0,
                    },
                    'max_rollout_length': 16,
                    'hidden_reset_min_prob': 0,
                    'hidden_reset_max_prob': 0,
                    'model_type': 'ModularPolicy2',
                    'recurrence_type': 'RecurrentAttention9',
                    'num_recurrence_blocks': 3,
                },
            },
            'env_test': env_config,
            'env_train': env_config,
            'test_frequency': None,
            'save_model_frequency': None,
            'verbose': True,
        })

        #env_name = 'MiniGrid-Empty-16x16-v0'
        #env_config_diff = {
        #    'env_configs': [{
        #        'env_name': env_name,
        #    }] * num_envs
        #}
        #params.add_change('exp-minigrid-002', {
        #    'env_test': env_config_diff,
        #    'env_train': env_config_diff,
        #})

    init_train_params()
    init_meta_rl_params()
    init_seaquest_params()
    init_breakout_params()
    init_atlantis_params()
    init_minigrid_params()
    init_minigrid_bandit_params()
    # Breakout, Atlantis, and Skiing are envs with four actions, so they're probably more suitable for the initial action shuffling experiments
    return params


def get_test_params():
    params = ExperimentConfigs()

    params.add('test-001', {
        'env': {
            'env_type': 'gym_async',
            'env_configs': [{
                'env_name': 'ALE/Pong-v5',
                'atari': True,
                'frame_stack': 1,
                'config': {
                    'frameskip': 1,
                    'mode': 0,
                    'difficulty': 0,
                    'repeat_action_probability': 0.25,
                    'render_mode': 'rgb_array',
                }
            } for _ in range(16)],
        },
    })

    params.add('test-002', {
        'env': {
            'env_type': 'gym_async',
            'env_configs': [{
                'env_name': 'ALE/Pong-v5',
                'atari': True,
                'frame_stack': 1,
                'dict_obs': True,
                'episode_stack': 1,
                'config': {
                    'frameskip': 1,
                    'mode': 0,
                    'difficulty': 0,
                    'repeat_action_probability': 0.25,
                    'render_mode': 'rgb_array',
                }
            } for _ in range(16)],
        },
    })

    params.add_change('test-003', {
        'env': {
            'env_type': 'gym_async',
            'env_configs': [{
                'config': {
                    'full_action_space': False,
                }
            } for _ in range(16)],
        },
    })

    return params


def _run_test(env, agent, verbose=False, total_episodes=3):
    if type(env).__name__ == 'AtariGymEnvPool':
        num_envs = env.config['num_envs']
    elif type(env).__name__ == 'AsyncVectorEnv':
        num_envs = env.num_envs
    else:
        raise ValueError('Unknown env type: {}'.format(type(env).__name__))

    steps = itertools.count()
    if verbose:
        steps = tqdm(steps, desc='test episode')

    rgb_array = []
    attention = []
    ff_gate = []

    dones = torch.tensor([False] * num_envs, dtype=torch.bool)
    ep_count = torch.tensor([0] * num_envs, dtype=torch.int)
    step_count = torch.tensor([0] * num_envs, dtype=torch.int)
    total_reward = [[] for _ in range(num_envs)]
    total_steps = [[] for _ in range(num_envs)]
    ep_steps = [[] for _ in range(num_envs)]

    rewards = torch.tensor([0] * num_envs, dtype=torch.float)

    obs = env.reset()
    agent.observe(obs, testing=True)
    attention.append(agent.net.last_attention)
    ff_gate.append(agent.net.last_ff_gating)
    for step in steps:
        obs, reward, done, info = env.step(agent.act(testing=True))
        agent.observe(obs, reward, np.array([False] * num_envs), testing=True)
        attention.append(agent.net.last_attention)
        ff_gate.append(agent.net.last_ff_gating)

        done = torch.tensor(done)
        rewards += torch.tensor(reward)
        ep_count += done.long()
        step_count += 1
        dones = dones.logical_or(ep_count >= total_episodes)
        for i,d in enumerate(done):
            if not d:
                continue
            total_reward[i].append(rewards[i].item())
            ep_steps[i].append(step_count[i].item())
            total_steps[i].append(step)
            tqdm.write(f'Env {i}\t Ep {ep_count[i]-1}\t Steps {step_count[i]}\t reward: {rewards[i].item():.2f}')
        rewards[done] = 0
        step_count[done] = 0

        if 'rgb' in info:
            rgb_array.append(info[0]['rgb'])
        if dones.all():
            break
    env.close()

    return {
        'total_steps': total_steps,
        'episode_steps': ep_steps,
        'total_reward': total_reward,
        #'option_choice_history': agent._option_choice_history,
        #'option_term_history': agent._option_term_history,
        'rgb_array': rgb_array,
        'attention': attention,
        'ff_gate': ff_gate,
    }


def make_app():
    import typer
    app = typer.Typer()

    @app.command()
    def run(exp_name : str,
            trial_id : Optional[str] = None,
            results_directory : Optional[str] = None,
            max_iterations : int = 5_000_000,
            slurm : bool = typer.Option(False, '--slurm'),
            wandb : bool = typer.Option(False, '--wandb'),
            debug : bool = typer.Option(False, '--debug')):
        config = get_params()[exp_name]
        pprint(config)
        if debug:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config={
                        **config,
                        #'save_model_frequency': 100_000, # This is number of iterations, not the number of transitions experienced
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    #checkpoint_frequency=250_000,
                    checkpoint_frequency=None,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
        else:
            exp_runner = make_experiment_runner(
                    TrainExperiment,
                    config={
                        **config,
                        'save_model_frequency': 100_000, # This is number of iterations, not the number of transitions experienced
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=250_000,
                    #checkpoint_frequency=None,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
        if wandb:
            exp_runner.exp.logger.init_wandb({
                'project': f'A2C-pathways-{exp_name}'
            })
        exp_runner.run()
        exp_runner.exp.logger.finish_wandb()

    @app.command()
    def checkpoint(filename):
        exp = load_checkpoint(TrainExperiment, filename)
        exp.run()

    @app.command()
    def plot(result_directory : Path):
        import experiment.plotter as eplt
        from experiment.plotter import EMASmoothing

        checkpoint_filename = os.path.join(result_directory,'checkpoint.pkl')
        with open(checkpoint_filename,'rb') as f:
            x = dill.load(f)
        logger = Logger()
        logger.load_state_dict(x['exp']['logger'])
        if isinstance(logger.data, experiment.logger.FileBackedList):
            logger.data.iterate_past_end = True
        logger.load_to_memory(verbose=True)
        output_directory = x['exp']['output_directory']
        plot_directory = os.path.join(output_directory,'plots')
        os.makedirs(plot_directory,exist_ok=True)

        for k in ['agent_train_state_value_target_net', 'agent_train_state_value', 'train_reward', 'reward']:
            try:
                filename = os.path.abspath(os.path.join(plot_directory,f'plot-{k}.png'))
                eplt.plot(logger,
                        filename=filename,
                        curves=[{
                            'key': k,
                            'smooth_fn': EMASmoothing(0.9),
                        }],
                        min_points=2,
                        xlabel='Steps',
                        ylabel=k,
                        aggregate='mean',
                        show_unaggregated=False,
                )
                print(f'Plot saved to {filename}')
            except KeyError:
                print(f'Could not plot {k}. Key not found.')

    @app.command()
    def test(config_name : str,
            model_filename : Path,
            output_filename : Path = Path('./plot.png')):
        # Run a number of test episodes and plot the result (return vs number of episodes)
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        num_episodes=3

        config = get_test_params()[config_name]
        agent = AttnRecAgent.from_deploy_state(str(model_filename))
        env = make_vec_env(**config['env'])
        results = _run_test(env, agent, verbose=True, total_episodes=num_episodes)
        #steps = np.array([x[:num_episodes] for x in results['total_steps']])
        rewards = np.array([x[:num_episodes] for x in results['total_reward']]).mean(0)
        plt.plot(rewards)
        #for x,y in zip(results['total_steps'], results['total_reward']):
        #    plt.plot(x,y)
        #plt.show()
        plt.savefig(str(output_filename))
        print(f'Saving plot to {os.path.abspath(output_filename)}')

        breakpoint()

    #@app.command()
    #def video(config_name : str,
    #        model_filename : Path,
    #        output_filename : Path = Path('./output.avi')):
    #    # Test the given model and save a video
    #    # Include in the video: visuzliation of action choice, action values, neural network gating patterns
    #    raise NotImplementedError()

    commands = {
            'run': run,
            'checkpoint': checkpoint,
            'plot': plot,
            'test': test,
    }

    return app, commands


def run():
    app,_ = make_app()
    app()


if __name__ == "__main__":
    run()
