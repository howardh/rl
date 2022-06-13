import time
import os
import itertools
from pathlib import Path
from collections import OrderedDict
#from torch import multiprocessing as mp

from rl.experiments.pathways.models import LinearInput
from typing import Optional, Tuple
from pprint import pprint

import torch
import gym
from tqdm import tqdm
import numpy as np
import dill
import experiment.logger
from experiment import load_checkpoint, make_experiment_runner
from experiment.logger import Logger
from torch.utils.data.dataloader import default_collate

from frankenstein.buffer.vec_history import VecHistoryBuffer

from rl.experiments.training.vectorized import TrainExperiment, make_vec_env
from rl.experiments.training._utils import ExperimentConfigs
from rl.trainer.ppo import PPOTrainer, AgentVec


def get_params():
    #from rl.agent.smdp.a2c import PPOAgentRecurrentVec as AgentPPO
    from rl.experiments.pathways.train import AttnRecAgentPPO as AgentPPO

    params = ExperimentConfigs()

    num_envs = 16
    env_name = 'MiniGrid-NRoomBanditsSmall-v0'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'episode_stack': 100,
            'dict_obs': True,
            'action_shuffle': False,
            'config': {}
        }] * num_envs
    }

    params.add('exp-001', {
        'agent': {
            'type': AgentPPO,
            'parameters': {
                'num_train_envs': num_envs,
                'num_test_envs': 1,
                'obs_scale': {
                    'obs (image)': 1.0 / 255.0,
                },
                'max_rollout_length': 128,
                'model_type': 'ModularPolicy5',
                'recurrence_type': 'RecurrentAttention10',
                'architecture': [3, 3]
            },
        },
        'env_test': env_config,
        'env_train': env_config,
        'test_frequency': None,
        'save_model_frequency': None,
        'verbose': True,
    })

    # Use a different wrapper for all the meta-learning stuff. Previous version was bugged.
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 100,
                'dict_obs': True,
                'randomize': True,
            },
            'config': {
                'rewards': [1, -1],
                'shuffle_goals_on_reset': False,
            }
        }] * num_envs
    }
    params.add('exp-002', {
        'agent': {
            'type': AgentPPO,
            'parameters': {
                'num_train_envs': num_envs,
                'num_test_envs': 1,
                'obs_scale': {
                    'obs (image)': 1.0 / 255.0,
                },
                'max_rollout_length': 128,
                'model_type': 'ModularPolicy5',
                'recurrence_type': 'RecurrentAttention11',
                'architecture': [3, 3]
            },
        },
        'env_test': env_config,
        'env_train': env_config,
        'test_frequency': None,
        'save_model_frequency': None,
        'verbose': True,
    }) 

    # Add the reward permutation as part of the observation
    env_config = {
        #'env_type': 'gym_sync',
        'env_configs': [{
            'config': {
                'include_reward_permutation': True,
            }
        }] * num_envs
    }
    params.add_change('exp-003', {
        'env_test': env_config,
        'env_train': env_config,
        #'save_model_frequency': 1_000_000,
    })

    # Stochastic rewards
    env_name = 'MiniGrid-NRoomBanditsSmallBernoulli-v0'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 100,
                'dict_obs': True,
                'randomize': True,
            },
            'config': {
                'reward_scale': 1,
                'prob': 0.9,
                'shuffle_goals_on_reset': False,
                'include_reward_permutation': True,
            }
        }] * num_envs
    }
    params.add('exp-004', {
        'agent': {
            'type': AgentPPO,
            'parameters': {
                'num_train_envs': num_envs,
                'num_test_envs': 1,
                'obs_scale': {
                    'obs (image)': 1.0 / 255.0,
                },
                'max_rollout_length': 128,
                'model_type': 'ModularPolicy5',
                'recurrence_type': 'RecurrentAttention11',
                'architecture': [3, 3]
            },
        },
        'env_test': env_config,
        'env_train': env_config,
        'test_frequency': None,
        'save_model_frequency': None,
        'verbose': True,
    }) 

    # Remove reward permutation from observation
    env_config = {
        'env_configs': [{
            'config': {
                'include_reward_permutation': False,
            }
        }] * num_envs
    }
    params.add_change('exp-005', {
        'env_test': env_config,
        'env_train': env_config,
    }) 
    # This keeps getting NaNs

    # Decreasing learning rate from default (1e-4) to 1e-5
    params.add_change('exp-006', {
        'agent': {
            'parameters': {
                'learning_rate': 1e-5,
            },
        },
    }) 
    # Still diverges
    # But then it works if I keep trying.

    # Train on many different randomness settings simultaneously
    # Testing with the reward permutation provided to start
    env_name = 'MiniGrid-NRoomBanditsSmallBernoulli-v0'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 100,
                'dict_obs': True,
                'randomize': True,
            },
            'config': {
                'reward_scale': 1,
                'prob': 1-(i/num_envs)*0.5,
                'shuffle_goals_on_reset': False,
                'include_reward_permutation': True,
            }
        } for i in range(num_envs)],
    }
    params.add_change('exp-007', {
        'env_test': env_config,
        'env_train': env_config,
    }) # Not working?

    # Increase model size
    params.add_change('exp-008', {
        'agent': {
            'parameters': {
                'architecture': [6, 6],
            },
        },
    }) 

    env_name = 'MiniGrid-NRoomBanditsSmallBernoulli-v0'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 100,
                'dict_obs': True,
                'randomize': True,
            },
            'config': {
                'reward_scale': 1,
                'prob': 1-(i%2)*0.1,
                'shuffle_goals_on_reset': False,
                'include_reward_permutation': True,
            }
        } for i in range(num_envs)],
    }
    params.add_change('exp-009', {
        'env_test': env_config,
        'env_train': env_config,
    })

    # Same thing, just without the reward permutation
    env_config = {
        'env_configs': [{
            'config': {
                'include_reward_permutation': False,
            }
        }] * num_envs
    }
    params.add_change('exp-010', {
        'env_test': env_config,
        'env_train': env_config,
    })

    # Retry setup of exp-007, but with the larger model
    env_name = 'MiniGrid-NRoomBanditsSmallBernoulli-v0'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 100,
                'dict_obs': True,
                'randomize': True,
            },
            'config': {
                'reward_scale': 1,
                'prob': 1-(i/num_envs)*0.5,
                'shuffle_goals_on_reset': False,
                'include_reward_permutation': False,
            }
        } for i in range(num_envs)],
    }
    params.add_change('exp-011', {
        'env_test': env_config,
        'env_train': env_config,
    })

    # Try a smaller range of probabilities
    env_config = {
        'env_configs': [{
            'config': {
                'prob': 1-(i/num_envs)*0.8,
            }
        } for i in range(num_envs)],
    }
    params.add_change('exp-012', {
        'env_test': env_config,
        'env_train': env_config,
    })

    env_name = 'MiniGrid-BanditsFetch-v0'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
            }
        } for _ in range(num_envs)],
    }
    params.add('exp-013', {
        **params['exp-012'],
        'env_test': env_config,
        'env_train': env_config,
    })

    env_name = 'MiniGrid-BanditsFetch-v0'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'reward_incorrect': 0,
                'num_objs': 2,
            }
        } for _ in range(num_envs)],
    }
    params.add('exp-014', {
        **params['exp-012'],
        'env_test': env_config,
        'env_train': env_config,
    })

    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'size': 5,
                'num_trials': 100,
                'reward_correct': 1,
                'reward_incorrect': -1,
                'num_objs': 2,
                'num_obj_types': 2,
                'num_obj_colors': 1,
                'unique_objs': True,
                'include_reward_permutation': True,
            }
        } for _ in range(num_envs)],
    }
    params.add('exp-015', {
        **params['exp-012'],
        'env_test': env_config,
        'env_train': env_config,
    })

    env_config = {
        'env_configs': [{
            'config': {
                'include_reward_permutation': False,
            }
        } for _ in range(num_envs)],
    }
    params.add_change('exp-016', {
        'env_test': env_config,
        'env_train': env_config,
    })

    env_config = {
        'env_configs': [{
            'config': {
                'num_obj_types': 1,
                'num_obj_colors': 2,
            }
        } for _ in range(num_envs)],
    }
    params.add_change('exp-017', {
        'env_test': env_config,
        'env_train': env_config,
    })

    # Independent core modules
    params.add_change('exp-018', {
        'agent': {
            'parameters': {
                'model_type': 'ModularPolicy5',
                'recurrence_type': 'RecurrentAttention14',
                'architecture': [3, 3]
            },
        },
    })

    # Try a faster learning rate
    params.add_change('exp-019', {
        'agent': {
            'parameters': {
                'learning_rate': 1e-3,
            },
        },
    })

    # Decreased rate for value function
    params.add_change('exp-020', {
        'agent': {
            'parameters': {
                'vf_loss_coeff': 0.05,
            },
        },
    }) # Converges too fast

    # Try a lower learning rate. Lower variance in the updates might also lead to faster learning.
    params.add_change('exp-021', {
        'agent': {
            'parameters': {
                'learning_rate': 1e-5,
                'vf_loss_coeff': 0.5,
            },
        },
    })

    # Train on a larger variety of objects
    env_config = {
        'env_configs': [{
            'config': {
                'num_obj_types': 2,
                'num_obj_colors': 2,
            }
        } for _ in range(num_envs)],
    }
    params.add('exp-022', {
        'env_test': env_config,
        'env_train': env_config,
    }, inherit='exp-018')

    # Train on a larger environment
    env_config = {
        'env_configs': [{
            'config': {
                'num_obj_types': 2,
                'num_obj_colors': 2,
                'num_objs': 2,
                'size': [5,6,7,8][i%4]
            }
        } for i in range(num_envs)],
    }
    params.add('exp-023', {
        'env_test': env_config,
        'env_train': env_config,
    }, inherit='exp-018')
    # Maybe this was too big of a jump

    # Train on larger environments, but fewer of them
    for j in [6,7,8]:
        env_config = {
            'env_configs': [{
                'config': {
                    'num_obj_types': 2,
                    'num_obj_colors': 2,
                    'num_objs': 2,
                    'size': list(range(5,j+1))[i%(j+1-5)]
                }
            } for i in range(num_envs)],
        }
        params.add(f'exp-023-{j}', {
            'env_test': env_config,
            'env_train': env_config,
        }, inherit='exp-018')
    # It's not learning size 7+ properly, despite the reward plot looking good

    # Train on a single size only
    for j in [6,7,8]:
        env_config = {
            'env_configs': [{
                'config': {
                    'num_obj_types': 2,
                    'num_obj_colors': 2,
                    'num_objs': 2,
                    'size': j,
                }
            } for _ in range(num_envs)],
        }
        params.add(f'exp-024-{j}', {
            'env_test': env_config,
            'env_train': env_config,
        }, inherit='exp-018')

    env_name = 'MiniGrid-BanditsFetch-v0'
    env_name2 = 'MiniGrid-MultiRoom-v0'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'size': s,
                'num_trials': 100,
                'reward_correct': 1,
                'reward_incorrect': -1,
                'num_objs': 2,
                'num_obj_types': 2,
                'num_obj_colors': 6,
                'unique_objs': True,
                'include_reward_permutation': False,
            }
        } for s in [5,6,7,8]] + [{
            'env_name': env_name2,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 2,
                'max_num_rooms': 2,
                'max_room_size': 4,
            }
        } for _ in range(num_envs-4)],
    }
    params.add('exp-025', {
        **params['exp-018'],
        'env_test': env_config,
        'env_train': env_config,
    })

    env_name = 'MiniGrid-MultiRoom-v1'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 2,
                'max_num_rooms': 2,
                'min_room_size': 5,
                'max_room_size': 8,
                'fetch_config': {
                    'num_objs': 2,
                }
            }
        } for _ in range(num_envs//2)] + [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 2,
                'max_num_rooms': 2,
                'min_room_size': 5,
                'max_room_size': 8,
                'bandits_config': {
                    'probs': [1,0],
                }
            }
        } for _ in range(num_envs//2)]
    }
    params.add('exp-026', {
        **params['exp-018'],
        'env_test': env_config,
        'env_train': env_config,
    })

    # Modified to include single room tasks
    env_name = 'MiniGrid-MultiRoom-v1'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 1,
                'max_num_rooms': 2,
                'min_room_size': 5,
                'max_room_size': 8,
                'fetch_config': {
                    'num_objs': 2,
                }
            }
        } for _ in range(num_envs//2)] + [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 1,
                'max_num_rooms': 2,
                'min_room_size': 5,
                'max_room_size': 8,
                'bandits_config': {
                    'probs': [1,0],
                }
            }
        } for _ in range(num_envs//2)]
    }
    params.add('exp-027', {
        **params['exp-018'],
        'env_test': env_config,
        'env_train': env_config,
    })

    # BanditsFetch and MultiRoom with the same task are different enough that the agent can't generalize. Train it on both simultaneously.
    env_name = 'MiniGrid-BanditsFetch-v0'
    env_name2 = 'MiniGrid-MultiRoom-v1'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'size': s,
                'num_trials': 100,
                'reward_correct': 1,
                'reward_incorrect': -1,
                'num_objs': 2,
                'num_obj_types': 2,
                'num_obj_colors': 6,
                'unique_objs': True,
                'include_reward_permutation': False,
            }
        } for s in [5,6,7,8]]*2 + [{
            'env_name': env_name2,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 1,
                'max_num_rooms': 1,
                'max_room_size': 8,
                'fetch_config': {
                    'num_objs': 2,
                },
            }
        } for _ in range(num_envs-8)],
    }
    params.add('exp-028', {
        **params['exp-018'],
        'env_test': env_config,
        'env_train': env_config,
    })

    env_name = 'MiniGrid-MultiRoom-v1'
    env_config = {
        'env_type': 'gym_async',
        'env_configs': [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 1,
                'dict_obs': True,
                'randomize': False,
            },
            'config': {
                'num_trials': 100,
                'min_num_rooms': 1,
                'max_num_rooms': 1,
                'min_room_size': 5,
                'max_room_size': 5,
                'fetch_config': {
                    'num_objs': 2,
                    'num_obj_colors': 2,
                },
            }
        } for _ in range(num_envs)],
    }
    params.add('exp-029', {
        **params['exp-018'],
        'env_test': env_config,
        'env_train': env_config,
    })

    return params


class PPOTrainer2(PPOTrainer):
    def __init__(self, core_module_targets, **kwargs):
        super().__init__(**kwargs)
        self.core_module_targets = core_module_targets

    def train_steps(self):
        history = VecHistoryBuffer(
                num_envs = self.batch_size,
                max_len=self.rollout_length+1,
                device=self.device)
        agent = AgentVec(
                observation_space=self.env.observation_space,
                action_space=self.env.single_action_space,
                model=self.model,
                batch_size=self.batch_size,
        )
        start_time = time.time()
        start_steps = self._steps

        obs = self.env.reset()
        history.append_obs(obs)
        agent.observe(obs)
        episode_reward = np.zeros(self.batch_size)
        episode_steps = np.zeros(self.batch_size)
        while True:
            # Check if we've trained for long enough
            if self._steps >= self.num_steps:
                break

            core_targets = [[] for _ in self.core_module_targets]

            # Gather data
            for i in range(self.rollout_length):
                with torch.no_grad():
                    action = agent.act()
                obs, reward, done, info = self.env.step(action)
                for ct_idx, target_fn in enumerate(self.core_module_targets):
                    core_targets[ct_idx].append(target_fn(
                        action, obs, reward, done, info))
                agent.observe(obs, done)

                history.append_action(action)
                episode_reward += reward
                episode_steps += 1

                reward *= self.reward_scale
                if self.reward_clip is not None:
                    reward = np.clip(reward, *self.reward_clip)

                history.append_obs(obs, reward, done)

                if done.any():
                    tqdm.write(f'{self._steps:,}\t reward: {episode_reward[done].mean():.2f}\t len: {episode_steps[done].mean()}')
                    self.logger.log(
                            reward = episode_reward[done].mean().item(),
                            episode_length = episode_steps[done].mean().item(),
                            step = self._steps + i*self.batch_size,
                    )
                    episode_reward[done] = 0
                    episode_steps[done] = 0
            core_targets = [default_collate(ct) for ct in core_targets]

            # Train
            losses = self._compute_losses(history)
            for x in losses:
                core_target_loss = []
                for i,targ in enumerate(core_targets):
                    targ['key'] = targ['key'].to(self.device)
                    targ['value'] = targ['value'].to(self.device)
                    targ['mask'] = targ['mask'].to(self.device)
                    attn_output = { # torch.Size([129, 3, 16, 512])
                            'key': x['output']['misc']['core_output']['key'][:-1,i,...], # type: ignore
                            'value': x['output']['misc']['core_output']['value'][:-1,i,...], # type: ignore
                    }
                    key_mse = (attn_output['key'] - targ['key']) ** 2
                    val_mse = (attn_output['value'] - targ['value']) ** 2
                    mask = torch.logical_not(targ['mask'])
                    core_target_loss.append(
                            torch.masked_select(key_mse, mask.unsqueeze(2)).mean() +
                            torch.masked_select(val_mse, mask.unsqueeze(2)).mean()
                    )
                core_target_loss = torch.stack(core_target_loss).mean()
                self.optimizer.zero_grad()
                loss = x['loss'] + core_target_loss
                loss.backward()
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
                    last_loss_core=core_target_loss.item(), # type: ignore
                    last_attn_output_key=x['output']['misc']['core_output']['key'].abs().mean().item(), # type: ignore
                    last_attn_output_value=x['output']['misc']['core_output']['value'].abs().mean().item(), # type: ignore
                    last_attn_output_query=x['output']['misc']['core_output']['x'].abs().mean().item(), # type: ignore
                    #last_approx_kl=approx_kl.item(), # type: ignore
                    #learning_rate=self.lr_scheduler.get_lr()[0], # type: ignore
                    step=self._steps,
            )
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Timing
            elapsed_time = time.time() - start_time
            steps_per_sec = (self._steps - start_steps) / elapsed_time
            remaining_time = int((self.num_steps - self._steps) / steps_per_sec)
            remaining_hours = remaining_time // 3600
            remaining_minutes = (remaining_time % 3600) // 60
            remaining_seconds = (remaining_time % 3600) % 60
            tqdm.write(f"Step {self._steps:,}/{self.num_steps:,} \t Remaining: {remaining_hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}")

            yield


def test_episode(agent, env, i=0):
    results = {}
    agent.reset()
    obs = env.reset()
    done = np.array([False] * env.num_envs)
    agent.observe(obs, testing=True)

    print(f'Trial {i}')
    results['total_reward'] = []
    results['reward'] = []
    total_reward = 0
    for _ in tqdm(itertools.count()):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        agent.observe(obs, reward, done, testing=True)

        total_reward += reward[0]
        results['total_reward'].append(total_reward)
        results['reward'].append(reward[0])
        if done.any():
            tqdm.write('-'*80)
            if 'regret' in info[0]:
                results['regret'] = info[0]['regret']
            break
    return results


def make_app():
    import typer
    app = typer.Typer()

    @app.command()
    def run(exp_name : str,
            trial_id : Optional[str] = None,
            results_directory : Optional[str] = None,
            max_iterations : int = 5_000_000,
            starting_model: Optional[str] = None,
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
                        #'save_model_frequency': 100_000, # This is number of iterations, not the number of transitions experienced
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=25_000,
                    #checkpoint_frequency=None,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
        if wandb:
            exp_runner.exp.logger.init_wandb({
                'project': f'PPO-minigrid-bandits-{exp_name}'
            })

        if starting_model is not None:
            print(f'Starting training from pretrained weights: {starting_model}')
            try:
                loaded_data = torch.load(starting_model)
            except:
                with open(starting_model, 'rb') as f:
                    loaded_data = dill.load(f)
            if 'exp' in loaded_data:
                exp_runner.exp.agent.net.load_state_dict(loaded_data['exp']['agent']['net'])
            else:
                exp_runner.exp.agent.net.load_state_dict(loaded_data)

        exp_runner.run()
        exp_runner.exp.logger.finish_wandb()

    @app.command()
    def run2(trial_id : Optional[str] = None,
            results_directory : Optional[str] = None,
            max_iterations : int = 5_000_000,
            slurm : bool = typer.Option(False, '--slurm'),
            wandb : bool = typer.Option(False, '--wandb'),
            debug : bool = typer.Option(False, '--debug')):
        from rl.experiments.pathways.train import AttnRecAgentPPO as AgentPPO
        from rl.experiments.pathways.models import ModularPolicy5

        # Environment
        num_envs = 16
        env_name = 'MiniGrid-NRoomBanditsSmall-v0'
        env_config = {
            'env_type': 'gym_async',
            'env_configs': [{
                'env_name': env_name,
                'minigrid': True,
                'minigrid_config': {},
                'meta_config': {
                    'episode_stack': 100,
                    'dict_obs': True,
                    'randomize': True,
                },
                'config': {
                    'rewards': [1, -1],
                    'shuffle_goals_on_reset': False,
                    'include_reward_permutation': True,
                }
            }] * num_envs
        }
        env = make_vec_env(**env_config)
        if isinstance(env, gym.vector.VectorEnv):
            observation_space = env.single_observation_space
            action_space = env.single_action_space
        else:
            observation_space = env.observation_space
            action_space = env.action_space

        # Pretrained module
        reward_permutation_module = LinearInput(
            key_size=512,
            value_size=512,
            input_size = observation_space['obs (reward_permutation)'].shape[0]
        )
        reward_permutation_module.load_state_dict(torch.load('reward_perm.pt'))
        for param in reward_permutation_module.parameters():
            param.requires_grad = False

        # Agent
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
            'obs (reward_permutation)': {
                'type': None,
                'module': reward_permutation_module,
            },
        }
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
        }
        model_params = {
            'inputs': inputs,
            'outputs': outputs,
            'input_size': 512,
            'key_size': 512,
            'value_size': 512,
            'num_heads': 8,
            'ff_size': 1024,
            'recurrence_type': 'RecurrentAttention11',
            'architecture': [3, 3],
        }
        config = {
            'agent': {
                'type': AgentPPO,
                'parameters': {
                    'num_train_envs': num_envs,
                    'num_test_envs': 1,
                    'obs_scale': {
                        'obs (image)': 1.0 / 255.0,
                    },
                    'max_rollout_length': 128,
                    'net': ModularPolicy5(**model_params),
                },
            },
            'env_test': env_config,
            'env_train': env_config,
            'test_frequency': None,
            'save_model_frequency': None,
            'verbose': True,
        }
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
                        #'save_model_frequency': 100_000, # This is number of iterations, not the number of transitions experienced
                    },
                    results_directory=results_directory,
                    trial_id=trial_id,
                    checkpoint_frequency=25_000,
                    #checkpoint_frequency=None,
                    max_iterations=max_iterations,
                    slurm_split=slurm,
                    verbose=True,
                    modifiable=True,
            )
        if wandb:
            exp_runner.exp.logger.init_wandb({
                'project': f'PPO-minigrid-bandits'
            })
        exp_runner.run()
        exp_runner.exp.logger.finish_wandb()

    @app.command()
    def run3(max_iterations : int = 50_000_000,
            wandb : bool = typer.Option(False, '--wandb')):
        from rl.experiments.pathways.models import ModularPolicy5

        # Environment
        num_envs = 16
        env_name = 'MiniGrid-NRoomBanditsSmall-v0'
        env_config = {
            'env_type': 'gym_async',
            'env_configs': [{
                'env_name': env_name,
                'minigrid': True,
                'minigrid_config': {},
                'meta_config': {
                    'episode_stack': 100,
                    'dict_obs': True,
                    'randomize': True,
                },
                'config': {
                    'rewards': [1, -1],
                    'shuffle_goals_on_reset': False,
                    'include_reward_permutation': False,
                }
            }] * num_envs
        }
        env = make_vec_env(**env_config)
        if isinstance(env, gym.vector.VectorEnv):
            observation_space = env.single_observation_space
            action_space = env.single_action_space
        else:
            observation_space = env.observation_space
            action_space = env.action_space

        # Pretrained module
        reward_permutation_module = LinearInput(
            key_size=512,
            value_size=512,
            #input_size = observation_space['obs (reward_permutation)'].shape[0]
            input_size = 2,
        )
        reward_permutation_module.load_state_dict(torch.load('reward_perm.pt'))

        def make_input_config():
            inputs = {
                'obs (image)': {
                    'type': 'ImageInput56',
                    'config': {
                        'in_channels': observation_space['obs (image)'].shape[0],
                        'scale': 1.0 / 255.0,
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
            }
            if 'obs (reward_permutation)' in observation_space.keys():
                inputs['obs (reward_permutation)'] = {
                    'type': 'LinearInput',
                    'config': {
                        'input_size': observation_space['obs (reward_permutation)'].shape[0]
                    }
                }
            return inputs
        def make_output_config():
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
            }
            return outputs
        model = ModularPolicy5(
            inputs = make_input_config(),
            outputs = make_output_config(),
            input_size = 512,
            key_size = 512,
            value_size = 512,
            num_heads = 8,
            ff_size = 1024,
            recurrence_type = 'RecurrentAttention11',
            architecture = [3,3],
        )
        def target_fn(action=None, obs=None, reward=None, done=None, info=None):
            action = action
            obs = obs
            reward = reward
            done = done
            info = info
            indices = [i for i in range(len(info)) if 'reward_permutation' in info[i].keys()]
            mask = torch.zeros(len(info))
            key = torch.zeros(len(info), 512)
            value = torch.zeros(len(info), 512)
            if len(indices) == 0:
                return {
                    'key': key,
                    'value': value,
                    'mask': mask,
                }
            else:
                reward_permutation = torch.tensor([info[i]['reward_permutation'] for i in indices])
                output = reward_permutation_module(reward_permutation.float())
                key[indices, :] = output['key']
                value[indices, :] = output['value']
                return {
                    'key': key,
                    'value': value,
                    'mask': mask,
                }
        core_module_targets = [
                target_fn
        ]
        optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
        trainer = PPOTrainer2(
            env = env,
            model = model,
            optimizer = optimizer,
            lr_scheduler = None,
            discount = 0.99,
            num_steps = max_iterations,
            reward_clip=(-1,1),
            rollout_length = 128,
            vf_loss_coeff = 0.5,
            entropy_loss_coeff = 0.01,
            max_grad_norm = 0.5,
            norm_adv = True,
            gae_lambda = 0.95,
            use_recurrence = True,
            num_epochs = 1,
            #minibatch_size = 256,
            #num_minibatches = 4*128*num_envs//256,
            core_module_targets = core_module_targets,
            wandb = {'project': f'PPO-minigrid-bandits'} if wandb else None,
        )
        trainer.train()

    @app.command()
    def run_impala(max_iterations : int = 200_000_000,
            results_directory : str = None,
            checkpoint_frequency: int = 1_000_000,
            starting_model: Optional[str] = None,
            wandb : bool = typer.Option(False, '--wandb')):
        from experiment.experiment import get_experiment_directories

        from rl.utils import get_results_root_directory
        from rl.agent.impala import ImpalaTrainer
        from rl.experiments.pathways.models import ModularPolicy5
        from rl.experiments.training._utils import make_env

        # Directories
        directories = get_experiment_directories(
                root_directory = get_results_root_directory(),
                results_directory = results_directory,
                experiment_name = 'impala-minigrid-bandits',
                trial_id = None,
        )

        # Environment
        num_envs = 24
        env_name = 'MiniGrid-NRoomBanditsSmallBernoulli-v0'
        env_config = [{
            'env_name': env_name,
            'minigrid': True,
            'minigrid_config': {},
            'meta_config': {
                'episode_stack': 100,
                'dict_obs': True,
                'randomize': True,
            },
            'config': {
                'reward_scale': 1,
                'prob': 0.9,
                'shuffle_goals_on_reset': False,
                'include_reward_permutation': True,
            }
        }] * num_envs
        env = make_env(**env_config[0])
        observation_space = env.observation_space
        action_space = env.action_space
        assert observation_space is not None
        assert action_space is not None

        def make_input_config(observation_space, action_space):
            inputs = {
                'obs (image)': {
                    'type': 'ImageInput56',
                    'config': {
                        'in_channels': observation_space['obs (image)'].shape[0],
                        'scale': 1.0 / 255.0,
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
            }
            if 'obs (reward_permutation)' in observation_space.keys():
                inputs['obs (reward_permutation)'] = {
                    'type': 'LinearInput',
                    'config': {
                        'input_size': observation_space['obs (reward_permutation)'].shape[0]
                    }
                }
            return inputs
        def make_output_config(action_space):
            outputs = {
                'state_value': {
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
            }
            return outputs
        model = ModularPolicy5(
            inputs = make_input_config(observation_space, action_space),
            outputs = make_output_config(action_space),
            input_size = 512,
            key_size = 512,
            value_size = 512,
            num_heads = 8,
            ff_size = 1024,
            recurrence_type = 'RecurrentAttention11',
            architecture = [3,3],
        )

        if starting_model:
            with open(starting_model, 'rb') as f:
                checkpoint = dill.load(f)
            if 'exp' in checkpoint.keys():
                model_state_dict = OrderedDict([
                        (k.replace('output_modules.value.','output_modules.state_value.'), v*0.9)
                        for k,v in checkpoint['exp']['agent']['net'].items()
                ])
            else:
                raise NotImplementedError('Starting model file format not supported.')
            model.load_state_dict(model_state_dict, strict=False)

        def make_buffer_space(observation_space):
            buffer_obs_space = {
                'type': 'dict',
                'data': {
                    'obs (image)': {
                        'type': 'box',
                        'shape': observation_space['obs (image)'].shape,
                        'dtype': torch.uint8,
                    },
                    'reward': {
                        'type': 'box',
                        'shape': (1,),
                        'dtype': torch.float32,
                    },
                    'action': {
                        'type': 'box',
                        'shape': (),
                        'dtype': torch.uint8,
                    },
                }
            }
            if 'obs (reward_permutation)' in observation_space.keys():
                buffer_obs_space['data']['obs (reward_permutation)'] = {
                    'type': 'box',
                    'shape': observation_space['obs (reward_permutation)'].shape,
                    'dtype': torch.float32,
                }
            buffer_act_space = {
                'type': 'discrete'
            }
            return buffer_obs_space, buffer_act_space
        buffer_obs_space, buffer_act_space = make_buffer_space(observation_space)
        #optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
        trainer = ImpalaTrainer(
            env_configs = env_config,
            net = model, # type: ignore
            #optimizer = optimizer,
            learning_rate=1e-1,
            lr_scheduler = None,
            buffer_obs_space=buffer_obs_space,
            buffer_act_space=buffer_act_space,
            num_train_loop_workers = 2,
            num_buffers_per_env = 2,
            batch_size = 16,
            discount_factor = 0.99,
            reward_clip=(-1,1),
            max_rollout_length = 128,
            vf_loss_coeff = 0.5*0.5,
            entropy_loss_coeff = 0.01,
            max_grad_norm = 0.5,
            total_steps=max_iterations,
            ignore_obs = ['obs (mission)','obs (direction)','done'],
            wandb = {'project': f'IMPALA-minigrid-bandits'} if wandb else None,
        )
        trainer.train(
                results_directory = directories['results'],
                checkpoint_frequency = checkpoint_frequency,
        )

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
    def test(
            checkpoint_filename: Path,
            env_name: str = 'MiniGrid-NRoomBanditsSmall-v0',
            output: Path = None,
            num_trials: int = 10,
            reward_permutation: bool = False,
            reward_config: Tuple[float,float] = (1, -1),
            reward_scale: float = 1.0,
            prob: float = 0.9):
        import matplotlib
        if output is not None:
            matplotlib.use('Agg')
        import matplotlib.transforms
        from matplotlib import pyplot as plt

        #episode_stack = 100
        episode_stack = 5

        exp = load_checkpoint(TrainExperiment, checkpoint_filename)

        num_steps = exp._steps * exp.exp.agent.num_training_envs
        if env_name == 'MiniGrid-NRoomBanditsSmall-v0':
            env = make_vec_env(
                env_type = 'gym_sync',
                env_configs = [{
                    'env_name': env_name,
                    'minigrid': True,
                    'minigrid_config': {},
                    'meta_config': {
                        'episode_stack': episode_stack,
                        'dict_obs': True,
                        'randomize': False,
                    },
                    'config': {
                        'rewards': reward_config,
                        'shuffle_goals_on_reset': False,
                        'include_reward_permutation': reward_permutation,
                    }
                }]
            )
        elif env_name == 'MiniGrid-NRoomBanditsSmallBernoulli-v0':
            env = make_vec_env(
                env_type = 'gym_sync',
                env_configs = [{
                    'env_name': env_name,
                    'minigrid': True,
                    'minigrid_config': {},
                    'meta_config': {
                        'episode_stack': episode_stack,
                        'dict_obs': True,
                        'randomize': False,
                    },
                    'config': {
                        'reward_scale': reward_scale,
                        'prob': prob,
                        'shuffle_goals_on_reset': False,
                        'include_reward_permutation': reward_permutation,
                    }
                }]
            )
        else:
            raise ValueError(f'Unknown env_name: {env_name}')

        ##################################################
        # Bandit algorithms
        # Test with plain bandit algorithms for comparison

        # UCB
        if env_name == 'MiniGrid-NRoomBanditsSmall-v0':
            rewards = reward_config
            arms = [
                torch.distributions.Categorical(probs=torch.tensor([1,0])),
                torch.distributions.Categorical(probs=torch.tensor([0,1])),
            ]
        elif env_name == 'MiniGrid-NRoomBanditsSmallBernoulli-v0':
            rewards = [-reward_scale, reward_scale]
            arms = [
                torch.distributions.Categorical(probs=torch.tensor([prob, 1-prob])),
                torch.distributions.Categorical(probs=torch.tensor([1-prob, prob])),
            ]
        mean_return = [ (np.array(rewards) * a.probs.numpy()).sum() for a in arms ]
        max_return = max(mean_return)
        mean_regret = max_return - np.array(mean_return).mean()

        regret = []
        #confidence = 0.05
        #logd = -np.log(confidence)
        for _ in tqdm(range(num_trials*100), desc='UCB'):
            regret.append([])
            total = [0. for _ in arms]
            count = [0 for _ in arms]

            for t in range(episode_stack):
                if min(count) == 0:
                    action = np.array(count).argmin()
                else:
                    action = np.array([
                        x/n + np.sqrt(2*np.log(t)/n)
                        for x,n in zip(total,count)
                    ]).argmax()
                
                r = arms[action].sample().item()
                total[action] += rewards[r]
                count[action] += 1
                regret[-1].append(max_return - mean_return[action])

        ##################################################
        # Test agent

        agent = exp.exp.agent
        results = {}

        results['total_reward'] = []
        results['reward'] = []
        results['regret'] = []
        agent = exp.exp.agent

        agent.reset()
        obs = env.reset()
        done = np.array([False] * env.num_envs)
        agent.observe(obs, testing=True)
        for i in range(num_trials):
            print(f'Trial {i}')
            results['total_reward'].append([])
            results['reward'].append([])
            total_reward = 0
            for _ in tqdm(itertools.count()):
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                agent.observe(obs, reward, done, testing=True)

                total_reward += reward[0]
                results['total_reward'][-1].append(total_reward)
                results['reward'][-1].append(reward[0])
                #print(f'{len(results["total_reward"][-1])} {action} {reward} {total_reward}')
                #if obs['done'].any():
                #    tqdm.write(str(env.envs[0].reward_permutation)) # type: ignore
                #    tqdm.write('ep done')
                if done.any():
                    tqdm.write('-'*80)
                    if 'regret' in info[0]:
                        results['regret'].append(info[0]['regret'])
                    break
        #ctx = mp.get_context('fork')
        #with ctx.Pool(2) as pool:
        #    results = pool.starmap(test_episode, zip([agent]*2, [env]*2, range(2)))
        #breakpoint()

        ##################################################
        # Plot

        ## Boxplot
        #data = [[],[]]
        #for i in range(num_trials):
        #    r = np.array(results['reward'][i])
        #    nzr = r[r.nonzero()]
        #    halfway_index = len(nzr)//2
        #    mean_r_1 = np.mean(nzr[:halfway_index])
        #    mean_r_2 = np.mean(nzr[halfway_index:])
        #    if not np.isfinite(mean_r_1) or not np.isfinite(mean_r_2):
        #        breakpoint()
        #        continue
        #    data[0].append(mean_r_1)
        #    data[1].append(mean_r_2)
        #plt.boxplot(data, labels=['1st half','2nd half'])
        #plt.ylabel('Average Rewards')
        #rp = env.envs[0].reward_permutation # type: ignore
        #plt.title(f'{num_trials} Trials on {str(rp)} after {num_steps:,} steps', wrap=True)
        #if output is not None:
        #    plt.savefig(output)
        #    print(f'Saved to {os.path.abspath(output)}')
        #else:
        #    plt.show()

        # Regret
        ucb_regret = np.array(regret).mean(0)
        agent_regret = np.array(results['regret']).mean(0)

        fig, ax = plt.subplots(1,1)
        line1, = ax.plot(range(len(ucb_regret)), ucb_regret.cumsum(), label='UCB')
        line2, = ax.plot(range(len(agent_regret)), agent_regret.cumsum(), label='agent')
        ax.plot([0, len(ucb_regret)-1], [mean_regret, (len(ucb_regret)+1)*mean_regret], 'k--', label='uniform random')
        # Don't scale with the uniform random plot (https://python.tutorialink.com/make-matplotlib-autoscaling-ignore-some-of-the-plots/)
        ax.dataLim = matplotlib.transforms.Bbox.unit()
        ax.dataLim.update_from_data_xy(np.vstack(line1.get_data()).T, ignore=False)
        ax.dataLim.update_from_data_xy(np.vstack(line2.get_data()).T, ignore=False)
        ax.autoscale_view()

        ax.grid()
        ax.set_ylabel('Regret')
        ax.set_xlabel('Steps')
        ax.legend()
        rp = env.envs[0].reward_permutation # type: ignore
        ax.set_title(f'Evaluation on {str(rp)} after {num_steps:,} steps', wrap=True)
        if output is not None:
            fig.savefig(output)
            print(f'Saved to {os.path.abspath(output)}')
        else:
            fig.show()

        breakpoint()

    @app.command()
    def video(checkpoint_filename : Path,
            env_name: str = 'MiniGrid-NRoomBanditsSmall-v0',
            reward_config: Tuple[float,float] = (1, -1),
            reward_scale: float = 1,
            prob: float = 0.9,
            num_objs: int = 2,
            num_obj_types: int = 2,
            num_obj_colors: int = 1,
            num_rooms: int = 2,
            max_room_size: int = 4,
            size: int = 5):
        import cv2
        import PIL.Image, PIL.ImageDraw, PIL.ImageFont
        from fonts.ttf import Roboto # type: ignore

        num_trials = 1
        exp = load_checkpoint(TrainExperiment, checkpoint_filename)
        desc_fn = lambda: None
        if env_name == 'MiniGrid-NRoomBanditsSmall-v0':
            env = make_vec_env(
                env_type = 'gym_sync',
                env_configs = [{
                    'env_name': env_name,
                    'minigrid': True,
                    'minigrid_config': {},
                    'meta_config': {
                        'episode_stack': 100,
                        'dict_obs': True,
                        'randomize': False,
                    },
                    'config': {
                        'rewards': reward_config,
                        'shuffle_goals_on_reset': False,
                        'include_reward_permutation': False,
                    }
                }]
            )
        elif env_name == 'MiniGrid-NRoomBanditsSmallBernoulli-v0':
            env = make_vec_env(
                env_type = 'gym_sync',
                env_configs = [{
                    'env_name': env_name,
                    'minigrid': True,
                    'minigrid_config': {},
                    'meta_config': {
                        'episode_stack': 100,
                        'dict_obs': True,
                        'randomize': False,
                    },
                    'config': {
                        'reward_scale': reward_scale,
                        'prob': prob,
                        'shuffle_goals_on_reset': False,
                        'include_reward_permutation': False,
                    }
                }]
            )
        elif env_name == 'MiniGrid-BanditsFetch-v0':
            env = make_vec_env(
                env_type = 'gym_sync',
                env_configs = [{
                    'env_name': env_name,
                    'minigrid': True,
                    'minigrid_config': {},
                    'meta_config': {
                        'episode_stack': 1,
                        'dict_obs': True,
                        'randomize': False,
                    },
                    'config': {
                        'size': size,
                        'num_trials': 100,
                        'num_objs': num_objs,
                        'num_obj_types': num_obj_types,
                        'num_obj_colors': num_obj_colors,
                        'unique_objs': True,
                    }
                }]
            )
            desc_fn = lambda: f'{env.envs[0].targetColor} {env.envs[0].targetType}' # type: ignore
        elif env_name == 'MiniGrid-MultiRoom-v0':
            env = make_vec_env(
                env_type = 'gym_sync',
                env_configs = [{
                    'env_name': env_name,
                    'minigrid': True,
                    'minigrid_config': {},
                    'meta_config': {
                        'episode_stack': 1,
                        'dict_obs': True,
                        'randomize': False,
                    },
                    'config': {
                        'num_trials': 100,
                        'min_num_rooms': num_rooms,
                        'max_num_rooms': num_rooms,
                        'max_room_size': max_room_size,
                    }
                }]
            )
            desc_fn = lambda: None
        elif env_name == 'MiniGrid-MultiRoom-v1':
            env = make_vec_env(
                env_type = 'gym_sync',
                env_configs = [{
                    'env_name': env_name,
                    'minigrid': True,
                    'minigrid_config': {},
                    'meta_config': {
                        'episode_stack': 1,
                        'dict_obs': True,
                        'randomize': False,
                    },
                    'config': {
                        'num_trials': 100,
                        'min_num_rooms': num_rooms,
                        'max_num_rooms': num_rooms,
                        'max_room_size': max_room_size,
                        'fetch_config': {
                            'num_objs': num_objs,
                            'num_obj_types': num_obj_types,
                            'num_obj_colors': num_obj_colors,
                        },
                        #'bandits_config': {
                        #    'probs': [0.9, 0.1]
                        #},
                    }
                }]
            )
            desc_fn = lambda: f'{env.envs[0].targetColor} {env.envs[0].targetType}' # type: ignore
        else:
            try:
                env = make_vec_env(
                    env_type = 'gym_sync',
                    env_configs = [{
                        'env_name': env_name,
                        'minigrid': True,
                        'minigrid_config': {},
                        'meta_config': {
                            'episode_stack': 10,
                            'dict_obs': True,
                            'randomize': False,
                        },
                        'config': {}
                    }]
                )
            except:
                raise ValueError(f'Unknown env_name: {env_name}')

        def concat_images(images, padding=0, direction='h', align=0):
            if direction == 'h':
                width = sum([i.size[0] for i in images]) + padding * (len(images) + 1)
                height = max([i.size[1] for i in images]) + padding*2
                new_image = PIL.Image.new('RGB', (width, height), color=(255,255,255))
                x = 0
                for i in images:
                    new_image.paste(i, (x+padding, (height - 2*padding - i.size[1]) // 2 * (align + 1) + padding))
                    x += i.size[0] + padding
                return new_image
            elif direction == 'v':
                width = max([i.size[0] for i in images]) + padding*2
                height = sum([i.size[1] for i in images]) + padding * (len(images) + 1)
                new_image = PIL.Image.new('RGB', (width, height), color=(255,255,255))
                y = 0
                for i in images:
                    new_image.paste(i, ((width - 2*padding - i.size[0]) // 2 * (align + 1) + padding, y + padding))
                    y += i.size[1] + padding
                return new_image
            else:
                raise ValueError('direction must be "h" or "v"')

        def draw_attention(core_attention, query_gating, output_attention):
            block_size = 24
            padding = 2

            core_images = []
            for layer in core_attention:
                num_blocks, _, num_inputs = layer.shape
                width = num_inputs*block_size + (num_inputs+1)*padding
                height = num_blocks*block_size + (num_blocks+1)*padding
                img = PIL.Image.new('RGB', (width, height), color=(255,255,255))
                for i in range(num_blocks):
                    for j in range(num_inputs):
                        weight = layer[i,0,j].item()
                        c = int(255*(1-weight))
                        x = j*(block_size+padding) + padding
                        y = i*(block_size+padding) + padding
                        PIL.ImageDraw.Draw(img).rectangle(
                                (x,y,x+block_size,y+block_size),
                                fill=(c,c,c),
                        )
                core_images.append(img)
            core_imags_concat = concat_images(core_images, padding=padding, direction='v', align=1)

            num_layers = len(query_gating)
            max_layer_size = max(layer.shape[0] for layer in query_gating)
            width = num_layers*block_size + (num_layers+1)*padding
            height = max_layer_size*block_size + (max_layer_size+1)*padding
            query_image = PIL.Image.new('RGB', (width, height), color=(255,255,255))
            for i, layer in enumerate(query_gating):
                num_blocks = layer.shape[0]
                for j in range(num_blocks):
                    weight = layer[j,0].item()
                    c = int(255*(1-weight))
                    x = i*(block_size+padding) + padding
                    y = j*(block_size+padding) + padding
                    PIL.ImageDraw.Draw(query_image).rectangle(
                            (x,y,x+block_size,y+block_size),
                            fill=(c,c,c)
                    )

            output_images = {}
            for k, layer in output_attention.items():
                layer = layer.squeeze()
                num_inputs = len(layer)
                width = num_inputs*block_size + (num_inputs+1)*padding
                height = block_size + 2*padding
                img = PIL.Image.new('RGB', (width, height), color=(255,255,255))
                for i in range(num_inputs):
                    weight = layer[i].item()
                    c = int(255*(1-weight))
                    x = i*(block_size+padding) + padding
                    y = padding
                    PIL.ImageDraw.Draw(img).rectangle(
                            (x,y,x+block_size,y+block_size),
                            fill=(c,c,c)
                    )
                output_images[k] = img

            font_family = Roboto
            font_size = 18
            font = PIL.ImageFont.truetype(font_family, font_size)
            text_images = {}
            for k in output_attention.keys():
                text_width, text_height = font.getsize(k)
                img = PIL.Image.new('RGB',
                        (text_width+2*padding, text_height+2*padding),
                        color=(255,255,255))
                draw = PIL.ImageDraw.Draw(img)
                draw.fontmode = 'L' # type: ignore
                draw.text(
                        (padding, padding),
                        k,
                        font=font,
                        fill=(0,0,0)
                )
                text_images[k] = img

            output_images_concat = concat_images(
                    [
                        concat_images(
                            [
                                text_images[k],
                                output_images[k],
                            ],
                            padding = padding,
                            direction='v',
                            align=-1,
                        )
                        for k in output_images.keys()],
                    padding=padding, direction='v'
            )

            all_images_concat = concat_images(
                    [
                        core_imags_concat,
                        query_image,
                        output_images_concat,
                    ],
                    padding=padding, direction='h'
            )

            return all_images_concat

        agent = exp.exp.agent
        results = {}
        fps = 25

        results['agent'] = []
        results['reward'] = []
        agent = exp.exp.agent
        for i in range(num_trials):
            video_writer = cv2.VideoWriter( # type: ignore
                    f'video-{i}.webm',
                    cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                    fps,
                    (env.envs[0].unwrapped.width*32, env.envs[0].unwrapped.height*32), # type: ignore
            )
            video_writer2 = cv2.VideoWriter( # type: ignore
                    f'video2-{i}.webm',
                    cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                    fps,
                    env.envs[0].observation_space['obs (image)'].shape[1:], # type: ignore
            )
            video_writer3 = None
            num_frames = 0

            results['agent'].append([])
            results['reward'].append([])
            agent.reset()
            obs = env.reset()
            description = desc_fn()
            print(description)
            done = np.array([False] * env.num_envs)
            agent.observe(obs, testing=True)

            frame = env.envs[0].render(mode=None) # type: ignore
            video_writer.write(frame[:,:,::-1])
            video_writer2.write(np.moveaxis(obs['obs (image)'].squeeze(), 0, 2)[:,:,::-1])
            while not done[0]:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                agent.observe(obs, reward, done, testing=True)
                results['reward'][-1].append(reward[0])
                num_frames += 1
                print(f'{num_frames} {action} {reward} {desc_fn()}')
                frame = env.envs[0].render(mode=None) # type: ignore
                video_writer.write(frame[:,:,::-1])
                video_writer2.write(np.moveaxis(obs['obs (image)'].squeeze(), 0, 2)[:,:,::-1])
                attn_img = draw_attention(
                        core_attention = agent.net.last_attention,
                        query_gating = agent.net.last_ff_gating,
                        output_attention = agent.net.last_output_attention)
                if video_writer3 is None:
                    video_writer3 = cv2.VideoWriter( # type: ignore
                            f'video3-{i}.webm',
                            cv2.VideoWriter_fourcc(*'VP80'), # type: ignore
                            fps,
                            attn_img.size,
                    )
                video_writer3.write(np.array(attn_img)[:,:,::-1])
                #if num_frames > 100:
                #    break
            video_writer.release()
            video_writer2.release()
            if video_writer3 is not None:
                video_writer3.release()
            print(f'Trial {i} total reward: {np.sum(results["reward"][-1])}')
            print(env.envs[0].reward_permutation) # type: ignore
            print(description) # type: ignore
            breakpoint()

    commands = {
            'run': run,
            'run2': run2,
            'run3': run3,
            'run_impala': run_impala,
            'checkpoint': checkpoint,
            'plot': plot,
            'test': test,
            'video': video,
    }

    return app, commands


def run():
    app,_ = make_app()
    app()


if __name__ == "__main__":
    run()
