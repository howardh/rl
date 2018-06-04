import gym

gym.envs.register(id='MountainCarSparse-v0',
        entry_point='envs.mountain_car:MountainCarEnv',
        max_episode_steps=200)

ENV_NAME = "MountainCarSparse-v0"
MIN_REWARD = 0
MAX_REWARD = 1
LEARNED_REWARD = 0.5
