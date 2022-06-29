import sys
import itertools
import time

import torch
import torch.distributed
import gym
import gym.spaces


def trainer():
    print('Starting trainer')
    torch.distributed.init_process_group(
            backend='gloo',
            rank=0,
            world_size=world_size,
            store=store,
    )

    assert torch.distributed.is_initialized()
    print('Initialized')

    # Init dummy env to get the obs/action space
    env = gym.make('ALE/Pong-v5')
    obs_space = env.observation_space
    action_space = env.action_space

    obs = torch.empty(obs_space.shape, dtype=torch.int8)
    reward = torch.empty([], dtype=torch.float)
    done = torch.empty([], dtype=torch.bool)
    if isinstance(action_space, gym.spaces.Discrete):
        action = torch.empty([], dtype=torch.long)
    elif isinstance(action_space, gym.spaces.Box):
        assert action_space.shape is not None
        action = torch.empty(action_space.shape)
    else:
        raise NotImplementedError()

    actor_rank = 1
    start_time = time.time()
    #while True:
    for step in itertools.count():
        #print('receiving tensor')
        torch.distributed.recv(obs, actor_rank)
        torch.distributed.recv(reward, actor_rank)
        torch.distributed.recv(done, actor_rank)
        torch.distributed.recv(action, actor_rank)

        # Compute steps per second
        if step % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_second = step / elapsed
            print(f'Step {step} ({steps_per_second:.2f} steps/sec)')


def actor(rank, world_size):
    TRAINER_RANK = 0

    print(f'Starting actor {rank}')
    torch.distributed.init_process_group(
            backend='gloo',
            rank=rank,
            world_size=world_size,
            store=store,
    )

    assert torch.distributed.is_initialized()
    print('Initialized')

    env = gym.make('ALE/Pong-v5')

    done = True
    action = env.action_space.sample()
    while True:
        if done:
            obs = env.reset()
            reward = 0.
            done = False
        else:
            obs, reward, done, _ = env.step(action)
        action = env.action_space.sample()
        torch.distributed.send(torch.tensor(obs), TRAINER_RANK)
        torch.distributed.send(torch.tensor(reward), TRAINER_RANK)
        torch.distributed.send(torch.tensor(done), TRAINER_RANK)
        torch.distributed.send(torch.tensor(action), TRAINER_RANK)


if __name__ == "__main__":
    world_size = 2
    store = torch.distributed.FileStore(
        #'/home/mila/h/huanghow/scratch/gloo_test.pid',
        '/home/ml/users/hhuang63/gloo_test.pid',
        world_size,
    )

    args = sys.argv[1:]
    if args[0] == 'trainer':
        trainer()
    elif args[0] == 'actor':
        actor(int(args[1]), world_size)
