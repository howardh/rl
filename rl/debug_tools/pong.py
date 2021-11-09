import os
import requests

import torch
import torch.nn
import appdirs
from tqdm import tqdm

MODEL_WEIGHTS_URL_DQN = 'https://github.com/howardh/pretrained-models/raw/31df5434b925ad8506910daa05ae229ba3709658/dqn-pong-v4.pt'
MODEL_WEIGHTS_URL_A2C = 'https://github.com/howardh/pretrained-models/raw/d36a69ba588d0aaab6a11e5a3bf6e8d9158554de/dqn-pong-v4.pt'

class QNetworkCNN(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.LeakyReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=512,out_features=num_actions),
        )
    def forward(self, obs):
        x = obs
        x = self.conv(x)
        x = x.view(-1,7*7*64)
        x = self.fc(x)
        return x

class A2CNetworkCNN(torch.nn.Module):
    """ A model which acts both as a policy network and a state-value estimator. """
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.LeakyReLU(),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
        )
        self.v = torch.nn.Linear(in_features=512,out_features=1)
        self.pi = torch.nn.Linear(in_features=512,out_features=num_actions)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,64*7*7)
        x = self.fc(x)
        v = self.v(x)
        pi = self.pi(x)
        return {
                'value': v,
                'action': pi, # Unnormalized action probabilities
        }

def _download_model(url, output_filename, overwrite=False):
    if os.path.isfile(output_filename) and not overwrite:
        return
    print(f'Downloading {url} to {os.path.abspath(output_filename)}')
    # https://stackoverflow.com/a/37573701/382388
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(output_filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def get_pretained_model(model='dqn'):
    download_directory = appdirs.user_cache_dir('rl_debug_tools')
    os.makedirs(download_directory, exist_ok=True)
    if model == 'dqn':
        model_filename = os.path.join(download_directory,'dqn-pong-v4.pt')
        _download_model(MODEL_WEIGHTS_URL_DQN, model_filename, overwrite=False)
        state_dict = torch.load(model_filename)
        net = QNetworkCNN(num_actions = len(state_dict['fc.2.bias']))
        net.load_state_dict(state_dict)
        return net
    elif model == 'a2c':
        model_filename = os.path.join(download_directory,'a2c-pong-v4.pt')
        _download_model(MODEL_WEIGHTS_URL_A2C, model_filename, overwrite=False)
        state_dict = torch.load(model_filename)
        net = A2CNetworkCNN(num_actions = len(state_dict['pi.bias']))
        net.load_state_dict(state_dict)
        return net
    else:
        raise ValueError(f'Invalid model type: {model}')

def test_dqn_pong():
    import itertools
    import gym
    from gym.wrappers import FrameStack, AtariPreprocessing
    from tqdm import tqdm
    import numpy as np

    model = get_pretained_model('dqn')

    env = gym.make('PongNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)

    total_reward = 0
    steps = 0
    all_values = []

    obs = env.reset()
    for steps in tqdm(itertools.count(), desc='Running Episode'):
        obs = torch.tensor(obs).unsqueeze(0).float()/255.
        values = model(obs)[0]
        if np.random.rand() < 1:
            action = env.action_space.sample()
        else:
            action = values.argmax()
        all_values.append(values[action].item())
        obs,reward,done,_ = env.step(action)
        total_reward += reward
        if done:
            break

    print('-'*50)
    print(f'Total steps: {steps}')
    print(f'Total reward: {total_reward}')
    print(f'Mean val: {np.mean(all_values)}')
    print('-'*50)

def test_a2c_pong():
    import itertools
    import gym
    from gym.wrappers import FrameStack, AtariPreprocessing
    from tqdm import tqdm
    import numpy as np

    model = get_pretained_model('a2c')

    env = gym.make('PongNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)

    total_reward = 0
    steps = 0
    all_values = []

    obs = env.reset()
    for steps in tqdm(itertools.count(), desc='Running Episode'):
        obs = torch.tensor(obs).unsqueeze(0).float()/255.
        model_output = model(obs)

        action_probs = model_output['action'].squeeze()
        dist = torch.distributions.Categorical(probs=action_probs.softmax(0))
        action = dist.sample().item()

        all_values.append(model_output['value'].item())
        obs,reward,done,_ = env.step(action)
        total_reward += reward
        if done:
            break

    print('-'*50)
    print(f'Total steps: {steps}')
    print(f'Total reward: {total_reward}')
    print(f'Mean val: {np.mean(all_values)}')
    print('-'*50)

if __name__ == '__main__':
    #test_dqn_pong()
    test_a2c_pong()