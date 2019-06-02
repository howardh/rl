import numpy as np
import itertools
import scipy.sparse
from tqdm import tqdm

import torch
from torch.autograd import Variable

import atari
from atari import utils

IDENTITY_NUM_FEATURES = 160*210

def identity(x):
    return np.append(x,1).reshape([IDENTITY_NUM_FEATURES,1])

def generate_tiles(w,h,tw,th):
    return [int(i/(w*th))*int(w/tw)+int((i%w)/tw) for i in range(h*w)]

TILES = generate_tiles(160,210,10,15)
BACKGROUND = [0]*IDENTITY_NUM_FEATURES
BASIC_NUM_FEATURES = 16*14*128
def basic(x, background=BACKGROUND):
    NUM_COLOURS = 128
    results = np.zeros(16*14*NUM_COLOURS)
    x = x.reshape((160*210,3))
    for i,c in enumerate(x):
        ci = atari.utils.rgb_to_index(c)
        if background[i] == ci:
            continue
        index = TILES[i]*NUM_COLOURS+ci
        results[index] = 1
    return results

def get_basic(env_name):
    bg = atari.utils.compute_background(env_name)
    def foo(x):
        return basic(x, background=bg)
    return foo

LATENT_SIZE = 20
AUTOENCODER_NUM_FEATURES = LATENT_SIZE
class ConvEncoder(torch.nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.relu = torch.nn.LeakyReLU()
        self.c1 = torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(8,8),stride=2,padding=(0,1))
        self.c2 = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(6,6),stride=2,padding=(1,1))
        self.c3 = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(6,6),stride=2,padding=(1,1))
        self.c4 = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(4,4),stride=2)

        self.fc1_mean = torch.nn.Linear(in_features=128*11*8,out_features=LATENT_SIZE,bias=True)
        self.fc1_std = torch.nn.Linear(in_features=128*11*8,out_features=LATENT_SIZE,bias=True)

    def forward(self, inputs, sample):
        output = inputs.view(-1,3,210,160)
        output = self.c1(output)
        output = self.relu(output)
        output = self.c2(output)
        output = self.relu(output)
        output = self.c3(output)
        output = self.relu(output)
        output = self.c4(output)
        output = self.relu(output)
        output = output.view(-1,128*11*8)

        mean = self.fc1_mean(output)
        logvar = self.fc1_std(output) # log(std^2)
        std = logvar.mul(0.5).exp()

        sample = sample.mul(std).add_(mean)
    
        return sample, mean, logvar

net = ConvEncoder()
net.cuda()
with open('/home/ml/hhuang63/vae/encoder-weights.pt', 'rb') as f:
    #net.load_state_dict(torch.load(f, map_location=lambda storage, location: storage))
    net.load_state_dict(torch.load(f))
def autoencoder(x):
    x = torch.Tensor(x)
    sample = torch.zeros([AUTOENCODER_NUM_FEATURES]).float()
    x = Variable(x.cuda())
    sample = Variable(sample.cuda())
    s,m,l = net(x, sample)
    return s.data.cpu().numpy()
