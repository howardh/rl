import torch

class QFunction(torch.nn.Module):
    def __init__(self, layer_sizes=[2,3,4], input_size=1, output_size=4):
        super().__init__()
        layers = []
        in_f = input_size
        for out_f in layer_sizes:
            layers.append(torch.nn.Linear(in_features=in_f,out_features=out_f))
            layers.append(torch.nn.LeakyReLU())
            in_f = out_f
        layers.append(torch.nn.Linear(in_features=in_f,out_features=output_size))
        self.seq = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.seq(x)

class PolicyFunction(torch.nn.Module):
    def __init__(self, layer_sizes=[2,3,4], input_size=1, output_size=4):
        super().__init__()
        layers = []
        in_f = input_size
        for out_f in layer_sizes:
            layers.append(torch.nn.Linear(in_features=in_f,out_features=out_f))
            layers.append(torch.nn.LeakyReLU())
            in_f = out_f
        layers.append(torch.nn.Linear(in_features=in_f,out_features=output_size))
        self.seq = torch.nn.Sequential(*layers)
        self.softmax = torch.nn.Softmax(dim=1)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, x, temperature=1, log=False):
        x = self.seq(x)
        x = x/temperature
        if log:
            return self.log_softmax(x)
        else:
            return self.softmax(x)

class PolicyFunctionAugmentatedState(torch.nn.Module):
    def __init__(self, layer_sizes=[2,3,4], state_size=1, num_actions=4, output_size=4):
        super().__init__()
        self.num_actions = num_actions
        layers = []
        in_f = state_size+num_actions
        for out_f in layer_sizes:
            layers.append(torch.nn.Linear(in_features=in_f,out_features=out_f))
            layers.append(torch.nn.LeakyReLU())
            in_f = out_f
        layers.append(torch.nn.Linear(in_features=in_f,out_features=output_size))
        layers.append(torch.nn.Softmax(dim=1))
        self.seq = torch.nn.Sequential(*layers)
    def forward(self, state, action):
        batch_size = state.shape[0]
        a = torch.zeros([batch_size,self.num_actions])
        a[range(batch_size),action] = 1
        x = torch.cat([state,a],dim=1)
        return self.seq(x)
