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
