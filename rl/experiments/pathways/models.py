import torch
from torchtyping.tensor_type import TensorType
from torch.utils.data.dataloader import default_collate

from rl.agent.smdp.a2c import PolicyValueNetworkRecurrent


class RecurrentAttention(torch.nn.Module):
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention, self).__init__()
        self.fc_query = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size)
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        query = self.fc_query(x).unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        return {
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0), # (batch_size, value_size)
            'x': attn_output.squeeze(0), # (batch_size, value_size)
        }


class RecurrentAttention2(torch.nn.Module):
    # Output to next block is computed from the attention output rather than just being the raw attention output
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention2, self).__init__()
        self.fc_query = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size)
        )
        self.fc_output = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, input_size)
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        query = self.fc_query(x).unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, value_size)
        return {
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0), # (batch_size, value_size)
            'x': output_x # (batch_size, value_size)
        }


class RecurrentAttention3(torch.nn.Module):
    # Output to next block includes information from the block's input.
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention3, self).__init__()
        self.fc_query = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size)
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size)
        )
        self.fc_output = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, input_size)
        )
        self.fc_gate = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, 1),
                torch.nn.Sigmoid()
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        query = self.fc_query(x).unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, 1)
        return {
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0), # (batch_size, value_size)
            'x': output_gate*output_x + (1-output_gate)*attn_output.squeeze(0) # (batch_size, value_size)
        }


class RecurrentAttention4(torch.nn.Module):
    # Bounded outputs with tanh
    def __init__(self, input_size, key_size, value_size, num_heads, ff_size):
        super(RecurrentAttention4, self).__init__()
        self.fc_query = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size),
                torch.nn.Tanh(),
        )
        self.fc_key = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, key_size),
                torch.nn.Tanh(),
        )
        self.fc_value = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, value_size),
                torch.nn.Tanh(),
        )
        self.fc_output = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, input_size),
                torch.nn.Tanh(),
        )
        self.fc_gate = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(input_size*2, ff_size),
                torch.nn.ReLU(),
                torch.nn.Linear(ff_size, 1),
                torch.nn.Sigmoid()
        )
        self.attention = torch.nn.MultiheadAttention(key_size, num_heads=num_heads, batch_first=False)
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        query = self.fc_query(x).unsqueeze(0) # (1, batch_size, key_size)
        attn_output, attn_output_weights = self.attention(query, input_keys, input_values) # (1, batch_size, value_size)
        output_keys = self.fc_key(attn_output) # (1, batch_size, key_size)
        output_values = self.fc_value(attn_output) # (1, batch_size, value_size)
        output_x = self.fc_output(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, value_size)
        output_gate = self.fc_gate(torch.cat([attn_output.squeeze(0),x], dim=1)) # (batch_size, 1)
        return {
            'attn_output': attn_output.squeeze(0), # (batch_size, value_size)
            'attn_output_weights': attn_output_weights.squeeze(1), # (batch_size, seq_len)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0), # (batch_size, value_size)
            'x': output_gate*output_x + (1-output_gate)*attn_output.squeeze(0) # (batch_size, value_size)
        }


class RecurrentAttention5(RecurrentAttention):
    # Just add tanh to RecurrentAttention, which we already know to work
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        output = super().forward(x, input_keys, input_values)
        output['x'] = output['x'].tanh()
        output['key'] = output['x'].tanh()
        output['value'] = output['x'].tanh()
        return output


class RecurrentAttention6(RecurrentAttention):
    # RecurrentAttention5 doesn't seem to work. Try only using tanh on the value output, since that's the part that propagates through time and has a higher potential of exploding.
    # This is working
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        output = super().forward(x, input_keys, input_values)
        output['value'] = output['x'].tanh()
        return output


class RecurrentAttention7(RecurrentAttention2):
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        output = super().forward(x, input_keys, input_values)
        output['value'] = output['x'].tanh()
        return output


class ConvPolicy(PolicyValueNetworkRecurrent):
    def __init__(self, num_actions, in_channels, input_size, key_size, value_size, num_heads, ff_size, num_blocks=1,
            recurrence_type='RecurrentAttention'):
        super(ConvPolicy, self).__init__()
        self.key_size = key_size
        self.input_size = input_size

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,out_channels=32,kernel_size=8,stride=4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=32,out_channels=64,kernel_size=4,stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=64,out_channels=64,kernel_size=3,stride=1),
            torch.nn.Flatten(),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64*7*7,out_features=512),
            torch.nn.LeakyReLU(),
        )
        self.fc_key = torch.nn.Linear(in_features=512, out_features=key_size)
        self.fc_value = torch.nn.Linear(in_features=512, out_features=value_size)

        recurrenceClasses = {
                cls.__name__: cls
                for cls in [
                    RecurrentAttention,
                    RecurrentAttention2,
                    RecurrentAttention3,
                    RecurrentAttention4,
                    RecurrentAttention5,
                    RecurrentAttention6,
                    RecurrentAttention7,
                ]
        }
        recurrenceCls = None
        if recurrence_type in recurrenceClasses:
            recurrenceCls = recurrenceClasses[recurrence_type]
        else:
            raise ValueError('Unknown recurrence type: {}'.format(recurrence_type))
        self.attention = torch.nn.ModuleList([
                recurrenceCls(input_size, key_size, value_size, num_heads, ff_size)
                for _ in range(num_blocks)
        ])

        self.fc_output = torch.nn.Sequential(
                torch.nn.LeakyReLU(),
                torch.nn.Linear(input_size, 512),
                torch.nn.LeakyReLU(),
        )
        self.v = torch.nn.Linear(in_features=512,out_features=1)
        self.pi = torch.nn.Linear(in_features=512,out_features=num_actions)

    def forward(self, x: TensorType['batch_size','observation_shape'], hidden):
        batch_size = x.shape[0]
        device = next(self.parameters()).device

        x = self.conv(x)
        keys = torch.cat([
            self.fc_key(x).unsqueeze(0),
            hidden[0]
        ], dim=0)
        values = torch.cat([
            self.fc_value(x).unsqueeze(0),
            hidden[1]
        ], dim=0)

        new_hidden = []
        x = torch.zeros([batch_size, self.input_size], device=device)
        for attention in self.attention:
            output = attention(x, keys, values)
            x = output['x']
            new_hidden.append((output['key'], output['value']))
        x = self.fc_output(x)

        return {
            'value': self.v(x),
            'action': self.pi(x),
            'hidden': default_collate(new_hidden)
        }

    def init_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        return (
                torch.zeros([len(self.attention), batch_size, self.key_size], device=device), # Key
                torch.zeros([len(self.attention), batch_size, self.key_size], device=device), # Query
        )


if __name__ == '__main__':
    rann = RecurrentAttention(input_size=10, key_size=10, value_size=10, num_heads=2, ff_size=10)
    out = rann(
        torch.randn(2, 10),
        torch.randn(10, 2, 10),
        torch.randn(10, 2, 10)
    )

    net = ConvPolicy(num_actions=6, in_channels=4, input_size=10, key_size=10, value_size=10, num_heads=2, ff_size=10)
    out = net(
        torch.randn(2, 4, 84, 84),
        net.init_hidden(2)
    )
    breakpoint()
