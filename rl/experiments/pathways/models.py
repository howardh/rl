from typing import List, Dict

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
        attn_output, attn_output_weights = self.attention(
                query, input_keys, input_values, average_attn_weights=True) # (1, batch_size, value_size)
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
    # Output to next block is gated
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
            'output_gate': output_gate.squeeze(1), # (batch_size)
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


class RecurrentAttention8(RecurrentAttention3):
    def forward(self,
            x: TensorType['batch_size','input_size',float],
            input_keys: TensorType['seq_len','batch_size','key_size',float],
            input_values: TensorType['seq_len','batch_size','value_size',float]):
        output = super().forward(x, input_keys, input_values)
        output['value'] = output['x'].tanh()
        return output


class RecurrentAttention9(RecurrentAttention3):
    # Same as RecurrentAttention8, but the feed-forward output gating chooses between something computed from the MHA output and the feed-forward input rather than between two things computed from the attention
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
            'output_gate': output_gate.squeeze(1), # (batch_size)
            'key': output_keys.squeeze(0), # (batch_size, key_size)
            'value': output_values.squeeze(0).tanh(), # (batch_size, value_size)
            'x': output_gate*output_x + (1-output_gate)*x # (batch_size, value_size)
        }


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
                    RecurrentAttention8,
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

    def forward(self,
            x: TensorType['batch_size','observation_shape'],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2
        batch_size = x.shape[0]
        device = next(self.parameters()).device

        x = self.conv(x) # (batch_size, 512)
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


class GreyscaleImageInput(torch.nn.Module):
    def __init__(self, key_size: int, value_size: int, in_channels: int):
        super().__init__()
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
    def forward(self, x: TensorType['batch_size','frame_stack','height','width',float]):
        x = self.conv(x)
        return {
            'key': self.fc_key(x),
            'value': self.fc_value(x),
        }


class ScalarInput(torch.nn.Module):
    def __init__(self, key_size: int, value_size: int):
        super().__init__()
        self.value_size = value_size
        self.key = torch.nn.Parameter(torch.rand([key_size]))
    def forward(self, value: TensorType['batch_size',float]):
        batch_size = value.shape[0]
        return {
            'key': self.key.expand(batch_size, -1),
            'value': value.expand(batch_size,self.value_size)
        }


class LinearInput(torch.nn.Module):
    def __init__(self, input_size: int, key_size: int, value_size: int, shared_key: bool = False):
        """
        Args:
            input_size: The size of the input vector.
            key_size: The size of the key.
            value_size: The size of the value.
            shared_key: If set to True, the same key will be used regardless of input. If set to False, the key will be computed as a linear function of the input.
        """
        super().__init__()
        self._shared_key = shared_key
        self.ff_value = torch.nn.Linear(in_features=input_size, out_features=value_size)
        if shared_key:
            self.key = torch.nn.Parameter(torch.rand([key_size])-0.5)
        else:
            self.ff_key = torch.nn.Linear(in_features=input_size, out_features=key_size)
    def forward(self, x: TensorType['batch_size',float]):
        batch_size = x.shape[0]
        return {
            'key': self.ff_key(x) if not self._shared_key else self.key.expand(batch_size, -1),
            'value': self.ff_value(x)
        }


class DiscreteInput(torch.nn.Module):
    def __init__(self, input_size: int, key_size: int, value_size: int, shared_key: bool = False):
        """
        Args:
            input_size: Number of possible input values.
            key_size: The size of the key.
            value_size: The size of the value.
            shared_key: If set to True, the same key will be used regardless of input. If set to False, the key will be computed as a linear function of the input.
        """
        super().__init__()
        self._shared_key = shared_key
        self._input_size = input_size
        self.value = torch.nn.Parameter(torch.rand([input_size, value_size])-0.5)
        if shared_key:
            self.key = torch.nn.Parameter(torch.rand([key_size])-0.5)
        else:
            self.key = torch.nn.Parameter(torch.rand([input_size, key_size])-0.5)
    def forward(self, x: TensorType['batch_size',int]):
        x = x.long()
        return {
            'key': self.key[x,:],
            'value': self.value[x,:]
        }


class ModularPolicy(PolicyValueNetworkRecurrent):
    def __init__(self, inputs, num_actions, input_size, key_size, value_size, num_heads, ff_size, num_blocks=1,
            recurrence_type='RecurrentAttention'):
        super().__init__()
        self.key_size = key_size
        self.input_size = input_size
        self.value_size = value_size

        self.input_modules = self._init_input_modules(inputs)

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
                    RecurrentAttention8,
                    RecurrentAttention9,
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

        self.last_attention = None # Store the attention for analysis purposes
        self.last_ff_gating = None

    def _init_input_modules(self, input_configs: Dict[str,Dict]):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ScalarInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            input_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = self.key_size,
                    value_size = self.value_size)
        return torch.nn.ModuleDict(input_modules)

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2
        batch_size = hidden[0].shape[1]
        device = next(self.parameters()).device

        self.last_attention = []
        self.last_ff_gating = []

        input_keys = []
        input_vals = []
        for k,x in inputs.items():
            if k not in self.input_modules:
                continue
            y = self.input_modules[k](x)
            input_keys.append(y['key'].unsqueeze(0))
            input_vals.append(y['value'].unsqueeze(0))

        keys = torch.cat([
            *input_keys,
            hidden[0]
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1]
        ], dim=0)

        new_hidden = []
        x = torch.zeros([batch_size, self.input_size], device=device)
        for attention in self.attention:
            output = attention(x, keys, values)
            x = output['x']
            new_hidden.append((output['key'], output['value']))
            self.last_attention.append([h.cpu().detach() for h in output['attn_output_weights']])
            self.last_ff_gating.append(output['output_gate'].cpu().detach())
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


class LinearOutput(torch.nn.Module):
    def __init__(self, output_size: int, key_size: int, num_heads: int):
        super().__init__()
        self.output_size = output_size

        self.query = torch.nn.Parameter((torch.rand([key_size])-0.5)*0.01)
        self.attention = torch.nn.MultiheadAttention(
                key_size, num_heads=num_heads, batch_first=False)
        self.ff = torch.nn.Linear(key_size, output_size)
    def forward(self,
            key: TensorType['num_blocks','batch_size','hidden_size',float],
            value: TensorType['num_blocks','batch_size','hidden_size',float],
            ) -> Dict[str,TensorType]:
        attn_output, attn_output_weights = self.attention(
                self.query.expand(1, key.shape[1], -1),
                key,
                value
        ) # (1, batch_size, value_size)
        output = self.ff(attn_output.squeeze(0))
        return {
            'output': output,
            'attn_output_weights': attn_output_weights,
        }


class ModularPolicy2(PolicyValueNetworkRecurrent):
    def __init__(self, inputs, outputs, input_size, key_size, value_size, num_heads, ff_size, num_blocks=1,
            recurrence_type='RecurrentAttention'):
        super().__init__()
        self.key_size = key_size
        self.input_size = input_size
        self.value_size = value_size

        self.input_modules = self._init_input_modules(inputs,
                key_size=key_size, value_size=value_size)
        self.output_modules = self._init_output_modules(outputs,
                key_size=key_size, num_heads=num_heads)

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
                    RecurrentAttention8,
                    RecurrentAttention9,
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

        # Store the attention for analysis purposes
        self.last_attention = None
        self.last_ff_gating = None
        self.last_output_attention = None

    def _init_input_modules(self, input_configs: Dict[str,Dict], key_size, value_size):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    GreyscaleImageInput,
                    ScalarInput,
                    DiscreteInput,
                    LinearInput,
                ]
        }
        input_modules: Dict[str,torch.nn.Module] = {}
        for k,v in input_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            input_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    value_size = value_size)
        return torch.nn.ModuleDict(input_modules)

    def _init_output_modules(self, output_configs: Dict[str,Dict], key_size, num_heads):
        valid_modules = {
                cls.__name__: cls
                for cls in [
                    LinearOutput,
                ]
        }
        output_modules: Dict[str,torch.nn.Module] = {}
        for k,v in output_configs.items():
            if v['type'] not in valid_modules:
                raise NotImplementedError(f'Unknown output module type: {v["type"]}')
            if k == 'hidden':
                raise Exception('Cannot use "hidden" as an output module name')
            output_modules[k] = valid_modules[v['type']](
                    **v.get('config', {}),
                    key_size = key_size,
                    num_heads = num_heads)
        return torch.nn.ModuleDict(output_modules)

    def forward(self,
            inputs: Dict[str,TensorType['batch_size','observation_shape']],
            hidden: List[TensorType['num_blocks','batch_size','hidden_size']]):
        assert len(hidden) == 2
        batch_size = hidden[0].shape[1]
        device = next(self.parameters()).device

        self.last_attention = []
        self.last_ff_gating = []
        self.last_output_attention = []

        # Compute input to core module
        input_keys = []
        input_vals = []
        for k,x in inputs.items():
            if k not in self.input_modules:
                continue
            y = self.input_modules[k](x)
            input_keys.append(y['key'].unsqueeze(0))
            input_vals.append(y['value'].unsqueeze(0))

        keys = torch.cat([
            *input_keys,
            hidden[0]
        ], dim=0)
        values = torch.cat([
            *input_vals,
            hidden[1]
        ], dim=0)

        # Core module computation
        new_hidden = []
        x = torch.zeros([batch_size, self.input_size], device=device)
        for attention in self.attention:
            output = attention(x, keys, values)
            x = output['x']
            new_hidden.append((output['key'], output['value']))
            self.last_attention.append([h.cpu().detach() for h in output['attn_output_weights']])
            self.last_ff_gating.append(output['output_gate'].cpu().detach())
        new_hidden = default_collate(new_hidden)

        # Compute output
        output = {}

        keys = torch.cat([
            *input_keys,
            new_hidden[0]
        ], dim=0)
        values = torch.cat([
            *input_vals,
            new_hidden[1]
        ], dim=0)

        for k,v in self.output_modules.items():
            y = v(keys, values)
            output[k] = y['output']
            self.last_output_attention.append([h.cpu().detach() for h in y['attn_output_weights']])

        return {
            **output,
            'hidden': new_hidden
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
