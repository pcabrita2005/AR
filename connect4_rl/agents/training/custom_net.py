import json
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

class CustomNetwork(nn.Module):
    """ Configurable Neural Network for Connect4. """

    def __init__(self,
                 conv_block: List = (),
                 fc_block: List = (),
                 first_head: List = (),
                 second_head: List = (),
                 name: str = 'CustomNetwork',
                 board_shape: Tuple[int, int] = (6, 7)):
        super().__init__()
        self.arch = {'conv_block': conv_block, 'fc_block': fc_block,
                     'first_head': first_head, 'second_head': second_head}
        self.board_shape = board_shape
        self.input_shape = (2, *self.board_shape)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv_block = nn.Sequential()
        in_ch = 2
        for layer in conv_block:
            in_ch = self._add_conv_layer(layer, self.conv_block, in_ch)

        with torch.no_grad():
            conv_out = self.conv_block(torch.zeros((1, *self.input_shape)))
            in_ft = torch.prod(torch.tensor(conv_out.shape)).item()

        self.fc_block = nn.Sequential()
        for layer in fc_block:
            in_ft = self._add_fc_layer(layer, self.fc_block, in_ft)

        self.first_head = nn.Sequential()
        in_ft1 = in_ft
        for layer in first_head:
            in_ft1 = self._add_fc_layer(layer, self.first_head, in_ft1)

        self.second_head = nn.Sequential()
        in_ft2 = in_ft
        for layer in second_head:
            in_ft2 = self._add_fc_layer(layer, self.second_head, in_ft2)
        
        self.name = name
        self.to(self.device)

    def _add_conv_layer(self, layer, module, in_ch):
        if isinstance(layer, list):
            module.append(nn.Conv2d(in_ch, layer[0], kernel_size=layer[1], padding=layer[2]))
            return layer[0]
        elif isinstance(layer, str):
            if layer.lower() == 'relu': module.append(nn.ReLU())
            elif layer.lower() == 'tanh': module.append(nn.Tanh())
        return in_ch

    def _add_fc_layer(self, layer, module, in_ft):
        if isinstance(layer, int):
            module.append(nn.Linear(in_ft, layer))
            return layer
        elif isinstance(layer, str):
            if layer.lower() == 'relu': module.append(nn.ReLU())
            elif layer.lower() == 'tanh': module.append(nn.Tanh())
        return in_ft

    def forward(self, x):
        if len(self.conv_block) > 0:
            x = self.conv_block(x)
            x = torch.flatten(x, 1)
        x = self.fc_block(x)
        out1 = self.first_head(x)
        if len(self.second_head) > 0:
            return out1, self.second_head(x)
        return out1

    @classmethod
    def from_architecture(cls, file_path: str, n_heads: int = 2):
        with open(file_path, 'r') as f:
            arch = json.load(f)
        net = cls(**arch)
        if n_heads == 1: net.second_head = nn.Sequential()
        return net

    def obs_to_model_input(self, obs: np.array) -> torch.Tensor:
        """ Preprocesses board for the network (one-hot + available cells). """
        inp = np.zeros((2, *obs.shape))
        inp[0][obs == 1] = 1
        inp[1][obs == -1] = 1
        # Highlight available cells
        filled = inp[0] + inp[1]
        for ch in inp:
            for col in range(ch.shape[1]):
                empty = np.where(filled[:, col] == 0)[0]
                if len(empty) > 0: ch[empty[-1], col] = -1
        
        device = next(self.parameters()).device
        return torch.from_numpy(inp).float().unsqueeze(0).to(device)

    def save_weights(self, file_path, training_hparams=None):
        torch.save(self.state_dict(), file_path)
        if training_hparams:
            with open(file_path.replace('.pt', '_hparams.json'), 'w') as f:
                json.dump(training_hparams, f, indent=4)

    def load_weights(self, file_path):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(file_path, map_location=device))

if __name__ == "__main__":
    arch = {'conv_block': [[32, 4, 0], 'relu'], 'fc_block': [64, 'relu'],
            'first_head': [32, 'relu', 7], 'second_head': [16, 'relu', 1]}
    model = CustomNetwork(**arch)
    summary(model, model.input_shape)
