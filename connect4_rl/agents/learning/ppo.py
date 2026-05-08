from __future__ import annotations

import random
from math import sqrt
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from connect4_rl.envs.connect_four import ConnectFourState, encode_state


class ConnectFourActorCritic(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        channel_sizes: Sequence[int] | None = None,
        kernel_sizes: Sequence[int] | None = None,
        stride_sizes: Sequence[int] | None = None,
        head_hidden_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        channel_sizes = list(channel_sizes or [64, 128])
        kernel_sizes = list(kernel_sizes or [4, 3])
        stride_sizes = list(stride_sizes or [1, 1])
        head_hidden_sizes = list(head_hidden_sizes or [hidden_dim, max(hidden_dim // 2, 64)])
        if not (len(channel_sizes) == len(kernel_sizes) == len(stride_sizes)):
            raise ValueError("channel_sizes, kernel_sizes and stride_sizes must have the same length")

        feature_layers: list[nn.Module] = []
        in_channels = 2
        current_height = 6
        current_width = 7
        for out_channels, kernel_size, stride in zip(channel_sizes, kernel_sizes, stride_sizes):
            feature_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
            feature_layers.append(nn.ReLU())
            current_height = ((current_height - kernel_size) // stride) + 1
            current_width = ((current_width - kernel_size) // stride) + 1
            if current_height < 1 or current_width < 1:
                raise ValueError("Invalid convolution configuration for Connect Four board size")
            in_channels = out_channels
        feature_layers.append(nn.Flatten())
        self.features = nn.Sequential(*feature_layers)

        feature_dim = in_channels * current_height * current_width
        self.shared = build_mlp(feature_dim, head_hidden_sizes)
        final_hidden_dim = head_hidden_sizes[-1]
        self.policy_head = nn.Linear(final_hidden_dim, 7)
        self.value_head = nn.Linear(final_hidden_dim, 1)
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(self.features(x))
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=sqrt(2.0))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=sqrt(2.0))
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class PPOAgent:
    def __init__(
        self,
        network: ConnectFourActorCritic,
        *,
        device: str = "cpu",
        sample_actions: bool = False,
        seed: int = 0,
        name: str = "ppo",
    ) -> None:
        self.name = name
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.sample_actions = sample_actions
        self._rng = random.Random(seed)

    def select_action(self, state: ConnectFourState, legal_actions: Sequence[int]) -> int:
        if len(legal_actions) == 1:
            return legal_actions[0]

        state_tensor = torch.tensor(
            encode_state(state, state.current_player),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _value = self.network(state_tensor)
            logits = logits.squeeze(0)

        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[list(legal_actions)] = logits[list(legal_actions)]
        dist = torch.distributions.Categorical(logits=masked_logits)
        if self.sample_actions:
            return int(dist.sample().item())
        return int(torch.argmax(masked_logits).item())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        sample_actions: bool = False,
        seed: int = 0,
        hidden_dim: int = 128,
        channel_sizes: Sequence[int] | None = None,
        kernel_sizes: Sequence[int] | None = None,
        stride_sizes: Sequence[int] | None = None,
        head_hidden_sizes: Sequence[int] | None = None,
    ) -> "PPOAgent":
        network = ConnectFourActorCritic(
            hidden_dim=hidden_dim,
            channel_sizes=channel_sizes,
            kernel_sizes=kernel_sizes,
            stride_sizes=stride_sizes,
            head_hidden_sizes=head_hidden_sizes,
        )
        state_dict = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(state_dict)
        return cls(network, device=device, sample_actions=sample_actions, seed=seed)


def build_mlp(input_dim: int, hidden_layers: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for layer_dim in hidden_layers:
        layers.append(nn.Linear(in_dim, layer_dim))
        layers.append(nn.ReLU())
        in_dim = layer_dim
    return nn.Sequential(*layers)
