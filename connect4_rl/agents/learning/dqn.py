from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from connect4_rl.config import DQNConfig
from connect4_rl.envs.connect_four import ConnectFourState, encode_state

BOARD_HEIGHT = 6
BOARD_WIDTH = 7

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque(maxlen=capacity)
        self.counter = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        self._buffer.append((state, action, reward, next_state, done, next_action_mask))
        self.counter += 1

    def sample(self, batch_size: int, rng: random.Random) -> tuple[np.ndarray, ...]:
        batch = rng.sample(list(self._buffer), batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            np.stack(next_masks),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class ConnectFourQNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        *,
        channel_sizes: Sequence[int] | None = None,
        kernel_sizes: Sequence[int] | None = None,
        stride_sizes: Sequence[int] | None = None,
        head_hidden_sizes: Sequence[int] | None = None,
        use_dueling_head: bool = True,
    ) -> None:
        super().__init__()
        channel_sizes = list(channel_sizes or [128])
        kernel_sizes = list(kernel_sizes or [4])
        stride_sizes = list(stride_sizes or [1])
        head_hidden_sizes = list(head_hidden_sizes or [64, 64])
        if not (len(channel_sizes) == len(kernel_sizes) == len(stride_sizes)):
            raise ValueError("channel_sizes, kernel_sizes and stride_sizes must have the same length")

        feature_layers: list[nn.Module] = []
        in_channels = 2
        current_height = BOARD_HEIGHT
        current_width = BOARD_WIDTH
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
        hidden_layers = list(head_hidden_sizes) or [hidden_dim]
        self.use_dueling_head = use_dueling_head
        if self.use_dueling_head:
            self.value_stream = build_mlp(feature_dim, hidden_layers, 1)
            self.advantage_stream = build_mlp(feature_dim, hidden_layers, 7)
            self.head = None
        else:
            self.value_stream = None
            self.advantage_stream = None
            self.head = build_mlp(feature_dim, hidden_layers, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        if self.use_dueling_head:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            centered_advantage = advantage - advantage.mean(dim=1, keepdim=True)
            return value + centered_advantage
        return self.head(features)


class DQNAgent:
    def __init__(
        self,
        network: ConnectFourQNetwork,
        *,
        device: str = "cpu",
        epsilon: float = 0.0,
        seed: int = 0,
        name: str = "dqn",
    ) -> None:
        self.name = name
        self.network = network.to(device)
        self.network.eval()
        self.device = device
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def select_action(self, state: ConnectFourState, legal_actions: Sequence[int]) -> int:
        state_array = np.asarray(encode_state(state, state.current_player), dtype=np.float32)
        action_mask = legal_actions_to_mask(legal_actions)
        return self.get_action(state_array, epsilon=self.epsilon, action_mask=action_mask)[0]

    def get_action(
        self,
        state_array: np.ndarray,
        epsilon: float | None = None,
        action_mask: Sequence[int] | np.ndarray | None = None,
    ) -> list[int]:
        epsilon = self.epsilon if epsilon is None else epsilon
        if state_array.ndim == 3:
            state_batch = np.expand_dims(state_array, axis=0)
        else:
            state_batch = state_array

        if action_mask is None:
            mask = np.ones(7, dtype=np.float32)
        else:
            mask = np.asarray(action_mask, dtype=np.float32)

        legal = [idx for idx, value in enumerate(mask.tolist()) if value > 0]
        if len(legal) == 1:
            return [legal[0]]

        if self._rng.random() < epsilon:
            return [self._rng.choice(legal)]

        state_tensor = torch.tensor(
            state_batch,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            q_values = self.network(state_tensor).squeeze(0)

        masked_q_values = torch.full_like(q_values, -1e9)
        masked_q_values[legal] = q_values[legal]
        return [int(torch.argmax(masked_q_values).item())]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        epsilon: float = 0.0,
        seed: int = 0,
        hidden_dim: int = 128,
        channel_sizes: Sequence[int] | None = None,
        kernel_sizes: Sequence[int] | None = None,
        stride_sizes: Sequence[int] | None = None,
        head_hidden_sizes: Sequence[int] | None = None,
        use_dueling_head: bool = True,
    ) -> "DQNAgent":
        network = ConnectFourQNetwork(
            hidden_dim=hidden_dim,
            channel_sizes=channel_sizes,
            kernel_sizes=kernel_sizes,
            stride_sizes=stride_sizes,
            head_hidden_sizes=head_hidden_sizes,
            use_dueling_head=use_dueling_head,
        )
        state_dict = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(state_dict)
        return cls(network, device=device, epsilon=epsilon, seed=seed)


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def build_network_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    device: str = "cpu",
    hidden_dim: int = 128,
    channel_sizes: Sequence[int] | None = None,
    kernel_sizes: Sequence[int] | None = None,
    stride_sizes: Sequence[int] | None = None,
    head_hidden_sizes: Sequence[int] | None = None,
    use_dueling_head: bool = True,
) -> ConnectFourQNetwork:
    network = ConnectFourQNetwork(
        hidden_dim=hidden_dim,
        channel_sizes=channel_sizes,
        kernel_sizes=kernel_sizes,
        stride_sizes=stride_sizes,
        head_hidden_sizes=head_hidden_sizes,
        use_dueling_head=use_dueling_head,
    )
    network.load_state_dict(state_dict)
    return network.to(device)


def build_mlp(input_dim: int, hidden_layers: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for layer_dim in hidden_layers:
        layers.append(nn.Linear(in_dim, layer_dim))
        layers.append(nn.ReLU())
        in_dim = layer_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def epsilon_by_step(config: DQNConfig, step: int) -> float:
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end
    progress = step / max(config.epsilon_decay_steps, 1)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def state_to_numpy(state: ConnectFourState, perspective_player: int) -> np.ndarray:
    return np.asarray(encode_state(state, perspective_player), dtype=np.float32)


def legal_actions_to_mask(legal_actions: Iterable[int]) -> np.ndarray:
    mask = np.zeros(7, dtype=np.float32)
    for action in legal_actions:
        mask[action] = 1.0
    return mask


def flip_state_horizontally(state: np.ndarray) -> np.ndarray:
    return np.flip(state, axis=2).copy()


def flip_action_horizontally(action: int) -> int:
    return 6 - action


def flip_action_mask_horizontally(mask: np.ndarray) -> np.ndarray:
    return np.flip(mask, axis=0).copy()
