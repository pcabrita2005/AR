from __future__ import annotations

import random
from math import sqrt
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from connect4_rl.envs.connect_four import ConnectFourState, encode_state


@dataclass
class PPOConfig:
    episodes: int = 500
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.02
    update_epochs: int = 6
    minibatch_size: int = 32
    hidden_dim: int = 256
    rollout_episodes_per_update: int = 8
    opponent_refresh_interval: int = 20
    opponent_pool_size: int = 5
    warmup_episodes: int = 40
    random_opponent_fraction: float = 0.20
    heuristic_opponent_fraction: float = 0.15
    eval_interval: int = 50
    eval_games: int = 24
    seed: int = 0
    device: str = "cpu"
    checkpoint_score_heuristic_weight: float = 2.0
    use_horizontal_symmetry_augmentation: bool = True
    max_grad_norm: float = 1.0
    anneal_learning_rate: bool = True
    use_reward_shaping: bool = True
    reward_shaping_scale: float = 0.05


class ConnectFourActorCritic(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.shared = nn.Sequential(
            nn.Linear(64 * 6 * 7, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 7)
        self.value_head = nn.Linear(hidden_dim, 1)
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
    ) -> "PPOAgent":
        network = ConnectFourActorCritic(hidden_dim=hidden_dim)
        state_dict = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(state_dict)
        return cls(network, device=device, sample_actions=sample_actions, seed=seed)
