from __future__ import annotations

import math
import random
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from connect4_rl.envs.connect_four import ConnectFourState, encode_state


@dataclass
class AlphaZeroConfig:
    episodes: int = 300
    learning_rate: float = 2.5e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 20_000
    replay_warmup_games: int = 16
    update_epochs: int = 4
    updates_per_episode: int = 2
    hidden_dim: int = 256
    mcts_simulations: int = 120
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_drop_move: int = 8
    eval_interval: int = 25
    eval_games: int = 24
    seed: int = 0
    device: str = "cpu"
    checkpoint_score_heuristic_weight: float = 2.0
    use_horizontal_symmetry_augmentation: bool = True
    value_loss_coef: float = 1.0
    max_grad_norm: float = 5.0
    anneal_learning_rate: bool = True
    root_noise_each_move: bool = True


class ConnectFourPolicyValueNet(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
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
        value = torch.tanh(self.value_head(hidden).squeeze(-1))
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


class AlphaZeroAgent:
    def __init__(
        self,
        network: ConnectFourPolicyValueNet,
        *,
        simulations: int = 80,
        c_puct: float = 1.5,
        device: str = "cpu",
        seed: int = 0,
        temperature: float = 0.0,
        name: str = "alphazero",
    ) -> None:
        self.name = name
        self.network = network.to(device)
        self.network.eval()
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device
        self.temperature = temperature
        self._rng = random.Random(seed)

    def select_action(self, state: ConnectFourState, legal_actions: Sequence[int]) -> int:
        if len(legal_actions) == 1:
            return legal_actions[0]

        visit_policy = run_policy_value_mcts(
            self.network,
            state,
            simulations=self.simulations,
            c_puct=self.c_puct,
            device=self.device,
            root_dirichlet_alpha=None,
            root_dirichlet_epsilon=0.0,
            rng=self._rng,
        )
        legal_policy = np.asarray([visit_policy[action] for action in legal_actions], dtype=np.float64)
        if self.temperature <= 0.0:
            return int(legal_actions[int(np.argmax(legal_policy))])

        scaled = np.power(np.maximum(legal_policy, 1e-8), 1.0 / self.temperature)
        scaled /= scaled.sum()
        return int(self._rng.choices(list(legal_actions), weights=scaled.tolist(), k=1)[0])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        simulations: int = 80,
        c_puct: float = 1.5,
        seed: int = 0,
        hidden_dim: int = 128,
        temperature: float = 0.0,
    ) -> "AlphaZeroAgent":
        network = ConnectFourPolicyValueNet(hidden_dim=hidden_dim)
        state_dict = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(state_dict)
        return cls(
            network,
            simulations=simulations,
            c_puct=c_puct,
            device=device,
            seed=seed,
            temperature=temperature,
        )


class PolicyValueMCTSNode:
    def __init__(self, state: ConnectFourState, prior: float = 0.0, parent: PolicyValueMCTSNode | None = None) -> None:
        self.state = state
        self.prior = prior
        self.parent = parent
        self.children: dict[int, PolicyValueMCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def run_policy_value_mcts(
    network: ConnectFourPolicyValueNet,
    root_state: ConnectFourState,
    *,
    simulations: int,
    c_puct: float,
    device: str,
    root_dirichlet_alpha: float | None,
    root_dirichlet_epsilon: float,
    rng: random.Random,
) -> np.ndarray:
    root = PolicyValueMCTSNode(root_state)
    expand_policy_value_node(
        root,
        network,
        device=device,
        rng=rng,
        add_dirichlet_noise=root_dirichlet_alpha is not None and root_dirichlet_epsilon > 0.0,
        dirichlet_alpha=root_dirichlet_alpha or 0.3,
        dirichlet_epsilon=root_dirichlet_epsilon,
    )

    for _ in range(simulations):
        node = root
        search_path = [node]

        while node.children:
            action, node = select_child(node, c_puct)
            _ = action
            search_path.append(node)

        value = evaluate_leaf(node, network, device=device, rng=rng)
        backpropagate(search_path, value)

    visit_policy = np.zeros(7, dtype=np.float32)
    total_visits = sum(child.visit_count for child in root.children.values())
    if total_visits == 0:
        return visit_policy

    for action, child in root.children.items():
        visit_policy[action] = child.visit_count / total_visits
    return visit_policy


def select_child(node: PolicyValueMCTSNode, c_puct: float) -> tuple[int, PolicyValueMCTSNode]:
    sqrt_parent = math.sqrt(max(node.visit_count, 1))
    best_action = -1
    best_child: PolicyValueMCTSNode | None = None
    best_score = -float("inf")

    for action, child in node.children.items():
        score = -child.q_value + c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    assert best_child is not None
    return best_action, best_child


def evaluate_leaf(
    node: PolicyValueMCTSNode,
    network: ConnectFourPolicyValueNet,
    *,
    device: str,
    rng: random.Random,
) -> float:
    from connect4_rl.envs.connect_four import apply_action, is_terminal, legal_actions, outcome_for_player

    if is_terminal(node.state):
        return float(outcome_for_player(node.state, node.state.current_player))

    expand_policy_value_node(
        node,
        network,
        device=device,
        rng=rng,
        add_dirichlet_noise=False,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
    )
    state_tensor = torch.tensor(
        encode_state(node.state, node.state.current_player),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        _logits, value = network(state_tensor)
    return float(value.item())


def expand_policy_value_node(
    node: PolicyValueMCTSNode,
    network: ConnectFourPolicyValueNet,
    *,
    device: str,
    rng: random.Random,
    add_dirichlet_noise: bool,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> None:
    from connect4_rl.envs.connect_four import apply_action, legal_actions

    if node.children:
        return

    legal = legal_actions(node.state)
    state_tensor = torch.tensor(
        encode_state(node.state, node.state.current_player),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        logits, _value = network(state_tensor)
        logits = logits.squeeze(0)

    masked_logits = torch.full_like(logits, -1e9)
    masked_logits[legal] = logits[legal]
    priors = torch.softmax(masked_logits, dim=0).detach().cpu().numpy()

    if add_dirichlet_noise:
        noise = np.random.default_rng(rng.randint(0, 1_000_000)).dirichlet([dirichlet_alpha] * len(legal))
        noisy_priors = priors.copy()
        for idx, action in enumerate(legal):
            noisy_priors[action] = (1.0 - dirichlet_epsilon) * priors[action] + dirichlet_epsilon * noise[idx]
        priors = noisy_priors

    for action in legal:
        next_state = apply_action(node.state, action)
        node.children[action] = PolicyValueMCTSNode(next_state, prior=float(priors[action]), parent=node)


def backpropagate(search_path: list[PolicyValueMCTSNode], value: float) -> None:
    current_value = value
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += current_value
        current_value = -current_value
