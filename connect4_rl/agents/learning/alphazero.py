from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from connect4_rl.config import AlphaZeroConfig
from connect4_rl.envs.connect_four import ConnectFourState, apply_action, is_terminal, legal_actions, outcome_for_player


def encode_alphazero_state(
    state: ConnectFourState,
    perspective_player: int | None = None,
) -> np.ndarray:
    """Encode state as three planes: own pieces, empty cells, opponent pieces."""
    player = perspective_player or state.current_player
    opponent = 2 if player == 1 else 1
    board = np.asarray(state.board, dtype=np.int8)
    return np.stack(
        (
            board == player,
            board == 0,
            board == opponent,
        ),
        axis=0,
    ).astype(np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, n_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ConvBase(nn.Module):
    def __init__(self, n_filters: int, n_res_blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(n_filters) for _ in range(n_res_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)
        return x


class ConnectFourPolicyValueNet(nn.Module):
    def __init__(self, n_filters: int = 128, n_res_blocks: int = 8) -> None:
        super().__init__()
        self.base = ConvBase(n_filters=n_filters, n_res_blocks=n_res_blocks)
        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters // 4),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear((n_filters // 4) * 6 * 7, 7),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, max(1, n_filters // 32), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(1, n_filters // 32)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(max(1, n_filters // 32) * 6 * 7, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.base(x)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


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
        dirichlet_alpha: float | None = None,
        dirichlet_epsilon: float = 0.0,
        name: str = "alphazero",
    ) -> None:
        self.name = name
        self.network = network.to(device)
        self.network.eval()
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self._rng = random.Random(seed)

    def select_action(self, state: ConnectFourState, legal_actions_now: Sequence[int]) -> int:
        if len(legal_actions_now) == 1:
            return int(legal_actions_now[0])

        visit_policy = run_policy_value_mcts(
            self.network,
            state,
            simulations=self.simulations,
            c_puct=self.c_puct,
            device=self.device,
            root_dirichlet_alpha=self.dirichlet_alpha,
            root_dirichlet_epsilon=self.dirichlet_epsilon,
            rng=self._rng,
        )
        return sample_action_from_policy(
            visit_policy,
            list(legal_actions_now),
            temperature=self.temperature,
            rng=self._rng,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        simulations: int = 80,
        c_puct: float = 1.5,
        seed: int = 0,
        temperature: float = 0.0,
        n_filters: int = 128,
        n_res_blocks: int = 8,
    ) -> "AlphaZeroAgent":
        network = ConnectFourPolicyValueNet(n_filters=n_filters, n_res_blocks=n_res_blocks)
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
    def __init__(
        self,
        state: ConnectFourState,
        *,
        prior: float = 0.0,
        parent: PolicyValueMCTSNode | None = None,
    ) -> None:
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


def sample_action_from_policy(
    policy: np.ndarray,
    legal: list[int],
    *,
    temperature: float,
    rng: random.Random,
) -> int:
    legal_probs = np.asarray([policy[action] for action in legal], dtype=np.float64)
    if temperature <= 0.0:
        return int(legal[int(np.argmax(legal_probs))])

    scaled = np.power(np.maximum(legal_probs, 1e-8), 1.0 / temperature)
    scaled /= scaled.sum()
    return int(rng.choices(legal, weights=scaled.tolist(), k=1)[0])


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
            _, node = select_child(node, c_puct)
            search_path.append(node)

        if is_terminal(node.state):
            value = float(outcome_for_player(node.state, node.state.current_player))
        else:
            value = expand_policy_value_node(
                node,
                network,
                device=device,
                rng=rng,
                add_dirichlet_noise=False,
                dirichlet_alpha=0.0,
                dirichlet_epsilon=0.0,
            )
        backpropagate(search_path, value)

    visit_policy = np.zeros(7, dtype=np.float32)
    total_visits = sum(child.visit_count for child in root.children.values())
    if total_visits == 0:
        legal = legal_actions(root_state)
        if legal:
            visit_policy[legal] = 1.0 / len(legal)
        return visit_policy

    for action, child in root.children.items():
        visit_policy[action] = child.visit_count / total_visits
    return visit_policy


def select_child(node: PolicyValueMCTSNode, c_puct: float) -> tuple[int, PolicyValueMCTSNode]:
    sqrt_parent_visits = math.sqrt(max(node.visit_count, 1))
    best_action = -1
    best_child: PolicyValueMCTSNode | None = None
    best_score = -float("inf")

    for action, child in node.children.items():
        exploitation = -child.q_value
        exploration = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
        score = exploitation + exploration
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    assert best_child is not None
    return best_action, best_child


def expand_policy_value_node(
    node: PolicyValueMCTSNode,
    network: ConnectFourPolicyValueNet,
    *,
    device: str,
    rng: random.Random,
    add_dirichlet_noise: bool,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> float:
    if node.children or is_terminal(node.state):
        return float(outcome_for_player(node.state, node.state.current_player))

    legal = legal_actions(node.state)
    state_tensor = torch.tensor(
        encode_alphazero_state(node.state, node.state.current_player),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        logits, value = network(state_tensor)
        logits = logits.squeeze(0)

    masked_logits = torch.full_like(logits, -1e9)
    masked_logits[legal] = logits[legal]
    priors = torch.softmax(masked_logits, dim=0).detach().cpu().numpy()

    if add_dirichlet_noise and legal:
        noise = np.random.default_rng(rng.randint(0, 1_000_000)).dirichlet([dirichlet_alpha] * len(legal))
        mixed_priors = priors.copy()
        for idx, action in enumerate(legal):
            mixed_priors[action] = (1.0 - dirichlet_epsilon) * priors[action] + dirichlet_epsilon * noise[idx]
        priors = mixed_priors

    for action in legal:
        child_state = apply_action(node.state, action)
        node.children[action] = PolicyValueMCTSNode(child_state, prior=float(priors[action]), parent=node)

    return float(value.item())


def backpropagate(search_path: list[PolicyValueMCTSNode], value: float) -> None:
    current_value = value
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += current_value
        current_value = -current_value


__all__ = [
    "AlphaZeroAgent",
    "AlphaZeroConfig",
    "ConnectFourPolicyValueNet",
    "PolicyValueMCTSNode",
    "clone_state_dict",
    "encode_alphazero_state",
    "run_policy_value_mcts",
    "sample_action_from_policy",
]
