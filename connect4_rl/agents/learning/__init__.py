"""Learning-based agents."""

from .alphazero import AlphaZeroAgent, AlphaZeroConfig, ConnectFourPolicyValueNet
from .dqn import DQNAgent, DQNConfig, ConnectFourQNetwork
from .ppo import PPOAgent, ConnectFourActorCritic
from connect4_rl.config import PPOConfig

__all__ = [
    "AlphaZeroAgent",
    "AlphaZeroConfig",
    "ConnectFourActorCritic",
    "ConnectFourPolicyValueNet",
    "ConnectFourQNetwork",
    "DQNAgent",
    "DQNConfig",
    "PPOAgent",
    "PPOConfig",
]
