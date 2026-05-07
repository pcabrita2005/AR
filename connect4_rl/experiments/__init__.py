"""Experiment utilities."""

from .alphazero_training import train_alphazero_self_play
from .checkpoints import (
    CompletedRun,
    build_agent_factory_from_run,
    build_agent_from_checkpoint,
    build_agent_from_run,
    build_reference_factory,
    find_best_run,
)
from .dqn_training import evaluate_against_agent, train_dqn_self_play
from .evaluation import compute_elo_ratings, play_match, round_robin, round_robin_detailed
from .ppo_curriculum import build_default_ppo_curricula, train_dual_ppo_co_training, train_ppo_with_curriculum
from .ppo_training import train_ppo_self_play

__all__ = [
    "CompletedRun",
    "build_agent_factory_from_run",
    "build_agent_from_checkpoint",
    "build_agent_from_run",
    "build_default_ppo_curricula",
    "build_reference_factory",
    "compute_elo_ratings",
    "evaluate_against_agent",
    "find_best_run",
    "play_match",
    "round_robin",
    "round_robin_detailed",
    "train_dual_ppo_co_training",
    "train_alphazero_self_play",
    "train_dqn_self_play",
    "train_ppo_with_curriculum",
    "train_ppo_self_play",
]
