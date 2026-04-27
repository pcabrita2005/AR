from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.agents.learning import AlphaZeroConfig
from connect4_rl.experiments.alphazero_training import train_alphazero_self_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simplified AlphaZero-style self-play agent for Connect Four.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--mcts-simulations", type=int, default=120)
    parser.add_argument("--eval-mcts-simulations", type=int, default=200)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/alphazero_checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AlphaZeroConfig(
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        mcts_simulations=args.mcts_simulations,
        eval_mcts_simulations=args.eval_mcts_simulations,
        device=args.device,
    )
    _agent, metrics = train_alphazero_self_play(config, checkpoint_dir=args.checkpoint_dir)
    summary = {
        "episodes": len(metrics.episode_rewards),
        "mean_reward_last_20": sum(metrics.episode_rewards[-20:]) / max(len(metrics.episode_rewards[-20:]), 1),
        "last_eval": metrics.evaluation[-1] if metrics.evaluation else {},
        "best_eval_score": metrics.best_score,
        "best_checkpoint_path": metrics.best_checkpoint_path,
        "policy_updates": len(metrics.policy_losses),
    }
    print("AlphaZero training summary")
    print("==========================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
