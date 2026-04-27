from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.agents.learning import DQNConfig
from connect4_rl.experiments.dqn_training import train_dqn_self_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-play DQN agent for Connect Four.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/dqn_checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DQNConfig(
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        device=args.device,
    )
    agent, metrics = train_dqn_self_play(config, checkpoint_dir=args.checkpoint_dir)
    del agent

    summary = {
        "episodes": len(metrics.episode_rewards),
        "mean_reward_last_20": sum(metrics.episode_rewards[-20:]) / max(len(metrics.episode_rewards[-20:]), 1),
        "last_eval": metrics.evaluation[-1] if metrics.evaluation else {},
        "best_eval_score": metrics.best_score,
        "best_checkpoint_path": metrics.best_checkpoint_path,
        "loss_points": len(metrics.losses),
    }
    print("DQN training summary")
    print("====================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
