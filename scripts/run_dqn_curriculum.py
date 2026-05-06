from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.config import load_config
from connect4_rl.experiments.dqn_curriculum import build_default_dqn_curricula, train_dqn_with_curriculum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DQN curriculum experiments for Connect Four.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--agenda", type=str, default="curriculum_classic", help="Curriculum agenda name")
    parser.add_argument("--episodes", type=int, help="Override episodes from config")
    parser.add_argument("--eval-interval", type=int, help="Override evaluation interval from config")
    parser.add_argument("--eval-games", type=int, help="Override number of evaluation games from config")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Output checkpoint directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.episodes:
        config.dqn.episodes = args.episodes
    if args.eval_interval:
        config.dqn.eval_interval = args.eval_interval
    if args.eval_games:
        config.dqn.eval_games = args.eval_games

    curricula = build_default_dqn_curricula()
    if args.agenda not in curricula:
        raise SystemExit(f"Unknown agenda '{args.agenda}'. Available: {', '.join(sorted(curricula))}")

    checkpoint_dir = args.checkpoint_dir or f"outputs/dqn_{args.agenda}"
    _agent, metrics = train_dqn_with_curriculum(curricula[args.agenda], config, checkpoint_dir=checkpoint_dir)

    summary = {
        "agenda": args.agenda,
        "episodes": len(metrics.episode_rewards),
        "phase_summary": metrics.phase_summary,
        "last_eval": metrics.evaluation[-1] if metrics.evaluation else {},
        "best_checkpoint_path": metrics.best_checkpoint_path,
        "loss_points": len(metrics.losses),
    }
    print("DQN curriculum summary")
    print("======================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
