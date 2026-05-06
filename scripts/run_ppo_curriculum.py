from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.config import load_config
from connect4_rl.experiments.ppo_curriculum import build_default_ppo_curricula, train_dual_ppo_co_training, train_ppo_with_curriculum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO curriculum experiments for Connect Four.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--agenda", type=str, default="curriculum_basic", help="Curriculum agenda name")
    parser.add_argument("--episodes", type=int, help="Override episodes from config")
    parser.add_argument("--eval-interval", type=int, help="Override evaluation interval from config")
    parser.add_argument("--eval-games", type=int, help="Override number of evaluation games from config")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Output checkpoint directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.episodes:
        config.ppo.episodes = args.episodes
    if args.eval_interval:
        config.ppo.eval_interval = args.eval_interval
    if args.eval_games:
        config.ppo.eval_games = args.eval_games

    curricula = build_default_ppo_curricula()
    if args.agenda not in curricula:
        raise SystemExit(f"Unknown agenda '{args.agenda}'. Available: {', '.join(sorted(curricula))}")

    checkpoint_dir = args.checkpoint_dir or f"outputs/ppo_{args.agenda}"
    if args.agenda == "co_training_dual":
        (_agent_a, _agent_b), metrics = train_dual_ppo_co_training(config, checkpoint_dir=checkpoint_dir)
        summary = {
            "agenda": args.agenda,
            "episodes": len(metrics.agent_a.episode_rewards),
            "agent_a_last_eval": metrics.agent_a.evaluation[-1] if metrics.agent_a.evaluation else {},
            "agent_b_last_eval": metrics.agent_b.evaluation[-1] if metrics.agent_b.evaluation else {},
            "last_head_to_head": metrics.head_to_head_win_rate_a[-1] if metrics.head_to_head_win_rate_a else {},
            "agent_a_best_checkpoint": metrics.agent_a.best_checkpoint_path,
            "agent_b_best_checkpoint": metrics.agent_b.best_checkpoint_path,
        }
    else:
        _agent, metrics = train_ppo_with_curriculum(curricula[args.agenda], config, checkpoint_dir=checkpoint_dir)
        summary = {
            "agenda": args.agenda,
            "episodes": len(metrics.episode_rewards),
            "phase_summary": metrics.phase_summary,
            "last_eval": metrics.evaluation[-1] if metrics.evaluation else {},
            "best_checkpoint_path": metrics.best_checkpoint_path,
        }

    print("PPO curriculum summary")
    print("=====================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
