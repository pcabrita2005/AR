from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.config import load_config
from connect4_rl.experiments.dqn_training import train_dqn_self_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the main DQN self-play pipeline for Connect Four.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, help="Override episodes from config")
    parser.add_argument("--eval-interval", type=int, help="Override evaluation interval from config")
    parser.add_argument("--eval-games", type=int, help="Override number of evaluation games from config")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/dqn_checkpoints")
    parser.add_argument("--lessons-dir", type=str, default=None, help="Directory containing tutorial lesson YAML files")
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
        
    agent, metrics = train_dqn_self_play(config, checkpoint_dir=args.checkpoint_dir, lessons_dir=args.lessons_dir)
    del agent

    summary = {
        "episodes": len(metrics.episode_rewards),
        "curriculum_name": metrics.curriculum_name,
        "mean_reward_last_20": sum(metrics.episode_rewards[-20:]) / max(len(metrics.episode_rewards[-20:]), 1),
        "phase_summary": metrics.phase_summary,
        "lesson_summaries": metrics.lesson_summaries,
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
