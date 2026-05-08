from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.config import load_config
from connect4_rl.experiments.ppo_notebook_variants import apply_variant_to_config, get_variant_spec
from connect4_rl.experiments.ppo_training import train_ppo_self_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-play PPO agent for Connect Four.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, help="Override episodes from config")
    parser.add_argument("--eval-interval", type=int, help="Override evaluation interval from config")
    parser.add_argument("--eval-games", type=int, help="Override number of evaluation games from config")
    parser.add_argument("--variant", type=str, default="baseline", help="Notebook PPO variant to apply")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/ppo_checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_variant_to_config(load_config(args.config), args.variant)
    
    if args.episodes:
        config.ppo.episodes = args.episodes
    if args.eval_interval:
        config.ppo.eval_interval = args.eval_interval
    if args.eval_games:
        config.ppo.eval_games = args.eval_games
        
    _agent, metrics = train_ppo_self_play(config, checkpoint_dir=args.checkpoint_dir)
    summary = {
        "variant": args.variant,
        "variant_description": get_variant_spec(args.variant)["description"],
        "episodes": len(metrics.episode_rewards),
        "mean_reward_last_20": sum(metrics.episode_rewards[-20:]) / max(len(metrics.episode_rewards[-20:]), 1),
        "last_eval": metrics.evaluation[-1] if metrics.evaluation else {},
        "best_eval_score": metrics.best_score,
        "best_checkpoint_path": metrics.best_checkpoint_path,
        "best_vs_strong_checkpoint_path": metrics.best_vs_strong_checkpoint_path,
        "policy_updates": len(metrics.policy_losses),
    }
    print("PPO training summary")
    print("====================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
