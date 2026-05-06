from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.config import load_config
from connect4_rl.experiments.alphazero_training import train_alphazero_self_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AlphaZero-style self-play agent for Connect Four.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, help="Override episodes from config")
    parser.add_argument("--eval-interval", type=int, help="Override evaluation interval from config")
    parser.add_argument("--eval-games", type=int, help="Override number of evaluation games from config")
    parser.add_argument("--mcts-simulations", type=int, help="Override MCTS simulations from config")
    parser.add_argument("--eval-mcts-simulations", type=int, help="Override evaluation MCTS simulations from config")
    parser.add_argument("--n-filters", type=int, help="Override number of residual filters from config")
    parser.add_argument("--n-res-blocks", type=int, help="Override number of residual blocks from config")
    parser.add_argument("--temperature", type=float, help="Override self-play temperature from config")
    parser.add_argument("--temperature-drop-move", type=int, help="Override move after which policy becomes greedy")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/alphazero_checkpoints")
    parser.add_argument("--device", type=str, help="Override device from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_config = load_config(args.config)

    if args.episodes:
        project_config.alphazero.episodes = args.episodes
    if args.eval_interval:
        project_config.alphazero.eval_interval = args.eval_interval
    if args.eval_games:
        project_config.alphazero.eval_games = args.eval_games
    if args.mcts_simulations:
        project_config.alphazero.mcts_simulations = args.mcts_simulations
    if args.eval_mcts_simulations:
        project_config.alphazero.eval_mcts_simulations = args.eval_mcts_simulations
    if args.n_filters:
        project_config.alphazero.n_filters = args.n_filters
    if args.n_res_blocks:
        project_config.alphazero.n_res_blocks = args.n_res_blocks
    if args.temperature is not None:
        project_config.alphazero.temperature = args.temperature
    if args.temperature_drop_move is not None:
        project_config.alphazero.temperature_drop_move = args.temperature_drop_move
    if args.device:
        project_config.alphazero.device = args.device
    elif project_config.alphazero.device == "auto":
        project_config.alphazero.device = project_config.resolve_device()

    _agent, metrics = train_alphazero_self_play(project_config.alphazero, checkpoint_dir=args.checkpoint_dir)
    summary = {
        "episodes": len(metrics.episode_rewards),
        "mean_reward_last_20": sum(metrics.episode_rewards[-20:]) / max(len(metrics.episode_rewards[-20:]), 1),
        "last_eval": metrics.evaluation[-1] if metrics.evaluation else {},
        "last_tactical_accuracy": metrics.tactical_accuracy[-1] if metrics.tactical_accuracy else {},
        "best_eval_score": metrics.best_score,
        "best_checkpoint_path": metrics.best_checkpoint_path,
        "policy_updates": len(metrics.policy_losses),
    }
    print("AlphaZero training summary")
    print("==========================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
