from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.agents.learning import AlphaZeroConfig
from connect4_rl.config import load_config
from connect4_rl.experiments.alphazero_training import train_alphazero_self_play


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AlphaZero notebook-equivalent profile for server training and export the outputs.",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--profile", choices=["quick", "full"], default="full", help="Notebook profile to reproduce")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--episodes", type=int, default=None, help="Override total episodes")
    parser.add_argument("--output-root", type=str, default="notebooks/alphazero/outputs", help="Root directory for run outputs")
    parser.add_argument("--archive", action="store_true", help="Create a .tar.gz archive of the run directory at the end")
    return parser.parse_args()


def build_profile_config(profile: str, *, project_config, device: str, seed: int) -> AlphaZeroConfig:
    if profile == "quick":
        settings = {
            "episodes": project_config.notebook_settings.quick_test_episodes,
            "eval_interval": project_config.notebook_settings.quick_test_eval_interval,
            "eval_games": project_config.notebook_settings.quick_test_eval_games,
            "mcts_simulations": project_config.notebook_settings.quick_test_mcts_simulations,
            "eval_mcts_simulations": project_config.notebook_settings.quick_test_eval_mcts_simulations,
            "n_filters": 128,
            "n_res_blocks": 4,
            "updates_per_episode": 1,
            "replay_warmup_games": 12,
            "tactical_eval_examples": 64,
        }
    else:
        settings = {
            "episodes": project_config.alphazero.episodes,
            "eval_interval": project_config.alphazero.eval_interval,
            "eval_games": project_config.alphazero.eval_games,
            "mcts_simulations": project_config.alphazero.mcts_simulations,
            "eval_mcts_simulations": project_config.alphazero.eval_mcts_simulations,
            "n_filters": project_config.alphazero.n_filters,
            "n_res_blocks": project_config.alphazero.n_res_blocks,
            "updates_per_episode": project_config.alphazero.updates_per_episode,
            "replay_warmup_games": project_config.alphazero.replay_warmup_games,
            "tactical_eval_examples": project_config.alphazero.tactical_eval_examples,
        }

    return AlphaZeroConfig(
        **{
            **project_config.alphazero.__dict__,
            **settings,
            "seed": seed,
            "device": device,
        }
    )


def main() -> None:
    args = parse_args()
    project_config = load_config(args.config)

    seed = int(args.seed if args.seed is not None else project_config.notebook_settings.seed)
    device = args.device or project_config.resolve_device()
    training_config = build_profile_config(args.profile, project_config=project_config, device=device, seed=seed)

    if args.episodes is not None:
        training_config.episodes = int(args.episodes)

    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = f"alphazero_self_play_{args.profile}_seed_{training_config.seed}"
    checkpoint_dir = output_root / run_name

    _agent, metrics = train_alphazero_self_play(training_config, checkpoint_dir=checkpoint_dir)

    summary = {
        "run_name": run_name,
        "checkpoint_dir": str(checkpoint_dir),
        "profile": args.profile,
        "episodes": len(metrics.episode_rewards),
        "best_score": metrics.best_score,
        "best_checkpoint_path": metrics.best_checkpoint_path,
        "last_eval": metrics.evaluation[-1] if metrics.evaluation else {},
        "last_tactical_accuracy": metrics.tactical_accuracy[-1] if metrics.tactical_accuracy else {},
        "policy_updates": len(metrics.policy_losses),
    }

    summary_path = checkpoint_dir / "server_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    archive_path = None
    if args.archive:
        archive_base = str(output_root / run_name)
        archive_path = shutil.make_archive(archive_base, "gztar", root_dir=output_root, base_dir=run_name)
        summary["archive_path"] = archive_path
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("AlphaZero server run summary")
    print("============================")
    print(json.dumps(summary, indent=2))
    if archive_path is not None:
        print(f"\nArchive created at: {archive_path}")


if __name__ == "__main__":
    main()
