from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.config import load_config
from connect4_rl.experiments.dqn_training import train_dqn_self_play


def main() -> None:
    base_config = load_config(str(ROOT / "config.yaml"))
    episodes = 180
    seeds = [7, 17, 27]

    def build_config(seed: int, *, gradient_updates_per_step: int | None = None, use_symmetry: bool | None = None):
        config = copy.deepcopy(base_config)
        config.global_.seed = seed
        config.dqn.episodes = episodes
        config.dqn.eval_interval = 30
        config.dqn.eval_games = 12
        if gradient_updates_per_step is not None:
            config.dqn.gradient_updates_per_step = gradient_updates_per_step
        if use_symmetry is not None:
            config.dqn.use_horizontal_symmetry_augmentation = use_symmetry
        return config

    candidate_builders = {
        "base": lambda seed: build_config(seed),
        "more_updates": lambda seed: build_config(seed, gradient_updates_per_step=3),
        "no_symmetry": lambda seed: build_config(seed, gradient_updates_per_step=3, use_symmetry=False),
    }

    summary: dict[str, dict[str, float | int | list[dict[str, float | int]]]] = {}
    for name, builder in candidate_builders.items():
        runs = []
        for seed in seeds:
            config = builder(seed)
            _agent, metrics = train_dqn_self_play(config, checkpoint_dir=ROOT / "outputs" / f"dqn_ablation_{name}_seed_{seed}")
            last_eval = metrics.evaluation[-1] if metrics.evaluation else {}
            runs.append(
                {
                    "seed": seed,
                    "best_score": metrics.best_score,
                    "last_vs_random": float(last_eval.get("vs_random_win_rate", 0.0)),
                    "last_vs_heuristic": float(last_eval.get("vs_heuristic_win_rate", 0.0)),
                    "loss_points": len(metrics.losses),
                    "mean_reward_last_20": statistics.fmean(metrics.episode_rewards[-20:]) if metrics.episode_rewards else 0.0,
                }
            )

        summary[name] = {
            "runs": runs,
            "mean_best_score": statistics.fmean(run["best_score"] for run in runs),
            "mean_last_vs_random": statistics.fmean(run["last_vs_random"] for run in runs),
            "mean_last_vs_heuristic": statistics.fmean(run["last_vs_heuristic"] for run in runs),
            "mean_reward_last_20": statistics.fmean(run["mean_reward_last_20"] for run in runs),
        }

    print("DQN ablation summary")
    print("====================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
