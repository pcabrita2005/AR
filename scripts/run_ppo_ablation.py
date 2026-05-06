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
from connect4_rl.experiments.ppo_training import train_ppo_self_play


def main() -> None:
    base_config = load_config(str(ROOT / "config.yaml"))
    episodes = 180
    seeds = [7, 17, 27]

    def build_config(
        seed: int,
        *,
        reward_shaping: bool | None = None,
        learning_rate: float | None = None,
        entropy_coeff: float | None = None,
        rollout_episodes_per_update: int | None = None,
    ):
        config = copy.deepcopy(base_config)
        config.global_.seed = seed
        config.ppo.episodes = episodes
        config.ppo.eval_interval = 30
        config.ppo.eval_games = 12
        if reward_shaping is not None:
            config.ppo.reward_shaping = reward_shaping
        if learning_rate is not None:
            config.ppo.learning_rate = learning_rate
        if entropy_coeff is not None:
            config.ppo.entropy_coeff = entropy_coeff
        if rollout_episodes_per_update is not None:
            config.ppo.rollout_episodes_per_update = rollout_episodes_per_update
        return config

    candidate_builders = {
        "base": lambda seed: build_config(seed),
        "no_shaping": lambda seed: build_config(seed, reward_shaping=False),
        "conservative": lambda seed: build_config(
            seed,
            learning_rate=1.5e-4,
            entropy_coeff=0.01,
            rollout_episodes_per_update=12,
        ),
    }

    summary: dict[str, dict[str, float | int | list[dict[str, float | int]]]] = {}
    for name, builder in candidate_builders.items():
        runs = []
        for seed in seeds:
            config = builder(seed)
            _agent, metrics = train_ppo_self_play(config, checkpoint_dir=ROOT / "outputs" / f"ppo_ablation_{name}_seed_{seed}")
            last_eval = metrics.evaluation[-1] if metrics.evaluation else {}
            runs.append(
                {
                    "seed": seed,
                    "best_score": metrics.best_score,
                    "last_vs_random": float(last_eval.get("vs_random_win_rate", 0.0)),
                    "last_vs_heuristic": float(last_eval.get("vs_heuristic_win_rate", 0.0)),
                    "policy_updates": len(metrics.policy_losses),
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

    print("PPO ablation summary")
    print("====================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
