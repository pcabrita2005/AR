#!/usr/bin/env python
"""Benchmark AlphaZero replay capacity trade-offs.

Runs quick training jobs with different replay buffer capacities and reports:
- elapsed time
- throughput
- estimated time for 800 episodes
- reward statistics
"""

from __future__ import annotations

import contextlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from connect4_rl.config import AlphaZeroConfig, load_config
from connect4_rl.experiments.alphazero_training import train_alphazero_self_play

# Resolve project root (find config.yaml)
ROOT = Path(__file__).parent
while not (ROOT / "config.yaml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
ROOT = ROOT.resolve()


def build_config(base: AlphaZeroConfig, *, replay_capacity: int, episodes: int, seed: int) -> AlphaZeroConfig:
    return AlphaZeroConfig(
        **{
            **base.__dict__,
            "replay_capacity": replay_capacity,
            "episodes": episodes,
            "seed": seed,
            "device": "auto",
            # Keep evaluation lighter for a quick benchmark.
            "eval_interval": max(20, episodes // 4),
            "eval_games": 6,
        }
    )


def run_single(base: AlphaZeroConfig, output_dir: Path, replay_capacity: int, episodes: int, seed: int) -> dict[str, float | int | str]:
    cfg = build_config(base, replay_capacity=replay_capacity, episodes=episodes, seed=seed)
    run_dir = output_dir / f"replay_{replay_capacity}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            _, metrics = train_alphazero_self_play(cfg, checkpoint_dir=run_dir)
    elapsed = time.time() - start

    rewards = [float(v) for v in metrics.episode_rewards]
    throughput = len(rewards) / elapsed if elapsed > 0 else 0.0

    return {
        "replay_capacity": replay_capacity,
        "episodes": len(rewards),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "throughput_eps_per_sec": throughput,
        "estimated_800ep_minutes": (800 / throughput) / 60.0 if throughput > 0 else float("inf"),
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
        "best_score": float(metrics.best_score),
        "updates": len(metrics.policy_losses),
        "log_path": str(log_path),
    }


def print_table(results: list[dict[str, float | int | str]]) -> None:
    print("\nReplay Capacity Benchmark")
    print("=" * 95)
    print(
        f"{'capacity':>10} | {'episodes':>8} | {'time (min)':>10} | {'ep/min':>8} | {'est 800 (min)':>13} | {'reward mean':>11} | {'reward std':>10}"
    )
    print("-" * 95)
    for item in results:
        ep_per_min = float(item["throughput_eps_per_sec"]) * 60.0
        print(
            f"{int(item['replay_capacity']):>10} | {int(item['episodes']):>8} | {float(item['elapsed_minutes']):>10.2f} | {ep_per_min:>8.2f} | {float(item['estimated_800ep_minutes']):>13.2f} | {float(item['reward_mean']):>11.4f} | {float(item['reward_std']):>10.4f}"
        )


def main() -> None:
    base_config = load_config(ROOT / "config.yaml").alphazero

    # Quick benchmark settings.
    capacities = [5000, 10000, 20000]
    episodes = 100
    seed = 42

    output_dir = ROOT / "notebooks" / "alphazero" / "outputs" / "replay_capacity_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting replay capacity benchmark")
    print(f"Capacities: {capacities}")
    print(f"Episodes per run: {episodes}")
    print(f"Base episodes_per_batch: {base_config.episodes_per_batch}")
    print(f"Base updates_per_episode: {base_config.updates_per_episode}")

    results: list[dict[str, float | int | str]] = []
    for cap in capacities:
        print(f"\nRunning replay_capacity={cap} ...")
        result = run_single(base_config, output_dir, cap, episodes, seed)
        results.append(result)
        print(
            f"Done cap={cap}: {result['elapsed_minutes']:.2f} min, "
            f"est_800={result['estimated_800ep_minutes']:.2f} min"
        )

    results_path = output_dir / "benchmark_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_table(results)

    fastest = min(results, key=lambda r: float(r["elapsed_seconds"]))
    print("\nSuggested capacity by speed:", int(fastest["replay_capacity"]))
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
