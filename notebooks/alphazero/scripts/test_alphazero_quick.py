#!/usr/bin/env python
"""
Quick AlphaZero test: 100 episodes with optimized hyperparams.
Measures: time, GPU utilization, reward stability, throughput.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import asdict

import torch
import numpy as np

# Resolve project root
ROOT = Path(__file__).parent
while not (ROOT / "config.yaml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
ROOT = ROOT.resolve()

from connect4_rl.config import load_config, AlphaZeroConfig
from connect4_rl.experiments.alphazero_training import train_alphazero_self_play


def main():
    # Load config
    config_dict = load_config(ROOT / "config.yaml")
    
    # Create optimized test config (100 episodes instead of 800)
    test_config = AlphaZeroConfig(
        **{
            **config_dict.alphazero.__dict__,
            "episodes": 100,  # Quick test
            "eval_interval": 25,  # Eval every 25 episodes
            "eval_games": 6,  # Fewer evals for speed
            "seed": 42,
            "device": "auto",
            "episodes_per_batch": 8,  # From config.yaml
            "updates_per_episode": 4,  # From config.yaml
            "batch_size": 256,  # From config.yaml
        }
    )
    
    # Create output dir
    output_dir = ROOT / "notebooks" / "alphazero" / "outputs" / "test_quick_100ep"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AlphaZero Quick Test: 100 Episodes")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  episodes_per_batch: {test_config.episodes_per_batch}")
    print(f"  updates_per_episode: {test_config.updates_per_episode}")
    print(f"  batch_size: {test_config.batch_size}")
    print(f"  learning_rate: {test_config.learning_rate}")
    print(f"  device: {test_config.device}")
    print(f"  total episodes: {test_config.episodes}")
    print()
    
    # Measure GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory (initial): {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print("\nStarting training...\n")
    start_time = time.time()
    
    try:
        agent, metrics = train_alphazero_self_play(test_config, checkpoint_dir=output_dir)
        elapsed = time.time() - start_time
        
        # Compute stats
        rewards = metrics.episode_rewards
        mean_reward = np.mean(rewards) if rewards else 0.0
        std_reward = np.std(rewards) if len(rewards) > 1 else 0.0
        min_reward = np.min(rewards) if rewards else 0.0
        max_reward = np.max(rewards) if rewards else 0.0
        
        # Estimate throughput
        episodes_trained = len(rewards)
        throughput = episodes_trained / elapsed if elapsed > 0 else 0.0
        
        # GPU memory
        gpu_memory_mb = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6
        
        print("\n" + "=" * 70)
        print("Test Results")
        print("=" * 70)
        print(f"Elapsed time: {elapsed / 60:.2f} minutes ({elapsed:.0f}s)")
        print(f"Episodes trained: {episodes_trained}")
        print(f"Throughput: {throughput:.2f} episodes/sec")
        print(f"Estimated time for 800 episodes: {(800 / throughput) / 60:.1f} minutes")
        print()
        print(f"Reward statistics (across {len(rewards)} episodes):")
        print(f"  Mean: {mean_reward:+.4f}")
        print(f"  Std:  {std_reward:.4f}")
        print(f"  Min:  {min_reward:+.4f}")
        print(f"  Max:  {max_reward:+.4f}")
        print()
        if device.type == "cuda":
            print(f"GPU Peak Memory: {gpu_memory_mb / 1024:.2f} GB")
        print()
        
        # Save results
        results = {
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "episodes_trained": episodes_trained,
            "throughput_eps_per_sec": throughput,
            "estimated_800ep_minutes": (800 / throughput) / 60 if throughput > 0 else None,
            "reward_mean": float(mean_reward),
            "reward_std": float(std_reward),
            "reward_min": float(min_reward),
            "reward_max": float(max_reward),
            "gpu_peak_memory_mb": gpu_memory_mb,
            "best_score": metrics.best_score,
            "num_updates": len(metrics.policy_losses),
            "best_checkpoint": metrics.best_checkpoint_path,
        }
        
        results_path = output_dir / "test_results.json"
        results_path.write_text(json.dumps(results, indent=2))
        print(f"Results saved to: {results_path}\n")
        
        return results
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n⚠️ Training interrupted after {elapsed / 60:.1f} minutes")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n\n❌ Error after {elapsed / 60:.1f} minutes:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
