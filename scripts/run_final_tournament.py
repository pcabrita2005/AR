from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.agents.baselines import HeuristicAgent, RandomAgent
from connect4_rl.agents.planning import MCTSAgent
from connect4_rl.agents.learning.dqn import DQNAgent
from connect4_rl.agents.learning.ppo import PPOAgent
from connect4_rl.experiments import round_robin_detailed, compute_elo_ratings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final Connect Four tournament.")
    parser.add_argument("--games-per-pair", type=int, default=50, help="Number of games for each agent pairing.")
    parser.add_argument("--mcts-simulations", type=int, default=100, help="MCTS simulations per move.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run neural networks on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    dqn_path = ROOT / "notebooks/dqn/outputs/models/dqn/lesson4_trained_agent.pt"
    ppo_path = ROOT / "notebooks/ppo/outputs/ppo_tutorial_full_final_push_midlevel_bc_seed_42/ppo_best.pt"
    
    print(f"Loading agents...")
    
    factories = {
        "random": lambda: RandomAgent(),
        "heuristic": lambda: HeuristicAgent(),
        "mcts_100": lambda: MCTSAgent(simulations=args.mcts_simulations),
    }
    
    if dqn_path.exists():
        print(f"Found DQN best model at {dqn_path}")
        factories["dqn_best"] = lambda: DQNAgent.from_checkpoint(
            dqn_path, 
            device=args.device,
            use_dueling_head=False
        )
    else:
        print(f"Warning: DQN best model not found at {dqn_path}")

    if ppo_path.exists():
        print(f"Found PPO best model at {ppo_path}")
        factories["ppo_best"] = lambda: PPOAgent.from_checkpoint(
            ppo_path, 
            device=args.device,
            hidden_dim=320,
            channel_sizes=[64, 128, 128],
            kernel_sizes=[3, 3, 2],
            stride_sizes=[1, 1, 1],
            head_hidden_sizes=[320, 160]
        )
    else:
        print(f"Warning: PPO best model not found at {ppo_path}")

    print(f"\nStarting tournament with {len(factories)} agents...")
    print(f"Games per pair: {args.games_per_pair}")
    
    scoreboard, match_log = round_robin_detailed(factories, games_per_pair=args.games_per_pair)
    elo_ratings = compute_elo_ratings(match_log)

    print("\nFinal Tournament Results")
    print("=" * 80)
    print(f"{'Agent':<15} | {'Elo':<6} | {'Games':<6} | {'Wins':<5} | {'Losses':<6} | {'Draws':<5} | {'Win Rate':<8}")
    print("-" * 80)
    
    # Sort agents by Elo
    sorted_agents = sorted(scoreboard.keys(), key=lambda x: elo_ratings.get(x, 0), reverse=True)
    
    for name in sorted_agents:
        m = scoreboard[name]
        elo = elo_ratings.get(name, 1200.0)
        print(
            f"{name:<15} | {elo:<6.1f} | {int(m['games']):<6} | {int(m['wins']):<5} | "
            f"{int(m['losses']):<6} | {int(m['draws']):<5} | {m['win_rate']:.3f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
