from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.agents.baselines import HeuristicAgent, RandomAgent
from connect4_rl.agents.learning.dqn import DQNAgent
from connect4_rl.agents.learning.ppo import PPOAgent
from connect4_rl.agents.training.pretrained import PretrainedAgent
from connect4_rl.agents.planning import MCTSAgent
from connect4_rl.experiments import round_robin_detailed, compute_elo_ratings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Connect Four tournament including pretrained external models.")
    parser.add_argument("--games-per-pair", type=int, default=10, help="Number of games per pair.")
    parser.add_argument("--mcts-simulations", type=int, default=100, help="Simulations for MCTS agent.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    
    # Local models
    dqn_path = root / "notebooks/dqn/outputs/models/dqn/lesson4_trained_agent.pt"
    ppo_path = root / "notebooks/ppo/outputs/ppo_tutorial_full_final_push_midlevel_bc_seed_42/ppo_best.pt"
    
    # Pretrained models
    pretrained_dir = root / "models/pretrained"
    arch_path = pretrained_dir / "architectures" / "cnet128.json"

    factories = {
        "random": lambda: RandomAgent(),
        "heuristic": lambda: HeuristicAgent(),
        f"mcts_{args.mcts_simulations}": lambda: MCTSAgent(simulations=args.mcts_simulations),
    }

    if dqn_path.exists():
        factories["dqn"] = lambda: DQNAgent.from_checkpoint(dqn_path, device=args.device, use_dueling_head=False)
    
    if ppo_path.exists():
        factories["ppo"] = lambda: PPOAgent.from_checkpoint(
            ppo_path, device=args.device, hidden_dim=320,
            channel_sizes=[64, 128, 128], kernel_sizes=[3, 3, 2],
            stride_sizes=[1, 1, 1], head_hidden_sizes=[320, 160]
        )

    # Add Pretrained models
    if (pretrained_dir / "weights" / "best_dqn.pt").exists():
        factories["pretrain_dqn"] = lambda: PretrainedAgent(
            pretrained_dir / "weights" / "best_dqn.pt", arch_path, name="pretrain_dqn", n_heads=1, agent_type="dqn", device=args.device
        )
    
    if (pretrained_dir / "weights" / "best_dueling_dqn.pt").exists():
        factories["pretrain_dueling"] = lambda: PretrainedAgent(
            pretrained_dir / "weights" / "best_dueling_dqn.pt", arch_path, name="pretrain_dueling", n_heads=2, agent_type="dueling", device=args.device
        )
        
    if (pretrained_dir / "weights" / "best_ppo.pt").exists():
        factories["pretrain_ppo"] = lambda: PretrainedAgent(
            pretrained_dir / "weights" / "best_ppo.pt", arch_path, name="pretrain_ppo", n_heads=2, agent_type="ppo", device=args.device
        )

    print(f"Starting tournament with {len(factories)} agents...")
    scoreboard, match_log = round_robin_detailed(factories, games_per_pair=args.games_per_pair)
    elo_ratings = compute_elo_ratings(match_log)

    print("\nTournament Results (including Pretrained Models)")
    print("=" * 90)
    print(f"{'Agent':<20} | {'Elo':<7} | {'Games':<6} | {'Wins':<5} | {'Losses':<6} | {'Draws':<5} | {'Win Rate':<8}")
    print("-" * 90)
    
    sorted_agents = sorted(scoreboard.keys(), key=lambda x: elo_ratings.get(x, 0), reverse=True)
    for name in sorted_agents:
        m = scoreboard[name]
        elo = elo_ratings.get(name, 1200.0)
        print(f"{name:<20} | {elo:<7.1f} | {int(m['games']):<6} | {int(m['wins']):<5} | {int(m['losses']):<6} | {int(m['draws']):<5} | {m['win_rate']:.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
