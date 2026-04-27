from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.agents.baselines import HeuristicAgent, RandomAgent
from connect4_rl.agents.planning import MCTSAgent
from connect4_rl.experiments import round_robin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline Connect Four tournament.")
    parser.add_argument("--games-per-pair", type=int, default=12, help="Number of games for each agent pairing.")
    parser.add_argument("--mcts-simulations", type=int, default=150, help="MCTS simulations per move.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factories = {
        "random": lambda: RandomAgent(),
        "heuristic": lambda: HeuristicAgent(),
        "mcts": lambda: MCTSAgent(simulations=args.mcts_simulations),
    }
    results = round_robin(factories, games_per_pair=args.games_per_pair)

    print("Baseline tournament results")
    print("==========================")
    for name, metrics in sorted(results.items()):
        wins = int(metrics.get("wins", 0))
        losses = int(metrics.get("losses", 0))
        draws = int(metrics.get("draws", 0))
        games = int(metrics.get("games", 0))
        win_rate = float(metrics.get("win_rate", 0.0))
        print(
            f"{name:>10} | games={games:3d} "
            f"wins={wins:3d} losses={losses:3d} "
            f"draws={draws:3d} win_rate={win_rate:.3f}"
        )


if __name__ == "__main__":
    main()
