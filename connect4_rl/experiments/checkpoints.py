from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from connect4_rl.agents.baselines import HeuristicAgent, RandomAgent, StrongHeuristicAgent, WeakHeuristicAgent
from connect4_rl.agents.learning import AlphaZeroAgent, DQNAgent, PPOAgent
from connect4_rl.agents.planning import MCTSAgent


@dataclass(frozen=True)
class CompletedRun:
    algorithm: str
    metrics_path: Path
    data: dict[str, Any]

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.data["best_checkpoint_path"])

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.data.get("config", {}))


def find_best_run(outputs_root: str | Path, algorithm: str) -> CompletedRun | None:
    root = Path(outputs_root)
    candidates: list[CompletedRun] = []

    for metrics_path in sorted(root.glob("**/metrics_final.json")):
        if algorithm not in metrics_path.parent.name:
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        if data.get("best_checkpoint_path"):
            candidates.append(
                CompletedRun(
                    algorithm=algorithm,
                    metrics_path=metrics_path,
                    data=data,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda run: (
            float(run.data.get("best_score", float("-inf"))),
            run.metrics_path.stat().st_mtime,
        ),
        reverse=True,
    )
    return candidates[0]


def build_agent_from_checkpoint(
    algorithm: str,
    checkpoint_path: str | Path,
    config: dict[str, Any],
    *,
    device: str,
) -> object:
    path = Path(checkpoint_path)

    if algorithm == "dqn":
        return DQNAgent.from_checkpoint(
            path,
            device=device,
            epsilon=0.0,
            seed=int(config.get("seed", 0)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            channel_sizes=config.get("channel_sizes"),
            kernel_sizes=config.get("kernel_sizes"),
            stride_sizes=config.get("stride_sizes"),
            head_hidden_sizes=config.get("head_hidden_sizes"),
            use_dueling_head=bool(config.get("use_dueling_head", True)),
        )
    if algorithm == "ppo":
        return PPOAgent.from_checkpoint(
            path,
            device=device,
            sample_actions=False,
            seed=int(config.get("seed", 0)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            channel_sizes=config.get("channel_sizes"),
            kernel_sizes=config.get("kernel_sizes"),
            stride_sizes=config.get("stride_sizes"),
            head_hidden_sizes=config.get("head_hidden_sizes"),
        )
    if algorithm == "alphazero":
        return AlphaZeroAgent.from_checkpoint(
            path,
            device=device,
            simulations=int(config.get("eval_mcts_simulations") or config.get("mcts_simulations", 40)),
            c_puct=float(config.get("c_puct", 1.5)),
            seed=int(config.get("seed", 0)),
            n_filters=int(config.get("n_filters", 128)),
            n_res_blocks=int(config.get("n_res_blocks", 8)),
            temperature=0.0,
        )
    raise ValueError(f"Unsupported checkpoint algorithm: {algorithm}")


def build_agent_from_run(run: CompletedRun, *, root: str | Path, device: str) -> object:
    checkpoint_path = Path(root) / run.checkpoint_path
    return build_agent_from_checkpoint(
        run.algorithm,
        checkpoint_path,
        run.config,
        device=device,
    )


def build_agent_factory_from_run(
    run: CompletedRun,
    *,
    root: str | Path,
    device: str,
) -> Callable[[], object]:
    checkpoint_path = Path(root) / run.checkpoint_path
    config = run.config
    algorithm = run.algorithm
    return lambda cp=checkpoint_path, cfg=config, algo=algorithm: build_agent_from_checkpoint(
        algo,
        cp,
        cfg,
        device=device,
    )


def build_reference_factory(name: str, *, seed: int, mcts_simulations: int | None = None) -> Callable[[], object]:
    if name == "random":
        return lambda: RandomAgent(seed=seed)
    if name == "weak":
        return lambda: WeakHeuristicAgent(seed=seed)
    if name == "strong":
        return lambda: StrongHeuristicAgent(seed=seed)
    if name == "heuristic":
        return lambda: HeuristicAgent(seed=seed)
    if name == "mcts":
        return lambda: MCTSAgent(simulations=int(mcts_simulations or 100), rollout_seed=seed)
    raise ValueError(f"Unsupported reference agent: {name}")


__all__ = [
    "CompletedRun",
    "build_agent_factory_from_run",
    "build_agent_from_checkpoint",
    "build_agent_from_run",
    "build_reference_factory",
    "find_best_run",
]
