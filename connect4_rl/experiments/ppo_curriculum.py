from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from connect4_rl.agents.baselines import HeuristicAgent, RandomAgent
from connect4_rl.agents.learning.ppo import ConnectFourActorCritic, PPOAgent
from connect4_rl.config import Config
from connect4_rl.envs.connect_four import apply_action, encode_state, initial_state, is_terminal, legal_actions, outcome_for_player
from connect4_rl.utils.seed_utils import set_all_seeds

from .ppo_training import (
    PPOMetrics,
    _load_previous_ppo_network,
    build_fixed_opponent,
    clone_state_dict,
    compute_step_reward,
    evaluate_against_agent,
    finalize_last_transition,
    legal_actions_to_mask,
    maybe_anneal_learning_rate,
    sample_policy_action,
    update_ppo,
    write_metrics_snapshot,
)


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    opponent_kind: str | None
    fraction: float
    opponent_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class CurriculumDefinition:
    name: str
    description: str
    phases: tuple[CurriculumPhase, ...]


@dataclass
class PPOCurriculumMetrics(PPOMetrics):
    curriculum_name: str = ""
    curriculum_description: str = ""
    phase_sequence: list[str] = field(default_factory=list)
    phase_summary: list[dict[str, object]] = field(default_factory=list)


@dataclass
class DualAgentMetrics:
    label: str
    best_checkpoint_path: str = ""
    best_score: float = float("-inf")
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    evaluation: list[dict[str, float]] = field(default_factory=list)


@dataclass
class DualPPOTrainingMetrics:
    config: dict[str, object]
    curriculum_name: str
    curriculum_description: str
    agent_a: DualAgentMetrics
    agent_b: DualAgentMetrics
    head_to_head_win_rate_a: list[dict[str, float]] = field(default_factory=list)


def build_default_ppo_curricula() -> dict[str, CurriculumDefinition]:
    return {
        "curriculum_basic": CurriculumDefinition(
            name="curriculum_basic",
            description="Comeca com random, passa pelo heuristico e termina em self-play.",
            phases=(
                CurriculumPhase("warmup_random", "random", 0.25),
                CurriculumPhase("guided_heuristic", "heuristic", 0.25),
                CurriculumPhase("self_play_finish", "self_play", 0.50),
            ),
        ),
        "curriculum_mid_self": CurriculumDefinition(
            name="curriculum_mid_self",
            description="Introduce self-play cedo, regressa ao heuristico no meio e fecha novamente em self-play.",
            phases=(
                CurriculumPhase("warmup_random", "random", 0.25),
                CurriculumPhase("early_self_play", "self_play", 0.25),
                CurriculumPhase("mid_heuristic", "heuristic", 0.25),
                CurriculumPhase("final_self_play", "self_play", 0.25),
            ),
        ),
        "curriculum_late_heuristic": CurriculumDefinition(
            name="curriculum_late_heuristic",
            description="Faz warmup random, investe mais cedo em self-play e usa o heuristico apenas no fim.",
            phases=(
                CurriculumPhase("warmup_random", "random", 0.25),
                CurriculumPhase("long_self_play", "self_play", 0.50),
                CurriculumPhase("late_heuristic", "heuristic", 0.25),
            ),
        ),
        "curriculum_short_heuristic_mid": CurriculumDefinition(
            name="curriculum_short_heuristic_mid",
            description="Usa um bloco heuristico curto no meio, rodeado por self-play.",
            phases=(
                CurriculumPhase("warmup_random", "random", 0.25),
                CurriculumPhase("self_play_growth", "self_play", 0.35),
                CurriculumPhase("short_heuristic_probe", "heuristic", 0.10),
                CurriculumPhase("self_play_refinement", "self_play", 0.30),
            ),
        ),
        "curriculum_short_heuristic_late": CurriculumDefinition(
            name="curriculum_short_heuristic_late",
            description="Mantem a fase heuristica curta e apenas no fim, evitando um bloco demasiado longo.",
            phases=(
                CurriculumPhase("warmup_random", "random", 0.25),
                CurriculumPhase("long_self_play", "self_play", 0.60),
                CurriculumPhase("short_late_heuristic", "heuristic", 0.15),
            ),
        ),
        "curriculum_probabilistic_bridge": CurriculumDefinition(
            name="curriculum_probabilistic_bridge",
            description="Faz a transicao para self-play com uma fase mista onde o heuristico aparece pouco mas de forma recorrente.",
            phases=(
                CurriculumPhase("warmup_random", "random", 0.25),
                CurriculumPhase(
                    "mixed_bridge",
                    None,
                    0.50,
                    opponent_weights={"self_play": 0.75, "heuristic": 0.15, "random": 0.10},
                ),
                CurriculumPhase("self_play_finish", "self_play", 0.25),
            ),
        ),
        "co_training_dual": CurriculumDefinition(
            name="co_training_dual",
            description="Dois PPO distintos aprendem em simultaneo jogando um contra o outro.",
            phases=(CurriculumPhase("dual_self_play", "dual", 1.0),),
        ),
    }


def allocate_phase_episodes(total_episodes: int, phases: tuple[CurriculumPhase, ...]) -> list[tuple[CurriculumPhase, int]]:
    raw_counts = [phase.fraction * total_episodes for phase in phases]
    counts = [int(value) for value in raw_counts]
    remainder = total_episodes - sum(counts)
    fractional_order = sorted(
        range(len(phases)),
        key=lambda idx: raw_counts[idx] - counts[idx],
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        counts[idx] += 1
    return [(phase, count) for phase, count in zip(phases, counts) if count > 0]


def expand_curriculum_schedule(
    total_episodes: int,
    definition: CurriculumDefinition,
    *,
    seed: int = 0,
) -> tuple[list[str], list[dict[str, object]]]:
    schedule: list[str] = []
    summary: list[dict[str, object]] = []
    cursor = 1
    rng = random.Random(seed)
    for phase, count in allocate_phase_episodes(total_episodes, definition.phases):
        start_episode = cursor
        end_episode = cursor + count - 1
        if phase.opponent_weights:
            phase_schedule = _sample_phase_schedule(phase.opponent_weights, count, rng)
            opponent_kind = "mixed"
            realized_counts = dict(Counter(phase_schedule))
        else:
            assert phase.opponent_kind is not None
            phase_schedule = [phase.opponent_kind] * count
            opponent_kind = phase.opponent_kind
            realized_counts = {phase.opponent_kind: count}
        schedule.extend(phase_schedule)
        summary.append(
            {
                "phase_name": phase.name,
                "opponent_kind": opponent_kind,
                "episodes": count,
                "start_episode": start_episode,
                "end_episode": end_episode,
                "realized_opponents": realized_counts,
            }
        )
        cursor = end_episode + 1
    return schedule, summary


def _sample_phase_schedule(
    opponent_weights: dict[str, float],
    count: int,
    rng: random.Random,
) -> list[str]:
    kinds = list(opponent_weights.keys())
    weights = [float(opponent_weights[kind]) for kind in kinds]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("opponent_weights must sum to a positive value")
    normalized = [weight / total_weight for weight in weights]
    return rng.choices(kinds, weights=normalized, k=count)


def train_ppo_with_curriculum(
    definition: CurriculumDefinition,
    config: Config | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
) -> tuple[PPOAgent, PPOCurriculumMetrics]:
    from connect4_rl.config import get_default_config

    config = config or get_default_config()
    set_all_seeds(config.global_.seed)
    rng = random.Random(config.global_.seed)
    device = torch.device(config.resolve_device())
    network = ConnectFourActorCritic(hidden_dim=config.ppo.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.ppo.learning_rate)

    metrics = PPOCurriculumMetrics(
        config=asdict(config.ppo),
        curriculum_name=definition.name,
        curriculum_description=definition.description,
    )
    best_state_dict = clone_state_dict(network)
    previous_eval_state_dict: dict[str, torch.Tensor] | None = None
    opponent_pool: list[dict[str, torch.Tensor]] = [clone_state_dict(network)]
    rollout_buffer: list[dict[str, object]] = []

    schedule, phase_summary = expand_curriculum_schedule(
        config.ppo.episodes,
        definition,
        seed=config.global_.seed,
    )
    metrics.phase_sequence = schedule
    metrics.phase_summary = phase_summary

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    for episode in range(1, config.ppo.episodes + 1):
        opponent_kind = schedule[episode - 1]

        if opponent_kind == "self_play":
            opponent_state_dict = rng.choice(opponent_pool)
            opponent_network = ConnectFourActorCritic(hidden_dim=config.ppo.hidden_dim).to(device)
            opponent_network.load_state_dict(opponent_state_dict)
            opponent_network.eval()
            trajectories, episode_reward, episode_steps = collect_curriculum_self_play_episode(
                network,
                device,
                config,
                opponent_network=opponent_network,
            )
            for trajectory in trajectories:
                if trajectory:
                    rollout_buffer.extend(
                        augment_trajectory(trajectory) if config.ppo.use_horizontal_symmetry_augmentation else trajectory
                    )
        else:
            opponent_agent = build_fixed_opponent(opponent_kind, config.global_.seed + episode)
            controlled_player = 1 if rng.random() < 0.5 else 2
            trajectory, episode_reward, episode_steps = collect_policy_episode_against_opponent(
                network,
                device,
                opponent_agent,
                controlled_player,
                config,
            )
            if trajectory:
                rollout_buffer.extend(augment_trajectory(trajectory) if config.ppo.use_horizontal_symmetry_augmentation else trajectory)

        if rollout_buffer and (episode % config.ppo.rollout_episodes_per_update == 0 or episode == config.ppo.episodes):
            maybe_anneal_learning_rate(optimizer, config, episode)
            policy_loss, value_loss, entropy = update_ppo(network, optimizer, rollout_buffer, config, device)
            opponent_pool.append(clone_state_dict(network))
            opponent_pool = opponent_pool[-config.ppo.opponent_pool_size :]
            metrics.policy_losses.append(policy_loss)
            metrics.value_losses.append(value_loss)
            metrics.entropies.append(entropy)
            rollout_buffer = []

        metrics.episode_rewards.append(episode_reward)
        metrics.episode_lengths.append(episode_steps)
        metrics.opponent_kinds.append(opponent_kind)

        if checkpoint_path is not None and (episode % config.ppo.eval_interval == 0 or episode == config.ppo.episodes):
            torch.save(network.state_dict(), checkpoint_path / f"ppo_{definition.name}_episode_{episode:04d}.pt")

        if episode % config.ppo.eval_interval == 0 or episode == config.ppo.episodes:
            best_state_dict = _run_single_agent_evaluation(
                network=network,
                config=config,
                metrics=metrics,
                previous_eval_state_dict=previous_eval_state_dict,
                checkpoint_path=checkpoint_path,
                best_state_dict=best_state_dict,
            )
            previous_eval_state_dict = clone_state_dict(network)

    network.load_state_dict(best_state_dict)
    final_agent = PPOAgent(network, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed)
    if checkpoint_path is not None:
        write_metrics_snapshot(metrics, checkpoint_path / "metrics_final.json")
    return final_agent, metrics


def train_dual_ppo_co_training(
    config: Config | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
) -> tuple[tuple[PPOAgent, PPOAgent], DualPPOTrainingMetrics]:
    from connect4_rl.config import get_default_config

    config = config or get_default_config()
    set_all_seeds(config.global_.seed)
    rng = random.Random(config.global_.seed)
    device = torch.device(config.resolve_device())

    network_a = ConnectFourActorCritic(hidden_dim=config.ppo.hidden_dim).to(device)
    network_b = ConnectFourActorCritic(hidden_dim=config.ppo.hidden_dim).to(device)
    optimizer_a = torch.optim.AdamW(network_a.parameters(), lr=config.ppo.learning_rate)
    optimizer_b = torch.optim.AdamW(network_b.parameters(), lr=config.ppo.learning_rate)
    best_state_a = clone_state_dict(network_a)
    best_state_b = clone_state_dict(network_b)
    previous_eval_a: dict[str, torch.Tensor] | None = None
    previous_eval_b: dict[str, torch.Tensor] | None = None
    buffer_a: list[dict[str, object]] = []
    buffer_b: list[dict[str, object]] = []

    metrics = DualPPOTrainingMetrics(
        config=asdict(config.ppo),
        curriculum_name="co_training_dual",
        curriculum_description=build_default_ppo_curricula()["co_training_dual"].description,
        agent_a=DualAgentMetrics(label="agent_a"),
        agent_b=DualAgentMetrics(label="agent_b"),
    )

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    for episode in range(1, config.ppo.episodes + 1):
        agent_a_player = 1 if episode % 2 == 1 else 2
        trajectory_a, trajectory_b, reward_a, reward_b, episode_steps = collect_dual_policy_episode(
            network_a,
            network_b,
            device,
            config,
            agent_a_player=agent_a_player,
        )
        buffer_a.extend(augment_trajectory(trajectory_a) if config.ppo.use_horizontal_symmetry_augmentation else trajectory_a)
        buffer_b.extend(augment_trajectory(trajectory_b) if config.ppo.use_horizontal_symmetry_augmentation else trajectory_b)

        if buffer_a and (episode % config.ppo.rollout_episodes_per_update == 0 or episode == config.ppo.episodes):
            maybe_anneal_learning_rate(optimizer_a, config, episode)
            maybe_anneal_learning_rate(optimizer_b, config, episode)
            policy_loss_a, value_loss_a, entropy_a = update_ppo(network_a, optimizer_a, buffer_a, config, device)
            policy_loss_b, value_loss_b, entropy_b = update_ppo(network_b, optimizer_b, buffer_b, config, device)
            metrics.agent_a.policy_losses.append(policy_loss_a)
            metrics.agent_a.value_losses.append(value_loss_a)
            metrics.agent_a.entropies.append(entropy_a)
            metrics.agent_b.policy_losses.append(policy_loss_b)
            metrics.agent_b.value_losses.append(value_loss_b)
            metrics.agent_b.entropies.append(entropy_b)
            buffer_a = []
            buffer_b = []

        metrics.agent_a.episode_rewards.append(reward_a)
        metrics.agent_a.episode_lengths.append(episode_steps)
        metrics.agent_b.episode_rewards.append(reward_b)
        metrics.agent_b.episode_lengths.append(episode_steps)

        if checkpoint_path is not None and (episode % config.ppo.eval_interval == 0 or episode == config.ppo.episodes):
            torch.save(network_a.state_dict(), checkpoint_path / f"agent_a_episode_{episode:04d}.pt")
            torch.save(network_b.state_dict(), checkpoint_path / f"agent_b_episode_{episode:04d}.pt")

        if episode % config.ppo.eval_interval == 0 or episode == config.ppo.episodes:
            best_state_a, previous_eval_a = _run_dual_agent_evaluation(
                label="agent_a",
                network=network_a,
                config=config,
                metrics=metrics.agent_a,
                previous_eval_state_dict=previous_eval_a,
                opponent_network=network_b,
                checkpoint_path=checkpoint_path,
                best_state_dict=best_state_a,
            )
            best_state_b, previous_eval_b = _run_dual_agent_evaluation(
                label="agent_b",
                network=network_b,
                config=config,
                metrics=metrics.agent_b,
                previous_eval_state_dict=previous_eval_b,
                opponent_network=network_a,
                checkpoint_path=checkpoint_path,
                best_state_dict=best_state_b,
            )
            head_to_head_wr = evaluate_head_to_head(network_a, network_b, config, games=config.ppo.eval_games)
            metrics.head_to_head_win_rate_a.append({"episode": float(episode), "agent_a_win_rate": head_to_head_wr})
            if checkpoint_path is not None:
                (checkpoint_path / "dual_metrics_latest.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    network_a.load_state_dict(best_state_a)
    network_b.load_state_dict(best_state_b)
    if checkpoint_path is not None:
        (checkpoint_path / "dual_metrics_final.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return (
        PPOAgent(network_a, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed, name="agent_a"),
        PPOAgent(network_b, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed + 1, name="agent_b"),
    ), metrics


def collect_curriculum_self_play_episode(
    network: ConnectFourActorCritic,
    device: torch.device,
    config: Config,
    opponent_network: ConnectFourActorCritic,
) -> tuple[list[list[dict[str, object]]], float, int]:
    trajectories: dict[int, list[dict[str, object]]] = {1: [], 2: []}
    state = initial_state()
    episode_steps = 0

    while not is_terminal(state):
        player = state.current_player
        current_network = network if player == 1 else opponent_network
        action, log_prob, value, entropy, action_mask = sample_policy_action(current_network, state, player, device)
        next_state = apply_action(state, action)
        reward = compute_step_reward(state, next_state, player, config)
        trajectories[player].append(
            {
                "state": np.asarray(encode_state(state, player), dtype=np.float32),
                "action": action,
                "action_mask": action_mask,
                "log_prob": log_prob,
                "value": value,
                "reward": reward,
                "done": is_terminal(next_state),
                "entropy": entropy,
            }
        )
        if is_terminal(next_state):
            other_player = 2 if player == 1 else 1
            if trajectories[other_player]:
                finalize_last_transition(trajectories[other_player], outcome_for_player(next_state, other_player))
        state = next_state
        episode_steps += 1

    return [trajectories[1], trajectories[2]], outcome_for_player(state, 1), episode_steps


def collect_policy_episode_against_opponent(
    network: ConnectFourActorCritic,
    device: torch.device,
    opponent_agent,
    controlled_player: int,
    config: Config,
) -> tuple[list[dict[str, object]], float, int]:
    trajectory: list[dict[str, object]] = []
    state = initial_state()
    episode_reward = 0.0
    episode_steps = 0

    while not is_terminal(state):
        if state.current_player == controlled_player:
            action, log_prob, value, entropy, action_mask = sample_policy_action(network, state, controlled_player, device)
            next_state = apply_action(state, action)
            reward = compute_step_reward(state, next_state, controlled_player, config)
            trajectory.append(
                {
                    "state": np.asarray(encode_state(state, controlled_player), dtype=np.float32),
                    "action": action,
                    "action_mask": action_mask,
                    "log_prob": log_prob,
                    "value": value,
                    "reward": reward,
                    "done": is_terminal(next_state),
                    "entropy": entropy,
                }
            )
            episode_reward = reward if is_terminal(next_state) else episode_reward
            state = next_state
        else:
            action = opponent_agent.select_action(state, legal_actions(state))
            state = apply_action(state, action)
            if is_terminal(state) and trajectory:
                finalize_last_transition(trajectory, outcome_for_player(state, controlled_player))
                episode_reward = outcome_for_player(state, controlled_player)
        episode_steps += 1

    return trajectory, episode_reward, episode_steps


def collect_dual_policy_episode(
    network_a: ConnectFourActorCritic,
    network_b: ConnectFourActorCritic,
    device: torch.device,
    config: Config,
    *,
    agent_a_player: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float, float, int]:
    trajectory_a: list[dict[str, object]] = []
    trajectory_b: list[dict[str, object]] = []
    state = initial_state()
    episode_steps = 0

    while not is_terminal(state):
        current_player = state.current_player
        is_agent_a_turn = current_player == agent_a_player
        current_network = network_a if is_agent_a_turn else network_b
        action, log_prob, value, entropy, action_mask = sample_policy_action(current_network, state, current_player, device)
        next_state = apply_action(state, action)
        reward = compute_step_reward(state, next_state, current_player, config)
        transition = {
            "state": np.asarray(encode_state(state, current_player), dtype=np.float32),
            "action": action,
            "action_mask": action_mask,
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
            "done": is_terminal(next_state),
            "entropy": entropy,
        }
        if is_agent_a_turn:
            trajectory_a.append(transition)
        else:
            trajectory_b.append(transition)

        if is_terminal(next_state):
            if trajectory_a:
                finalize_last_transition(trajectory_a, outcome_for_player(next_state, agent_a_player))
            agent_b_player = 2 if agent_a_player == 1 else 1
            if trajectory_b:
                finalize_last_transition(trajectory_b, outcome_for_player(next_state, agent_b_player))
        state = next_state
        episode_steps += 1

    reward_a = outcome_for_player(state, agent_a_player)
    reward_b = outcome_for_player(state, 2 if agent_a_player == 1 else 1)
    return trajectory_a, trajectory_b, reward_a, reward_b, episode_steps


def evaluate_head_to_head(
    network_a: ConnectFourActorCritic,
    network_b: ConnectFourActorCritic,
    config: Config,
    *,
    games: int,
) -> float:
    agent_a = PPOAgent(network_a, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed, name="agent_a_eval")
    agent_b = PPOAgent(network_b, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed + 1, name="agent_b_eval")
    wins_a = 0

    for game_idx in range(games):
        agent_a_player = 1 if game_idx % 2 == 0 else 2
        state = initial_state()
        while not is_terminal(state):
            if state.current_player == agent_a_player:
                action = agent_a.select_action(state, legal_actions(state))
            else:
                action = agent_b.select_action(state, legal_actions(state))
            state = apply_action(state, action)
        if state.winner == agent_a_player:
            wins_a += 1
    return wins_a / games


def augment_trajectory(trajectory: list[dict[str, object]]) -> list[dict[str, object]]:
    augmented: list[dict[str, object]] = []
    for step in trajectory:
        augmented.append(step)
        flipped = dict(step)
        flipped["state"] = np.flip(step["state"], axis=2).copy()
        flipped["action"] = 6 - int(step["action"])
        flipped["action_mask"] = np.flip(step["action_mask"], axis=0).copy()
        augmented.append(flipped)
    return augmented


def _run_single_agent_evaluation(
    *,
    network: ConnectFourActorCritic,
    config: Config,
    metrics: PPOCurriculumMetrics,
    previous_eval_state_dict: dict[str, torch.Tensor] | None,
    checkpoint_path: Path | None,
    best_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    eval_agent = PPOAgent(network, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed)
    random_wr = evaluate_against_agent(
        eval_agent,
        lambda game_idx: RandomAgent(seed=config.global_.seed + 10_000 + game_idx),
        games=config.ppo.eval_games,
    )
    heuristic_wr = evaluate_against_agent(
        eval_agent,
        lambda game_idx: HeuristicAgent(seed=config.global_.seed + 20_000 + game_idx),
        games=config.ppo.eval_games,
    )
    previous_wr = 0.0
    if previous_eval_state_dict is not None:
        previous_wr = evaluate_against_agent(
            eval_agent,
            lambda game_idx: PPOAgent(
                _load_previous_ppo_network(previous_eval_state_dict, config.ppo.hidden_dim),
                device=config.resolve_device(),
                sample_actions=False,
                seed=config.global_.seed + 30_000 + game_idx,
                name="previous_snapshot",
            ),
            games=config.ppo.eval_games,
        )
    metrics.evaluation.append(
        {
            "episode": float(len(metrics.episode_rewards)),
            "vs_random_win_rate": random_wr,
            "vs_heuristic_win_rate": heuristic_wr,
            "vs_previous_win_rate": previous_wr,
        }
    )
    score = heuristic_wr * config.ppo.checkpoint_score_heuristic_weight + random_wr
    if score >= metrics.best_score:
        metrics.best_score = score
        best_state_dict = clone_state_dict(network)
        if checkpoint_path is not None:
            best_path = checkpoint_path / "ppo_best.pt"
            torch.save(best_state_dict, best_path)
            metrics.best_checkpoint_path = str(best_path)
    if checkpoint_path is not None:
        write_metrics_snapshot(metrics, checkpoint_path / "metrics_latest.json")
    return best_state_dict


def _run_dual_agent_evaluation(
    *,
    label: str,
    network: ConnectFourActorCritic,
    config: Config,
    metrics: DualAgentMetrics,
    previous_eval_state_dict: dict[str, torch.Tensor] | None,
    opponent_network: ConnectFourActorCritic,
    checkpoint_path: Path | None,
    best_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    eval_agent = PPOAgent(network, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed, name=f"{label}_eval")
    random_wr = evaluate_against_agent(
        eval_agent,
        lambda game_idx: RandomAgent(seed=config.global_.seed + 10_000 + game_idx),
        games=config.ppo.eval_games,
    )
    heuristic_wr = evaluate_against_agent(
        eval_agent,
        lambda game_idx: HeuristicAgent(seed=config.global_.seed + 20_000 + game_idx),
        games=config.ppo.eval_games,
    )
    other_wr = evaluate_against_agent(
        eval_agent,
        lambda _game_idx: PPOAgent(
            opponent_network,
            device=config.resolve_device(),
            sample_actions=False,
            seed=config.global_.seed + 40_000,
            name="peer_snapshot",
        ),
        games=config.ppo.eval_games,
    )
    previous_wr = 0.0
    if previous_eval_state_dict is not None:
        previous_wr = evaluate_against_agent(
            eval_agent,
            lambda game_idx: PPOAgent(
                _load_previous_ppo_network(previous_eval_state_dict, config.ppo.hidden_dim),
                device=config.resolve_device(),
                sample_actions=False,
                seed=config.global_.seed + 30_000 + game_idx,
                name=f"{label}_previous",
            ),
            games=config.ppo.eval_games,
        )

    metrics.evaluation.append(
        {
            "episode": float(len(metrics.episode_rewards)),
            "vs_random_win_rate": random_wr,
            "vs_heuristic_win_rate": heuristic_wr,
            "vs_previous_win_rate": previous_wr,
            "vs_peer_win_rate": other_wr,
        }
    )
    score = heuristic_wr * config.ppo.checkpoint_score_heuristic_weight + random_wr + 0.5 * other_wr
    if score >= metrics.best_score:
        metrics.best_score = score
        best_state_dict = clone_state_dict(network)
        if checkpoint_path is not None:
            best_path = checkpoint_path / f"{label}_best.pt"
            torch.save(best_state_dict, best_path)
            metrics.best_checkpoint_path = str(best_path)
    return best_state_dict, clone_state_dict(network)
