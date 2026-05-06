from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from connect4_rl.agents.baselines import HeuristicAgent, RandomAgent
from connect4_rl.agents.learning.alphazero import (
    AlphaZeroAgent,
    AlphaZeroConfig,
    ConnectFourPolicyValueNet,
    clone_state_dict,
    encode_alphazero_state,
    run_policy_value_mcts,
    sample_action_from_policy,
)
from connect4_rl.envs.connect_four import apply_action, initial_state, is_terminal, legal_actions, outcome_for_player

from .alphazero_training import (
    AlphaZeroMetrics,
    _load_previous_alphazero_network,
    evaluate_against_agent,
    evaluate_tactical_accuracy,
    generate_self_play_episode,
    get_training_mcts_simulations,
    maybe_anneal_learning_rate,
    update_policy_value_network,
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
class AlphaZeroCurriculumMetrics(AlphaZeroMetrics):
    curriculum_name: str = ""
    curriculum_description: str = ""
    phase_sequence: list[str] = field(default_factory=list)
    phase_summary: list[dict[str, object]] = field(default_factory=list)
    opponent_kinds: list[str] = field(default_factory=list)


def build_default_alphazero_curricula() -> dict[str, CurriculumDefinition]:
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
            description="Introduz self-play cedo, regressa ao heuristico no meio e fecha novamente em self-play.",
            phases=(
                CurriculumPhase("warmup_random", "random", 0.25),
                CurriculumPhase("early_self_play", "self_play", 0.25),
                CurriculumPhase("mid_heuristic", "heuristic", 0.25),
                CurriculumPhase("final_self_play", "self_play", 0.25),
            ),
        ),
        "curriculum_short_heuristic_late": CurriculumDefinition(
            name="curriculum_short_heuristic_late",
            description="Mantem uma fase heuristica curta e apenas no fim, preservando a maior parte do treino em self-play.",
            phases=(
                CurriculumPhase("warmup_random", "random", 0.25),
                CurriculumPhase("long_self_play", "self_play", 0.60),
                CurriculumPhase("short_late_heuristic", "heuristic", 0.15),
            ),
        ),
        "curriculum_probabilistic_bridge": CurriculumDefinition(
            name="curriculum_probabilistic_bridge",
            description="Faz a transicao para self-play com uma fase mista que injeta heuristic e random de forma rara.",
            phases=(
                CurriculumPhase("warmup_random", "random", 0.25),
                CurriculumPhase(
                    "mixed_bridge",
                    None,
                    0.50,
                    opponent_weights={"self_play": 0.70, "heuristic": 0.20, "random": 0.10},
                ),
                CurriculumPhase("self_play_finish", "self_play", 0.25),
            ),
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


def train_alphazero_with_curriculum(
    definition: CurriculumDefinition,
    config: AlphaZeroConfig | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
) -> tuple[AlphaZeroAgent, AlphaZeroCurriculumMetrics]:
    config = config or AlphaZeroConfig()
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    network = ConnectFourPolicyValueNet(
        n_filters=config.n_filters,
        n_res_blocks=config.n_res_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    replay_buffer: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=config.replay_capacity)

    metrics = AlphaZeroCurriculumMetrics(
        config=asdict(config),
        curriculum_name=definition.name,
        curriculum_description=definition.description,
    )
    best_state_dict = clone_state_dict(network)
    previous_eval_state_dict: dict[str, torch.Tensor] | None = None

    schedule, phase_summary = expand_curriculum_schedule(config.episodes, definition, seed=config.seed)
    metrics.phase_sequence = schedule
    metrics.phase_summary = phase_summary

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    eval_simulations = config.eval_mcts_simulations or config.mcts_simulations

    for episode in range(1, config.episodes + 1):
        opponent_kind = schedule[episode - 1]
        training_simulations = get_training_mcts_simulations(config, episode)
        examples, final_reward, episode_steps = generate_curriculum_episode(
            network,
            config,
            rng,
            opponent_kind,
            episode,
            simulations=training_simulations,
        )
        replay_buffer.extend(examples)

        policy_loss = 0.0
        value_loss = 0.0
        if episode >= config.replay_warmup_games and len(replay_buffer) >= min(config.batch_size, 8):
            maybe_anneal_learning_rate(optimizer, config, episode)
            batch_policy_losses: list[float] = []
            batch_value_losses: list[float] = []
            for _ in range(config.updates_per_episode):
                step_policy_loss, step_value_loss = update_policy_value_network(network, optimizer, replay_buffer, config, device, rng)
                batch_policy_losses.append(step_policy_loss)
                batch_value_losses.append(step_value_loss)
            policy_loss = float(np.mean(batch_policy_losses))
            value_loss = float(np.mean(batch_value_losses))
            metrics.policy_losses.append(policy_loss)
            metrics.value_losses.append(value_loss)

        metrics.episode_rewards.append(final_reward)
        metrics.episode_lengths.append(episode_steps)
        metrics.opponent_kinds.append(opponent_kind)

        if checkpoint_path is not None and (episode % config.eval_interval == 0 or episode == config.episodes):
            torch.save(network.state_dict(), checkpoint_path / f"alphazero_episode_{episode:04d}.pt")

        if episode % config.eval_interval == 0 or episode == config.episodes:
            eval_agent = AlphaZeroAgent(
                network,
                simulations=eval_simulations,
                c_puct=config.c_puct,
                device=config.device,
                seed=config.seed,
                temperature=0.0,
            )
            random_wr = evaluate_against_agent(
                eval_agent,
                lambda game_idx: RandomAgent(seed=config.seed + 10_000 + game_idx),
                games=config.eval_games,
            )
            heuristic_wr = evaluate_against_agent(
                eval_agent,
                lambda game_idx: HeuristicAgent(seed=config.seed + 20_000 + game_idx),
                games=config.eval_games,
            )
            previous_wr = 0.0
            if previous_eval_state_dict is not None:
                previous_wr = evaluate_against_agent(
                    eval_agent,
                    lambda game_idx: AlphaZeroAgent(
                        _load_previous_alphazero_network(
                            previous_eval_state_dict,
                            n_filters=config.n_filters,
                            n_res_blocks=config.n_res_blocks,
                        ),
                        simulations=eval_simulations,
                        c_puct=config.c_puct,
                        device=config.device,
                        seed=config.seed + 30_000 + game_idx,
                        temperature=0.0,
                        name="previous_snapshot",
                    ),
                    games=config.eval_games,
                )
            metrics.evaluation.append(
                {
                    "episode": float(episode),
                    "vs_random_win_rate": random_wr,
                    "vs_heuristic_win_rate": heuristic_wr,
                    "vs_previous_win_rate": previous_wr,
                }
            )
            metrics.tactical_accuracy.append(
                {
                    "episode": float(episode),
                    "accuracy": evaluate_tactical_accuracy(
                        network,
                        config,
                        examples_per_type=config.tactical_eval_examples,
                    ),
                }
            )
            score = heuristic_wr * config.checkpoint_score_heuristic_weight + random_wr
            if score >= metrics.best_score:
                metrics.best_score = score
                best_state_dict = clone_state_dict(network)
                if checkpoint_path is not None:
                    best_path = checkpoint_path / "alphazero_best.pt"
                    torch.save(best_state_dict, best_path)
                    metrics.best_checkpoint_path = str(best_path)
            if checkpoint_path is not None:
                write_metrics_snapshot(metrics, checkpoint_path / "metrics_latest.json")
            previous_eval_state_dict = clone_state_dict(network)

    network.load_state_dict(best_state_dict)
    final_agent = AlphaZeroAgent(
        network,
        simulations=eval_simulations,
        c_puct=config.c_puct,
        device=config.device,
        seed=config.seed,
        temperature=0.0,
    )
    if checkpoint_path is not None:
        write_metrics_snapshot(metrics, checkpoint_path / "metrics_final.json")
    return final_agent, metrics


def generate_curriculum_episode(
    network: ConnectFourPolicyValueNet,
    config: AlphaZeroConfig,
    rng: random.Random,
    opponent_kind: str,
    episode: int,
    *,
    simulations: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, int]:
    if opponent_kind == "self_play":
        return generate_self_play_episode(network, config, rng, simulations=simulations)
    if opponent_kind not in {"random", "heuristic"}:
        raise ValueError(f"Unsupported opponent kind '{opponent_kind}'")

    controlled_player = 1 if episode % 2 == 1 else 2
    if opponent_kind == "random":
        opponent = RandomAgent(seed=config.seed + 50_000 + episode, name="curriculum_random")
    else:
        opponent = HeuristicAgent(seed=config.seed + 60_000 + episode, name="curriculum_heuristic")

    examples: list[tuple[np.ndarray, np.ndarray, int]] = []
    state = initial_state()
    episode_steps = 0
    agent_turns = 0

    while not is_terminal(state):
        if state.current_player == controlled_player:
            root_noise = config.root_noise_each_move or agent_turns == 0
            visit_policy = run_policy_value_mcts(
                network,
                state,
                simulations=simulations or config.mcts_simulations,
                c_puct=config.c_puct,
                device=config.device,
                root_dirichlet_alpha=config.dirichlet_alpha if root_noise else None,
                root_dirichlet_epsilon=config.dirichlet_epsilon if root_noise else 0.0,
                rng=rng,
            )
            temperature = config.temperature if agent_turns < config.temperature_drop_move else 0.0
            action = sample_action_from_policy(visit_policy, legal_actions(state), temperature=temperature, rng=rng)
            examples.append((encode_alphazero_state(state, state.current_player), visit_policy.copy(), controlled_player))
            agent_turns += 1
        else:
            action = opponent.select_action(state, legal_actions(state))
        state = apply_action(state, action)
        episode_steps += 1

    final_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    final_value = outcome_for_player(state, controlled_player)
    for encoded_state, policy_target, player in examples:
        value = outcome_for_player(state, player)
        final_examples.append((encoded_state, policy_target, value))
        if config.use_horizontal_symmetry_augmentation:
            final_examples.append((np.flip(encoded_state, axis=2).copy(), np.flip(policy_target, axis=0).copy(), value))

    return final_examples, final_value, episode_steps
