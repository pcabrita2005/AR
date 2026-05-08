from __future__ import annotations

import copy
import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from connect4_rl.agents.baselines import MinimaxAgent, RandomAgent, StrongHeuristicAgent, WeakHeuristicAgent
from connect4_rl.agents.baselines.heuristic_agent import score_position
from connect4_rl.agents.learning.ppo import ConnectFourActorCritic, PPOAgent
from connect4_rl.config import Config
from connect4_rl.envs.connect_four import apply_action, encode_state, initial_state, is_terminal, legal_actions, outcome_for_player
from connect4_rl.experiments.dqn_curriculum_utils import count_winnable_windows
from connect4_rl.utils.seed_utils import set_all_seeds


@dataclass(frozen=True)
class PPOLessonConfig:
    name: str
    opponent_kind: str | None
    eval_opponent: str
    max_train_episodes: int
    opponent_weights: dict[str, float] | None = None
    learning_rate_scale: float = 1.0
    entropy_coeff_scale: float = 1.0
    imitation_coeff_scale: float = 0.0


@dataclass
class PPOMetrics:
    config: dict[str, object]
    bootstrap_summary: dict[str, object] = field(default_factory=dict)
    lesson_summaries: list[dict[str, object]] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    evaluation: list[dict[str, float | str]] = field(default_factory=list)
    opponent_kinds: list[str] = field(default_factory=list)
    curriculum_name: str = "ppo_tutorial_pipeline"
    curriculum_description: str = "Tutorial-style PPO pipeline with staged opponents and conservative self-play finetuning."
    phase_summary: list[dict[str, object]] = field(default_factory=list)
    best_checkpoint_path: str = ""
    best_score: float = float("-inf")
    best_vs_strong_checkpoint_path: str = ""
    best_vs_strong_win_rate: float = float("-inf")
    best_vs_strong_draw_rate: float = 0.0


def train_ppo_self_play(
    config: Config | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
) -> tuple[PPOAgent, PPOMetrics]:
    from connect4_rl.config import get_default_config

    config = copy.deepcopy(config or get_default_config())
    lessons = build_tutorial_ppo_lessons(config.ppo.episodes, profile=config.ppo.curriculum_profile)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    aggregate = PPOMetrics(config={**asdict(config.ppo), "seed": config.global_.seed})
    device = torch.device(config.resolve_device())
    initial_state_dict: dict[str, torch.Tensor] | None = None
    final_agent: PPOAgent | None = None
    episode_offset = 0
    global_best_state_dict: dict[str, torch.Tensor] | None = None
    global_best_vs_strong_state_dict: dict[str, torch.Tensor] | None = None

    if config.ppo.enable_policy_bootstrap and config.ppo.bootstrap_samples > 0:
        initial_state_dict, bootstrap_summary = bootstrap_policy_from_heuristics(config, device=device)
        bootstrap_eval = evaluate_state_dict_against_reference_opponents(
            initial_state_dict,
            config,
            games=config.ppo.eval_games,
        )
        aggregate.bootstrap_summary = {**bootstrap_summary, "bootstrap_eval": bootstrap_eval}
        if checkpoint_root is not None:
            bootstrap_path = checkpoint_root / "ppo_bootstrap.pt"
            torch.save(initial_state_dict, bootstrap_path)
            aggregate.bootstrap_summary["bootstrap_checkpoint_path"] = str(bootstrap_path)
        bootstrap_score = score_evaluation_for_checkpoint(
            {
                "eval_opponent": "strong",
                "eval_mean_outcome": float(bootstrap_eval["vs_strong"]["mean_outcome"]),
                "vs_random_win_rate": float(bootstrap_eval["vs_random"]["win_rate"]),
                "vs_random_draw_rate": float(bootstrap_eval["vs_random"]["draw_rate"]),
                "vs_weak_heuristic_win_rate": float(bootstrap_eval["vs_weak"]["win_rate"]),
                "vs_weak_draw_rate": float(bootstrap_eval["vs_weak"]["draw_rate"]),
                "vs_minimax_1_win_rate": float(bootstrap_eval["vs_minimax_1"]["win_rate"]),
                "vs_minimax_1_draw_rate": float(bootstrap_eval["vs_minimax_1"]["draw_rate"]),
                "vs_minimax_2_win_rate": float(bootstrap_eval["vs_minimax_2"]["win_rate"]),
                "vs_minimax_2_draw_rate": float(bootstrap_eval["vs_minimax_2"]["draw_rate"]),
                "vs_strong_heuristic_win_rate": float(bootstrap_eval["vs_strong"]["win_rate"]),
                "vs_strong_draw_rate": float(bootstrap_eval["vs_strong"]["draw_rate"]),
            },
            config,
        )
        aggregate.best_score = bootstrap_score
        aggregate.best_vs_strong_win_rate = float(bootstrap_eval["vs_strong"]["win_rate"])
        aggregate.best_vs_strong_draw_rate = float(bootstrap_eval["vs_strong"]["draw_rate"])
        if checkpoint_root is not None:
            aggregate.best_checkpoint_path = str(checkpoint_root / "ppo_bootstrap.pt")
            aggregate.best_vs_strong_checkpoint_path = str(checkpoint_root / "ppo_bootstrap.pt")
        global_best_state_dict = clone_state_dict_from_state_dict(initial_state_dict)
        global_best_vs_strong_state_dict = clone_state_dict_from_state_dict(initial_state_dict)

    for lesson_index, lesson in enumerate(lessons, start=1):
        lesson_checkpoint_dir = checkpoint_root / lesson.name if checkpoint_root is not None else None
        final_agent, lesson_metrics, lesson_state_dict = train_ppo_lesson(
            lesson=lesson,
            lesson_index=lesson_index,
            config=config,
            checkpoint_dir=lesson_checkpoint_dir,
            initial_state_dict=initial_state_dict,
        )
        initial_state_dict = lesson_state_dict
        if global_best_state_dict is None or lesson_metrics.best_score >= aggregate.best_score:
            global_best_state_dict = {key: value.detach().cpu().clone() for key, value in lesson_state_dict.items()}
        if lesson_metrics.best_vs_strong_win_rate >= aggregate.best_vs_strong_win_rate:
            aggregate.best_vs_strong_win_rate = lesson_metrics.best_vs_strong_win_rate
            aggregate.best_vs_strong_draw_rate = lesson_metrics.best_vs_strong_draw_rate
            aggregate.best_vs_strong_checkpoint_path = lesson_metrics.best_vs_strong_checkpoint_path
            if lesson_metrics.best_vs_strong_checkpoint_path:
                global_best_vs_strong_state_dict = torch.load(
                    lesson_metrics.best_vs_strong_checkpoint_path,
                    map_location=device,
                )

        aggregate.episode_rewards.extend(lesson_metrics.episode_rewards)
        aggregate.episode_lengths.extend(lesson_metrics.episode_lengths)
        aggregate.policy_losses.extend(lesson_metrics.policy_losses)
        aggregate.value_losses.extend(lesson_metrics.value_losses)
        aggregate.entropies.extend(lesson_metrics.entropies)
        aggregate.opponent_kinds.extend(lesson_metrics.opponent_kinds)

        for phase in lesson_metrics.phase_summary:
            shifted = dict(phase)
            shifted["start_episode"] = int(shifted["start_episode"]) + episode_offset
            shifted["end_episode"] = int(shifted["end_episode"]) + episode_offset
            aggregate.phase_summary.append(shifted)

        for evaluation in lesson_metrics.evaluation:
            shifted = dict(evaluation)
            shifted["episode"] = float(shifted["episode"]) + episode_offset
            aggregate.evaluation.append(shifted)

        aggregate.lesson_summaries.append(
            {
                "lesson_name": lesson.name,
                "opponent": lesson.opponent_kind or "mixed",
                "eval_opponent": lesson.eval_opponent,
                "episodes": lesson.max_train_episodes,
                "best_score": lesson_metrics.best_score,
                "best_checkpoint_path": lesson_metrics.best_checkpoint_path,
                "best_vs_strong_checkpoint_path": lesson_metrics.best_vs_strong_checkpoint_path,
                "best_vs_strong_win_rate": lesson_metrics.best_vs_strong_win_rate,
                "best_vs_strong_draw_rate": lesson_metrics.best_vs_strong_draw_rate,
                "last_eval": lesson_metrics.evaluation[-1] if lesson_metrics.evaluation else {},
            }
        )
        aggregate.best_score = max(aggregate.best_score, lesson_metrics.best_score)
        episode_offset += lesson.max_train_episodes

    if final_agent is None or initial_state_dict is None or global_best_state_dict is None:
        raise RuntimeError("Tutorial PPO pipeline produced no final agent.")

    if checkpoint_root is not None:
        final_path = checkpoint_root / "ppo_best.pt"
        torch.save(global_best_state_dict, final_path)
        aggregate.best_checkpoint_path = str(final_path)
        if global_best_vs_strong_state_dict is not None:
            best_vs_strong_path = checkpoint_root / "ppo_best_vs_strong.pt"
            torch.save(global_best_vs_strong_state_dict, best_vs_strong_path)
            aggregate.best_vs_strong_checkpoint_path = str(best_vs_strong_path)
        write_metrics_snapshot(aggregate, checkpoint_root / "metrics_latest.json")
        write_metrics_snapshot(aggregate, checkpoint_root / "metrics_final.json")

    best_network = build_ppo_network_from_config(config).to(device)
    best_network.load_state_dict(global_best_state_dict)
    return PPOAgent(best_network, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed), aggregate


def train_ppo_lesson(
    *,
    lesson: PPOLessonConfig,
    lesson_index: int,
    config: Config,
    checkpoint_dir: str | Path | None = None,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[PPOAgent, PPOMetrics, dict[str, torch.Tensor]]:
    set_all_seeds(config.global_.seed)
    rng = random.Random(config.global_.seed)
    device = torch.device(config.resolve_device())

    network = build_ppo_network_from_config(config).to(device)
    if initial_state_dict is not None:
        network.load_state_dict(initial_state_dict)
    if initial_state_dict is not None and lesson_index <= config.ppo.freeze_feature_extractor_lessons:
        set_module_requires_grad(network.features, False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in network.parameters() if parameter.requires_grad],
        lr=max(config.ppo.learning_rate * lesson.learning_rate_scale, 1e-6),
    )
    effective_entropy_coeff = config.ppo.entropy_coeff * lesson.entropy_coeff_scale
    effective_imitation_coeff = config.ppo.imitation_loss_coeff * lesson.imitation_coeff_scale

    metrics = PPOMetrics(
        config={**asdict(config.ppo), "seed": config.global_.seed},
        curriculum_name=lesson.name,
        curriculum_description=f"Tutorial PPO lesson against {lesson.opponent_kind or 'mixed'} opponents.",
    )
    schedule = build_lesson_schedule(lesson, seed=config.global_.seed)
    metrics.phase_summary = [
        {
            "phase_name": lesson.name,
            "opponent_kind": lesson.opponent_kind or "mixed",
            "episodes": lesson.max_train_episodes,
            "start_episode": 1,
            "end_episode": lesson.max_train_episodes,
            "realized_opponents": dict(Counter(schedule)),
        }
    ]

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_state_dict = clone_state_dict(network)
    previous_eval_state_dict: dict[str, torch.Tensor] | None = None
    opponent_pool: deque[dict[str, torch.Tensor]] = deque([clone_state_dict(network)], maxlen=config.ppo.opponent_pool_size)
    rollout_buffer: list[dict[str, object]] = []
    no_improve_vs_strong_evals = 0

    for episode, opponent_kind in enumerate(schedule, start=1):
        if opponent_kind == "self_play":
            opponent_state_dict = rng.choice(list(opponent_pool))
            opponent_network = build_ppo_network_from_config(config).to(device)
            opponent_network.load_state_dict(opponent_state_dict)
            opponent_network.eval()
            controlled_player = 1 if rng.random() < 0.5 else 2
            trajectory, episode_reward, episode_steps = collect_self_play_episode(
                network,
                device,
                config,
                opponent_network=opponent_network,
                controlled_player=controlled_player,
            )
            if trajectory:
                rollout_buffer.extend(augment_trajectory(trajectory) if config.ppo.use_horizontal_symmetry_augmentation else trajectory)
        else:
            opponent_agent = build_fixed_opponent(opponent_kind, config.global_.seed + episode)
            controlled_player = 1 if rng.random() < 0.5 else 2
            trajectory, episode_reward, episode_steps = collect_policy_episode_against_opponent(
                network,
                device,
                opponent_agent,
                controlled_player,
                config,
                expert_kind=opponent_kind if opponent_kind in {"weak", "strong"} else None,
            )
            if trajectory:
                rollout_buffer.extend(augment_trajectory(trajectory) if config.ppo.use_horizontal_symmetry_augmentation else trajectory)

        if rollout_buffer and (episode % config.ppo.rollout_episodes_per_update == 0 or episode == lesson.max_train_episodes):
            maybe_anneal_learning_rate(optimizer, config, episode, total_episodes=lesson.max_train_episodes, learning_rate_scale=lesson.learning_rate_scale)
            policy_loss, value_loss, entropy = update_ppo(
                network,
                optimizer,
                rollout_buffer,
                config,
                device,
                entropy_coeff=effective_entropy_coeff,
                imitation_coeff=effective_imitation_coeff,
            )
            metrics.policy_losses.append(policy_loss)
            metrics.value_losses.append(value_loss)
            metrics.entropies.append(entropy)
            opponent_pool.append(clone_state_dict(network))
            rollout_buffer = []

        metrics.episode_rewards.append(episode_reward)
        metrics.episode_lengths.append(episode_steps)
        metrics.opponent_kinds.append(opponent_kind)

        if checkpoint_path is not None and (episode % config.ppo.eval_interval == 0 or episode == lesson.max_train_episodes):
            torch.save(network.state_dict(), checkpoint_path / f"{lesson.name}_episode_{episode:04d}.pt")

        if episode % config.ppo.eval_interval == 0 or episode == lesson.max_train_episodes:
            evaluation = evaluate_ppo_lesson(
                network=network,
                config=config,
                lesson=lesson,
                previous_eval_state_dict=previous_eval_state_dict,
            )
            evaluation["episode"] = float(episode)
            evaluation["lesson_name"] = lesson.name
            metrics.evaluation.append(evaluation)
            score = score_evaluation_for_checkpoint(evaluation, config)
            if score >= metrics.best_score:
                metrics.best_score = score
                best_state_dict = clone_state_dict(network)
                if checkpoint_path is not None:
                    best_path = checkpoint_path / "ppo_best.pt"
                    torch.save(best_state_dict, best_path)
                    metrics.best_checkpoint_path = str(best_path)
            current_vs_strong = float(evaluation.get("vs_strong_heuristic_win_rate", 0.0))
            current_vs_strong_draw = float(evaluation.get("vs_strong_draw_rate", 0.0))
            best_strong_tuple = (float(metrics.best_vs_strong_win_rate), float(metrics.best_vs_strong_draw_rate))
            current_strong_tuple = (current_vs_strong, current_vs_strong_draw)
            if current_strong_tuple > best_strong_tuple:
                metrics.best_vs_strong_win_rate = current_vs_strong
                metrics.best_vs_strong_draw_rate = current_vs_strong_draw
                no_improve_vs_strong_evals = 0
                if checkpoint_path is not None:
                    best_vs_strong_path = checkpoint_path / "ppo_best_vs_strong.pt"
                    torch.save(clone_state_dict(network), best_vs_strong_path)
                    metrics.best_vs_strong_checkpoint_path = str(best_vs_strong_path)
                else:
                    metrics.best_vs_strong_checkpoint_path = metrics.best_checkpoint_path
            else:
                no_improve_vs_strong_evals += 1
            if checkpoint_path is not None:
                write_metrics_snapshot(metrics, checkpoint_path / "metrics_latest.json")
            previous_eval_state_dict = clone_state_dict(network)
            if (
                lesson.name in {"lesson5_self_play", "lesson5_endgame"}
                and episode >= config.ppo.self_play_min_episodes_before_early_stop
                and no_improve_vs_strong_evals >= config.ppo.self_play_early_stop_patience_evals
            ):
                break

    if checkpoint_path is not None and not metrics.best_checkpoint_path:
        fallback_best = checkpoint_path / "ppo_best.pt"
        torch.save(best_state_dict, fallback_best)
        metrics.best_checkpoint_path = str(fallback_best)
    if checkpoint_path is not None and not metrics.best_vs_strong_checkpoint_path:
        metrics.best_vs_strong_checkpoint_path = metrics.best_checkpoint_path
    if metrics.best_vs_strong_win_rate == float("-inf"):
        metrics.best_vs_strong_win_rate = 0.0
        metrics.best_vs_strong_draw_rate = 0.0

    network.load_state_dict(best_state_dict)
    final_agent = PPOAgent(network, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed)
    if checkpoint_path is not None:
        write_metrics_snapshot(metrics, checkpoint_path / "metrics_final.json")
    return final_agent, metrics, clone_state_dict(network)


def build_tutorial_ppo_lessons(total_episodes: int | None = None, *, profile: str = "tutorial") -> list[PPOLessonConfig]:
    if profile == "final_push_hard_bridge":
        lesson_specs = [
            PPOLessonConfig("lesson1_random", "random", "random", 80),
            PPOLessonConfig("lesson2_weak", "weak", "weak", 80, imitation_coeff_scale=0.20),
            PPOLessonConfig(
                "lesson3_strong_core",
                "strong",
                "strong",
                280,
                learning_rate_scale=0.70,
                entropy_coeff_scale=0.22,
                imitation_coeff_scale=1.15,
            ),
            PPOLessonConfig(
                "lesson4_hard_bridge",
                None,
                "minimax_2",
                240,
                opponent_weights={"strong": 0.40, "minimax_1": 0.30, "minimax_2": 0.25, "weak": 0.05},
                learning_rate_scale=0.38,
                entropy_coeff_scale=0.16,
                imitation_coeff_scale=0.70,
            ),
            PPOLessonConfig(
                "lesson5_endgame",
                None,
                "strong",
                40,
                opponent_weights={"strong": 0.45, "minimax_2": 0.30, "self_play": 0.15, "minimax_1": 0.10},
                learning_rate_scale=0.12,
                entropy_coeff_scale=0.07,
                imitation_coeff_scale=0.25,
            ),
        ]
    elif profile == "final_push":
        lesson_specs = [
            PPOLessonConfig("lesson1_random", "random", "random", 80),
            PPOLessonConfig("lesson2_weak", "weak", "weak", 80, imitation_coeff_scale=0.20),
            PPOLessonConfig(
                "lesson3_strong_core",
                "strong",
                "strong",
                320,
                learning_rate_scale=0.70,
                entropy_coeff_scale=0.22,
                imitation_coeff_scale=1.15,
            ),
            PPOLessonConfig(
                "lesson4_bridge",
                None,
                "strong",
                200,
                opponent_weights={"strong": 0.55, "minimax_1": 0.35, "weak": 0.10},
                learning_rate_scale=0.45,
                entropy_coeff_scale=0.18,
                imitation_coeff_scale=0.75,
            ),
            PPOLessonConfig(
                "lesson5_endgame",
                None,
                "strong",
                40,
                opponent_weights={"strong": 0.60, "minimax_1": 0.15, "self_play": 0.15, "weak": 0.10},
                learning_rate_scale=0.14,
                entropy_coeff_scale=0.08,
                imitation_coeff_scale=0.35,
            ),
        ]
    else:
        lesson_specs = [
            PPOLessonConfig("lesson1_random", "random", "random", 120),
            PPOLessonConfig("lesson2_weak", "weak", "weak", 120, imitation_coeff_scale=0.15),
            PPOLessonConfig(
                "lesson3_minimax",
                None,
                "minimax_2",
                160,
                opponent_weights={"minimax_1": 0.65, "minimax_2": 0.35},
                learning_rate_scale=0.75,
                entropy_coeff_scale=0.45,
                imitation_coeff_scale=0.55,
            ),
            PPOLessonConfig("lesson4_strong", "strong", "strong", 260, learning_rate_scale=0.65, entropy_coeff_scale=0.30, imitation_coeff_scale=1.0),
            PPOLessonConfig(
                "lesson5_self_play",
                None,
                "strong",
                60,
                opponent_weights={"strong": 0.45, "minimax_1": 0.15, "minimax_2": 0.20, "self_play": 0.10, "weak": 0.10},
                learning_rate_scale=0.18,
                entropy_coeff_scale=0.12,
                imitation_coeff_scale=0.30,
            ),
        ]
    if total_episodes is None or total_episodes <= 0:
        return lesson_specs

    weights = [lesson.max_train_episodes for lesson in lesson_specs]
    raw_counts = [weight * total_episodes / sum(weights) for weight in weights]
    counts = [int(value) for value in raw_counts]
    remainder = total_episodes - sum(counts)
    order = sorted(range(len(raw_counts)), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    adjusted: list[PPOLessonConfig] = []
    for lesson, count in zip(lesson_specs, counts):
        adjusted.append(
            PPOLessonConfig(
                name=lesson.name,
                opponent_kind=lesson.opponent_kind,
                eval_opponent=lesson.eval_opponent,
                max_train_episodes=count,
                opponent_weights=lesson.opponent_weights,
                learning_rate_scale=lesson.learning_rate_scale,
                entropy_coeff_scale=lesson.entropy_coeff_scale,
                imitation_coeff_scale=lesson.imitation_coeff_scale,
            )
        )
    return adjusted


def build_lesson_schedule(lesson: PPOLessonConfig, *, seed: int) -> list[str]:
    if lesson.max_train_episodes <= 0:
        return []
    if lesson.opponent_weights:
        rng = random.Random(seed)
        kinds = list(lesson.opponent_weights.keys())
        weights = [float(lesson.opponent_weights[kind]) for kind in kinds]
        return rng.choices(kinds, weights=weights, k=lesson.max_train_episodes)
    if lesson.opponent_kind is None:
        raise ValueError("lesson requires either opponent_kind or opponent_weights")
    return [lesson.opponent_kind] * lesson.max_train_episodes


def update_ppo(
    network: ConnectFourActorCritic,
    optimizer: torch.optim.Optimizer,
    trajectory: list[dict[str, object]],
    config: Config,
    device: torch.device,
    *,
    entropy_coeff: float | None = None,
    imitation_coeff: float = 0.0,
) -> tuple[float, float, float]:
    if not trajectory:
        return 0.0, 0.0, 0.0

    states = torch.tensor(np.stack([step["state"] for step in trajectory]), dtype=torch.float32, device=device)
    actions = torch.tensor([step["action"] for step in trajectory], dtype=torch.int64, device=device)
    action_masks = torch.tensor(np.stack([step["action_mask"] for step in trajectory]), dtype=torch.bool, device=device)
    old_log_probs = torch.tensor([step["log_prob"] for step in trajectory], dtype=torch.float32, device=device)
    values = torch.tensor([step["value"] for step in trajectory], dtype=torch.float32, device=device)
    rewards = torch.tensor([step["reward"] for step in trajectory], dtype=torch.float32, device=device)
    dones = torch.tensor([step["done"] for step in trajectory], dtype=torch.float32, device=device)
    expert_actions = torch.tensor([step.get("expert_action", -1) for step in trajectory], dtype=torch.int64, device=device)
    expert_weights = torch.tensor([step.get("expert_weight", 0.0) for step in trajectory], dtype=torch.float32, device=device)

    returns, advantages = compute_gae(rewards, values, dones, config.ppo.gamma, config.ppo.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    effective_entropy_coeff = config.ppo.entropy_coeff if entropy_coeff is None else entropy_coeff
    batch_size = states.shape[0]
    minibatch_size = min(config.ppo.minibatch_size, batch_size)

    network.train()
    stop_early = False
    for _ in range(config.ppo.n_epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            batch_indices = indices[start:end]

            logits, predicted_values = network(states[batch_indices])
            masked_logits = masked_logits_from_logits(logits, action_masks[batch_indices])
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(actions[batch_indices])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
            unclipped = ratio * advantages[batch_indices]
            clipped = torch.clamp(ratio, 1.0 - config.ppo.clip_ratio, 1.0 + config.ppo.clip_ratio) * advantages[batch_indices]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.smooth_l1_loss(predicted_values, returns[batch_indices])
            imitation_loss = torch.tensor(0.0, device=device)
            if imitation_coeff > 0.0:
                batch_expert_actions = expert_actions[batch_indices]
                batch_expert_weights = expert_weights[batch_indices]
                valid = batch_expert_actions >= 0
                if torch.any(valid):
                    per_item = torch.nn.functional.cross_entropy(
                        masked_logits[valid],
                        batch_expert_actions[valid],
                        reduction="none",
                    )
                    weighted = per_item * batch_expert_weights[valid]
                    imitation_loss = weighted.sum() / batch_expert_weights[valid].sum().clamp_min(1e-6)
            loss = policy_loss + config.ppo.value_coeff * value_loss - effective_entropy_coeff * entropy + imitation_coeff * imitation_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.ppo.max_grad_norm)
            optimizer.step()

            approx_kl = (old_log_probs[batch_indices] - new_log_probs).mean().abs()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            if float(approx_kl.item()) > config.ppo.target_kl:
                stop_early = True
                break
        if stop_early:
            break

    network.eval()
    return (
        float(np.mean(policy_losses)) if policy_losses else 0.0,
        float(np.mean(value_losses)) if value_losses else 0.0,
        float(np.mean(entropies)) if entropies else 0.0,
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, device=rewards.device)
    next_value = torch.tensor(0.0, device=rewards.device)

    for step in reversed(range(len(rewards))):
        mask = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
        next_value = values[step]

    returns = advantages + values
    return returns.detach(), advantages.detach()


def augment_trajectory(trajectory: list[dict[str, object]]) -> list[dict[str, object]]:
    augmented = []
    for step in trajectory:
        augmented.append(step)
        flipped = dict(step)
        flipped["state"] = np.flip(step["state"], axis=2).copy()
        flipped["action"] = 6 - int(step["action"])
        flipped["action_mask"] = np.flip(step["action_mask"], axis=0).copy()
        augmented.append(flipped)
    return augmented


def legal_actions_to_mask(actions: list[int]) -> np.ndarray:
    mask = np.zeros(7, dtype=np.bool_)
    mask[actions] = True
    return mask


def finalize_last_transition(trajectory: list[dict[str, object]], terminal_reward: float) -> None:
    trajectory[-1]["reward"] = terminal_reward
    trajectory[-1]["done"] = True


def maybe_anneal_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: Config,
    episode: int,
    *,
    total_episodes: int | None = None,
    learning_rate_scale: float = 1.0,
) -> None:
    if not config.ppo.anneal_learning_rate:
        return
    episode_budget = max(total_episodes or config.ppo.episodes, 1)
    progress = max(0.0, 1.0 - ((episode - 1) / episode_budget))
    current_lr = max(config.ppo.learning_rate * learning_rate_scale * progress, 1e-6)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr


def collect_self_play_episode(
    network: ConnectFourActorCritic,
    device: torch.device,
    config: Config,
    *,
    opponent_network: ConnectFourActorCritic,
    controlled_player: int,
) -> tuple[list[dict[str, object]], float, int]:
    trajectory: list[dict[str, object]] = []
    state = initial_state()
    episode_steps = 0

    while not is_terminal(state):
        player = state.current_player
        current_network = network if player == controlled_player else opponent_network
        action, log_prob, value, entropy, action_mask = sample_policy_action(current_network, state, player, device)
        next_state = apply_action(state, action)
        if player == controlled_player:
            reward = compute_step_reward(state, next_state, player, config)
            trajectory.append(
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
        if is_terminal(next_state) and trajectory:
            finalize_last_transition(trajectory, outcome_for_player(next_state, controlled_player))
        state = next_state
        episode_steps += 1

    return trajectory, outcome_for_player(state, controlled_player), episode_steps


def collect_policy_episode_against_opponent(
    network: ConnectFourActorCritic,
    device: torch.device,
    opponent_agent,
    controlled_player: int,
    config: Config,
    *,
    expert_kind: str | None = None,
) -> tuple[list[dict[str, object]], float, int]:
    trajectory: list[dict[str, object]] = []
    state = initial_state()
    episode_reward = 0.0
    episode_steps = 0
    expert_agent = build_fixed_opponent(expert_kind, 123456) if expert_kind in {"weak", "strong"} else None
    expert_weight = 1.0 if expert_kind == "strong" else 0.35 if expert_kind == "weak" else 0.0

    while not is_terminal(state):
        if state.current_player == controlled_player:
            action, log_prob, value, entropy, action_mask = sample_policy_action(
                network,
                state,
                controlled_player,
                device,
            )
            next_state = apply_action(state, action)
            reward = compute_step_reward(state, next_state, controlled_player, config)
            expert_action = expert_agent.select_action(state, legal_actions(state)) if expert_agent is not None else -1
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
                    "expert_action": expert_action,
                    "expert_weight": expert_weight,
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


def sample_policy_action(
    network: ConnectFourActorCritic,
    state,
    player: int,
    device: torch.device,
) -> tuple[int, float, float, float, np.ndarray]:
    state_tensor = torch.tensor(
        encode_state(state, player),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    legal = legal_actions(state)
    with torch.no_grad():
        logits, value = network(state_tensor)
        logits = logits.squeeze(0)
        value = value.squeeze(0)
        masked_logits = masked_logits_from_logits(logits, torch.tensor(legal_actions_to_mask(legal), dtype=torch.bool, device=device))
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.tensor(action, device=device)).item())
        entropy = float(dist.entropy().item())
    return action, log_prob, float(value.item()), entropy, legal_actions_to_mask(legal)


def compute_step_reward(
    state,
    next_state,
    player: int,
    config: Config,
) -> float:
    if is_terminal(next_state):
        return outcome_for_player(next_state, player)
    if not config.ppo.reward_shaping:
        return 0.0
    opponent = 2 if player == 1 else 1
    previous_score = score_position(state, player)
    next_score = score_position(next_state, player)
    score_delta = float(config.ppo.reward_shaping_scale * np.tanh((next_score - previous_score) / 100.0))

    own_threat_delta = count_winnable_windows(next_state, player) - count_winnable_windows(state, player)
    opponent_threat_delta = count_winnable_windows(next_state, opponent) - count_winnable_windows(state, opponent)
    tactical_delta = (
        config.ppo.threat_bonus_scale * own_threat_delta
        - config.ppo.opponent_threat_penalty_scale * opponent_threat_delta
    )

    previous_opp_immediate_wins = count_immediate_winning_actions(state, opponent)
    next_opp_immediate_wins = count_immediate_winning_actions(next_state, opponent)
    blocked_threats = max(previous_opp_immediate_wins - next_opp_immediate_wins, 0)
    allowed_threats = max(next_opp_immediate_wins - previous_opp_immediate_wins, 0)
    threat_management_delta = (
        config.ppo.blocked_threat_bonus_scale * blocked_threats
        - config.ppo.allowed_threat_penalty_scale * allowed_threats
    )

    center_before = center_control_score(state, player)
    center_after = center_control_score(next_state, player)
    center_delta = config.ppo.center_control_scale * (center_after - center_before)
    return float(score_delta + tactical_delta + threat_management_delta + center_delta)


def build_training_mode(
    config: Config,
    episode: int,
    rng: random.Random,
) -> str:
    lessons = build_tutorial_ppo_lessons(config.ppo.episodes, profile=config.ppo.curriculum_profile)
    lesson_ranges: list[tuple[int, PPOLessonConfig]] = []
    cursor = 1
    for lesson in lessons:
        lesson_ranges.append((cursor, lesson))
        cursor += lesson.max_train_episodes
    for start_episode, lesson in lesson_ranges:
        end_episode = start_episode + lesson.max_train_episodes - 1
        if start_episode <= episode <= end_episode:
            if lesson.opponent_weights:
                kinds = list(lesson.opponent_weights.keys())
                weights = [float(lesson.opponent_weights[k]) for k in kinds]
                return rng.choices(kinds, weights=weights, k=1)[0]
            return lesson.opponent_kind or "self_play"
    return "self_play"


def build_fixed_opponent(kind: str, seed: int):
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "weak":
        return WeakHeuristicAgent(seed=seed)
    if kind in {"heuristic", "strong"}:
        return StrongHeuristicAgent(seed=seed)
    if kind == "minimax_1":
        return MinimaxAgent(depth=1, seed=seed)
    if kind == "minimax_2":
        return MinimaxAgent(depth=2, seed=seed)
    raise ValueError(f"Unsupported fixed opponent kind: {kind}")


def evaluate_against_agent(agent: PPOAgent, opponent_factory, *, games: int = 20) -> float:
    wins = 0
    draws = 0
    for game_idx in range(games):
        controlled_player = 1 if game_idx % 2 == 0 else 2
        try:
            opponent = opponent_factory(game_idx)
        except TypeError:
            opponent = opponent_factory()
        state = initial_state()
        while not is_terminal(state):
            if state.current_player == controlled_player:
                action = agent.select_action(state, legal_actions(state))
            else:
                action = opponent.select_action(state, legal_actions(state))
            state = apply_action(state, action)
        if state.winner == controlled_player:
            wins += 1
        elif state.winner == 0:
            draws += 1
    return wins / games


def evaluate_match_summary(agent: PPOAgent, opponent_factory, *, games: int = 20) -> dict[str, float]:
    wins = 0
    draws = 0
    for game_idx in range(games):
        controlled_player = 1 if game_idx % 2 == 0 else 2
        try:
            opponent = opponent_factory(game_idx)
        except TypeError:
            opponent = opponent_factory()
        state = initial_state()
        while not is_terminal(state):
            if state.current_player == controlled_player:
                action = agent.select_action(state, legal_actions(state))
            else:
                action = opponent.select_action(state, legal_actions(state))
            state = apply_action(state, action)
        if state.winner == controlled_player:
            wins += 1
        elif state.winner == 0:
            draws += 1
    losses = games - wins - draws
    return {
        "win_rate": wins / games,
        "draw_rate": draws / games,
        "loss_rate": losses / games,
        "mean_outcome": (wins - losses) / games,
    }


def evaluate_ppo_lesson(
    *,
    network: ConnectFourActorCritic,
    config: Config,
    lesson: PPOLessonConfig,
    previous_eval_state_dict: dict[str, torch.Tensor] | None,
) -> dict[str, float | str]:
    eval_agent = PPOAgent(network, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed)
    random_summary = evaluate_match_summary(
        eval_agent,
        lambda game_idx: RandomAgent(seed=config.global_.seed + 10_000 + game_idx),
        games=config.ppo.eval_games,
    )
    weak_summary = evaluate_match_summary(
        eval_agent,
        lambda game_idx: WeakHeuristicAgent(seed=config.global_.seed + 20_000 + game_idx),
        games=config.ppo.eval_games,
    )
    minimax1_summary = evaluate_match_summary(
        eval_agent,
        lambda game_idx: MinimaxAgent(depth=1, seed=config.global_.seed + 25_000 + game_idx),
        games=config.ppo.eval_games,
    )
    minimax2_summary = evaluate_match_summary(
        eval_agent,
        lambda game_idx: MinimaxAgent(depth=2, seed=config.global_.seed + 27_000 + game_idx),
        games=config.ppo.eval_games,
    )
    strong_summary = evaluate_match_summary(
        eval_agent,
        lambda game_idx: StrongHeuristicAgent(seed=config.global_.seed + 30_000 + game_idx),
        games=config.ppo.eval_games,
    )
    previous_wr = 0.0
    previous_draw_rate = 0.0
    if previous_eval_state_dict is not None:
        previous_summary = evaluate_match_summary(
            eval_agent,
            lambda game_idx: PPOAgent(
                _load_previous_ppo_network(
                    previous_eval_state_dict,
                    config.ppo.hidden_dim,
                    channel_sizes=config.ppo.channel_sizes,
                    kernel_sizes=config.ppo.kernel_sizes,
                    stride_sizes=config.ppo.stride_sizes,
                    head_hidden_sizes=config.ppo.head_hidden_sizes,
                ),
                device=config.resolve_device(),
                sample_actions=False,
                seed=config.global_.seed + 40_000 + game_idx,
                name="previous_snapshot",
            ),
            games=config.ppo.eval_games,
        )
        previous_wr = float(previous_summary["win_rate"])
        previous_draw_rate = float(previous_summary["draw_rate"])

    random_wr = float(random_summary["win_rate"])
    weak_wr = float(weak_summary["win_rate"])
    minimax1_wr = float(minimax1_summary["win_rate"])
    minimax2_wr = float(minimax2_summary["win_rate"])
    strong_wr = float(strong_summary["win_rate"])
    eval_mean_outcome = (
        float(strong_summary["mean_outcome"]) if lesson.eval_opponent == "strong"
        else float(minimax2_summary["mean_outcome"]) if lesson.eval_opponent == "minimax_2"
        else float(minimax1_summary["mean_outcome"]) if lesson.eval_opponent == "minimax_1"
        else float(weak_summary["mean_outcome"]) if lesson.eval_opponent == "weak"
        else float(random_summary["mean_outcome"])
    )
    return {
        "eval_opponent": lesson.eval_opponent,
        "eval_mean_outcome": eval_mean_outcome,
        "vs_random_win_rate": random_wr,
        "vs_random_draw_rate": float(random_summary["draw_rate"]),
        "vs_weak_heuristic_win_rate": weak_wr,
        "vs_weak_draw_rate": float(weak_summary["draw_rate"]),
        "vs_minimax_1_win_rate": minimax1_wr,
        "vs_minimax_1_draw_rate": float(minimax1_summary["draw_rate"]),
        "vs_minimax_2_win_rate": minimax2_wr,
        "vs_minimax_2_draw_rate": float(minimax2_summary["draw_rate"]),
        "vs_strong_heuristic_win_rate": strong_wr,
        "vs_strong_draw_rate": float(strong_summary["draw_rate"]),
        "vs_heuristic_win_rate": strong_wr,
        "vs_previous_win_rate": previous_wr,
        "vs_previous_draw_rate": previous_draw_rate,
    }


def score_evaluation_for_checkpoint(evaluation: dict[str, float | str], config: Config) -> float:
    eval_opponent = str(evaluation.get("eval_opponent", "random"))
    eval_mean_outcome = float(evaluation.get("eval_mean_outcome", 0.0))
    vs_random = float(evaluation.get("vs_random_win_rate", 0.0))
    vs_random_draw = float(evaluation.get("vs_random_draw_rate", 0.0))
    vs_weak = float(evaluation.get("vs_weak_heuristic_win_rate", 0.0))
    vs_weak_draw = float(evaluation.get("vs_weak_draw_rate", 0.0))
    vs_minimax_1 = float(evaluation.get("vs_minimax_1_win_rate", 0.0))
    vs_minimax_1_draw = float(evaluation.get("vs_minimax_1_draw_rate", 0.0))
    vs_minimax_2 = float(evaluation.get("vs_minimax_2_win_rate", 0.0))
    vs_minimax_2_draw = float(evaluation.get("vs_minimax_2_draw_rate", 0.0))
    vs_strong = float(evaluation.get("vs_strong_heuristic_win_rate", 0.0))
    vs_strong_draw = float(evaluation.get("vs_strong_draw_rate", 0.0))
    if eval_opponent == "strong":
        return (
            (8.0 * vs_strong)
            + (2.5 * vs_strong_draw)
            + (1.5 * eval_mean_outcome)
            + (0.4 * vs_weak)
            + (0.1 * vs_weak_draw)
            + (0.2 * vs_random)
            + (0.05 * vs_random_draw)
        )
    if eval_opponent == "minimax_2":
        return (
            (4.5 * vs_minimax_2)
            + (1.5 * vs_minimax_2_draw)
            + (1.5 * eval_mean_outcome)
            + (1.6 * vs_strong)
            + (0.5 * vs_minimax_1)
            + (0.2 * vs_weak)
            + (0.1 * vs_random)
        )
    if eval_opponent == "minimax_1":
        return (
            (3.0 * vs_minimax_1)
            + (1.0 * vs_minimax_1_draw)
            + (1.2 * eval_mean_outcome)
            + (0.8 * vs_strong)
            + (0.4 * vs_weak)
            + (0.1 * vs_random)
        )
    if eval_opponent == "weak":
        return eval_mean_outcome + (0.4 * vs_random) + (0.2 * vs_random_draw) + (1.5 * vs_weak) + (0.25 * vs_strong)
    return eval_mean_outcome + (0.5 * vs_random) + (0.2 * vs_random_draw) + vs_weak + (config.ppo.checkpoint_score_heuristic_weight * vs_strong)


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def clone_state_dict_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def set_module_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _load_previous_ppo_network(
    state_dict: dict[str, torch.Tensor],
    hidden_dim: int,
    *,
    channel_sizes: list[int] | None = None,
    kernel_sizes: list[int] | None = None,
    stride_sizes: list[int] | None = None,
    head_hidden_sizes: list[int] | None = None,
) -> ConnectFourActorCritic:
    network = ConnectFourActorCritic(
        hidden_dim=hidden_dim,
        channel_sizes=channel_sizes,
        kernel_sizes=kernel_sizes,
        stride_sizes=stride_sizes,
        head_hidden_sizes=head_hidden_sizes,
    )
    network.load_state_dict(state_dict)
    return network


def write_metrics_snapshot(metrics: PPOMetrics, path: Path) -> None:
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def build_ppo_network_from_config(config: Config) -> ConnectFourActorCritic:
    return ConnectFourActorCritic(
        hidden_dim=config.ppo.hidden_dim,
        channel_sizes=config.ppo.channel_sizes,
        kernel_sizes=config.ppo.kernel_sizes,
        stride_sizes=config.ppo.stride_sizes,
        head_hidden_sizes=config.ppo.head_hidden_sizes,
    )


def masked_logits_from_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)


def evaluate_state_dict_against_reference_opponents(
    state_dict: dict[str, torch.Tensor],
    config: Config,
    *,
    games: int,
) -> dict[str, dict[str, float]]:
    network = build_ppo_network_from_config(config)
    network.load_state_dict(state_dict)
    agent = PPOAgent(network, device=config.resolve_device(), sample_actions=False, seed=config.global_.seed)
    return {
        "vs_random": evaluate_match_summary(
            agent,
            lambda game_idx: RandomAgent(seed=config.global_.seed + 10_000 + game_idx),
            games=games,
        ),
        "vs_weak": evaluate_match_summary(
            agent,
            lambda game_idx: WeakHeuristicAgent(seed=config.global_.seed + 20_000 + game_idx),
            games=games,
        ),
        "vs_minimax_1": evaluate_match_summary(
            agent,
            lambda game_idx: MinimaxAgent(depth=1, seed=config.global_.seed + 25_000 + game_idx),
            games=games,
        ),
        "vs_minimax_2": evaluate_match_summary(
            agent,
            lambda game_idx: MinimaxAgent(depth=2, seed=config.global_.seed + 27_000 + game_idx),
            games=games,
        ),
        "vs_strong": evaluate_match_summary(
            agent,
            lambda game_idx: StrongHeuristicAgent(seed=config.global_.seed + 30_000 + game_idx),
            games=games,
        ),
    }


def bootstrap_policy_from_heuristics(
    config: Config,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    set_all_seeds(config.global_.seed)
    rng = random.Random(config.global_.seed)
    network = build_ppo_network_from_config(config).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.ppo.bootstrap_learning_rate)

    states: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    expert_actions: list[int] = []
    teacher_kind = config.ppo.bootstrap_teacher_kind
    opponent_factories = [
        lambda seed: RandomAgent(seed=seed),
        lambda seed: WeakHeuristicAgent(seed=seed),
        lambda seed: StrongHeuristicAgent(seed=seed),
    ]

    while len(states) < config.ppo.bootstrap_samples:
        opponent_factory = rng.choice(opponent_factories)
        state = initial_state()
        controlled_player = 1 if rng.random() < 0.5 else 2
        opponent_agent = opponent_factory(config.global_.seed + len(states))
        if teacher_kind == "minimax_1":
            expert_agent = MinimaxAgent(depth=1, seed=config.global_.seed + 500_000 + len(states))
        elif teacher_kind == "mixed_strong_minimax_1":
            if rng.random() < 0.5:
                expert_agent = StrongHeuristicAgent(seed=config.global_.seed + 500_000 + len(states))
            else:
                expert_agent = MinimaxAgent(depth=1, seed=config.global_.seed + 500_000 + len(states))
        else:
            expert_agent = StrongHeuristicAgent(seed=config.global_.seed + 500_000 + len(states))
        while not is_terminal(state) and len(states) < config.ppo.bootstrap_samples:
            if state.current_player == controlled_player:
                legal = legal_actions(state)
                expert_action = expert_agent.select_action(state, legal)
                states.append(np.asarray(encode_state(state, controlled_player), dtype=np.float32))
                action_masks.append(legal_actions_to_mask(legal))
                expert_actions.append(expert_action)
                action = expert_action if rng.random() < 0.70 else rng.choice(legal)
            else:
                action = opponent_agent.select_action(state, legal_actions(state))
            state = apply_action(state, action)

    state_tensor = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
    action_mask_tensor = torch.tensor(np.stack(action_masks), dtype=torch.bool, device=device)
    expert_action_tensor = torch.tensor(expert_actions, dtype=torch.int64, device=device)

    losses: list[float] = []
    network.train()
    for _ in range(config.ppo.bootstrap_epochs):
        indices = torch.randperm(state_tensor.shape[0], device=device)
        for start in range(0, state_tensor.shape[0], config.ppo.bootstrap_batch_size):
            end = min(start + config.ppo.bootstrap_batch_size, state_tensor.shape[0])
            batch_idx = indices[start:end]
            logits, _values = network(state_tensor[batch_idx])
            masked_logits = masked_logits_from_logits(logits, action_mask_tensor[batch_idx])
            loss = torch.nn.functional.cross_entropy(masked_logits, expert_action_tensor[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.ppo.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
    network.eval()

    with torch.no_grad():
        logits, _values = network(state_tensor)
        masked_logits = masked_logits_from_logits(logits, action_mask_tensor)
        predictions = masked_logits.argmax(dim=1)
        accuracy = float((predictions == expert_action_tensor).float().mean().item())

    return clone_state_dict(network), {
        "samples": len(states),
        "epochs": config.ppo.bootstrap_epochs,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "last_loss": float(losses[-1]) if losses else 0.0,
        "expert_match_accuracy": accuracy,
    }


def count_immediate_winning_actions(state, player: int) -> int:
    wins = 0
    for action in legal_actions(state):
        if state.current_player == player:
            candidate_state = state
        else:
            candidate_state = type(state)(
                board=state.board,
                current_player=player,
                winner=state.winner,
                moves_played=state.moves_played,
                last_action=state.last_action,
            )
        next_state = apply_action(candidate_state, action)
        if next_state.winner == player:
            wins += 1
    return wins


def center_control_score(state, player: int) -> float:
    center_column = len(state.board[0]) // 2
    return float(sum(1 for row in range(len(state.board)) if state.board[row][center_column] == player))
