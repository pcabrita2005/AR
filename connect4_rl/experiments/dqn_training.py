from __future__ import annotations

import copy
import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from connect4_rl.agents.baselines import RandomAgent, StrongHeuristicAgent, WeakHeuristicAgent
from connect4_rl.agents.learning.dqn import (
    ConnectFourQNetwork,
    DQNAgent,
    ReplayBuffer,
    build_network_from_state_dict,
    clone_state_dict,
    flip_action_horizontally,
    flip_action_mask_horizontally,
    flip_state_horizontally,
    legal_actions_to_mask,
    state_to_numpy,
)
from connect4_rl.config import Config
from connect4_rl.envs.connect_four import ConnectFourState, apply_action, initial_state, is_terminal, legal_actions, outcome_for_player
from connect4_rl.experiments.dqn_curriculum_utils import RewardProfile, check_vertical_win, count_winnable_windows
from connect4_rl.utils.seed_utils import set_all_seeds


@dataclass
class DQNLessonConfig:
    name: str
    opponent: str
    eval_opponent: str
    max_train_episodes: int
    pretrained_path: str | None = None
    save_path: str | None = None
    opponent_pool_size: int = 6
    opponent_upgrade: int = 6000
    buffer_warm_up: bool = False
    warm_up_opponent: str | None = None
    agent_warm_up: int = 0
    block_vert_coef: float = 1.0
    learning_rate_scale: float = 1.0
    epsilon_start: float | None = None
    epsilon_end: float | None = None
    rewards: RewardProfile = field(default_factory=RewardProfile)


@dataclass
class DQNTrainingMetrics:
    config: dict[str, object]
    lesson_summaries: list[dict[str, object]] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    evaluation: list[dict[str, float | str]] = field(default_factory=list)
    replay_sizes: list[int] = field(default_factory=list)
    opponent_kinds: list[str] = field(default_factory=list)
    curriculum_name: str = "dqn_tutorial_pipeline"
    curriculum_description: str = "Full tutorial-style DQN pipeline with curriculum, replay warmup, self-play and population evolution."
    phase_summary: list[dict[str, object]] = field(default_factory=list)
    population_history: list[dict[str, object]] = field(default_factory=list)
    best_checkpoint_path: str = ""
    best_score: float = float("-inf")
    best_vs_strong_checkpoint_path: str = ""
    best_vs_strong_win_rate: float = float("-inf")


@dataclass
class EvoDQNMember:
    member_id: int
    network: ConnectFourQNetwork
    target_network: ConnectFourQNetwork
    optimizer: torch.optim.Optimizer
    learning_rate: float
    batch_size: int
    learn_step: int
    scores: list[float] = field(default_factory=list)
    fitness: list[float] = field(default_factory=list)

    def clone(self, *, device: torch.device) -> "EvoDQNMember":
        network = copy.deepcopy(self.network).to(device)
        target_network = copy.deepcopy(self.target_network).to(device)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate)
        return EvoDQNMember(
            member_id=self.member_id,
            network=network,
            target_network=target_network,
            optimizer=optimizer,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            learn_step=self.learn_step,
            scores=list(self.scores),
            fitness=list(self.fitness),
        )


class CurriculumRandomAgent:
    def __init__(self, *, seed: int, block_vertical_bias: float = 1.0, name: str = "curriculum_random") -> None:
        self.name = name
        self._rng = random.Random(seed)
        self.block_vertical_bias = block_vertical_bias

    def select_action(self, state: ConnectFourState, legal_actions_now) -> int:
        legal = list(legal_actions_now)
        if len(legal) == 1:
            return legal[0]
        weights = [1.0 for _ in legal]
        if state.last_action is not None and state.last_action in legal:
            weights[legal.index(state.last_action)] *= max(self.block_vertical_bias, 1.0)
        return int(self._rng.choices(legal, weights=weights, k=1)[0])


def train_dqn_self_play(
    config: Config | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
    lessons_dir: str | Path | None = None,
) -> tuple[DQNAgent, DQNTrainingMetrics]:
    from connect4_rl.config import get_default_config

    config = copy.deepcopy(config or get_default_config())
    lessons = build_tutorial_dqn_lessons(config.dqn.episodes, lessons_dir=lessons_dir)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    aggregate = DQNTrainingMetrics(config={**asdict(config.dqn), "seed": config.global_.seed})
    initial_state_dict: dict[str, torch.Tensor] | None = None
    final_agent: DQNAgent | None = None
    episode_offset = 0
    global_best_state_dict: dict[str, torch.Tensor] | None = None
    global_best_checkpoint_path = ""
    global_best_vs_strong_state_dict: dict[str, torch.Tensor] | None = None
    global_best_vs_strong_checkpoint_path = ""
    global_best_vs_strong_win_rate = float("-inf")

    for lesson_index, lesson in enumerate(lessons, start=1):
        lesson_checkpoint_dir = checkpoint_root / lesson.name if checkpoint_root is not None else None
        final_agent, lesson_metrics, final_state_dict = train_dqn_lesson_population(
            lesson=lesson,
            config=config,
            checkpoint_dir=lesson_checkpoint_dir,
            initial_state_dict=initial_state_dict,
        )
        initial_state_dict = final_state_dict
        if global_best_state_dict is None or lesson_metrics.best_score >= aggregate.best_score:
            global_best_state_dict = {key: value.detach().cpu().clone() for key, value in final_state_dict.items()}
            global_best_checkpoint_path = lesson_metrics.best_checkpoint_path
        if lesson_metrics.best_vs_strong_win_rate >= global_best_vs_strong_win_rate:
            global_best_vs_strong_win_rate = lesson_metrics.best_vs_strong_win_rate
            global_best_vs_strong_checkpoint_path = lesson_metrics.best_vs_strong_checkpoint_path
            if lesson_metrics.best_vs_strong_checkpoint_path and lesson_checkpoint_dir is not None:
                global_best_vs_strong_state_dict = torch.load(
                    lesson_metrics.best_vs_strong_checkpoint_path,
                    map_location=config.resolve_device(),
                )

        aggregate.episode_rewards.extend(lesson_metrics.episode_rewards)
        aggregate.episode_lengths.extend(lesson_metrics.episode_lengths)
        aggregate.epsilons.extend(lesson_metrics.epsilons)
        aggregate.losses.extend(lesson_metrics.losses)
        aggregate.replay_sizes.extend(lesson_metrics.replay_sizes)
        aggregate.opponent_kinds.extend(lesson_metrics.opponent_kinds)
        aggregate.population_history.extend(lesson_metrics.population_history)

        for phase in lesson_metrics.phase_summary:
            shifted = dict(phase)
            shifted["start_episode"] = int(shifted["start_episode"]) + episode_offset
            shifted["end_episode"] = int(shifted["end_episode"]) + episode_offset
            aggregate.phase_summary.append(shifted)

        for evaluation in lesson_metrics.evaluation:
            shifted = dict(evaluation)
            shifted["episode"] = float(shifted["episode"]) + episode_offset
            aggregate.evaluation.append(shifted)

        last_eval = lesson_metrics.evaluation[-1] if lesson_metrics.evaluation else {}
        aggregate.lesson_summaries.append(
            {
                "lesson_index": lesson_index,
                "lesson_name": lesson.name,
                "opponent": lesson.opponent,
                "eval_opponent": lesson.eval_opponent,
                "episodes": lesson.max_train_episodes,
                "best_score": lesson_metrics.best_score,
                "best_checkpoint_path": lesson_metrics.best_checkpoint_path,
                "last_eval": last_eval,
            }
        )
        episode_offset += lesson.max_train_episodes
        aggregate.best_score = max(aggregate.best_score, lesson_metrics.best_score)

    if final_agent is None or initial_state_dict is None or global_best_state_dict is None:
        raise RuntimeError("Tutorial DQN pipeline produced no final agent.")

    if checkpoint_root is not None:
        final_path = checkpoint_root / "dqn_best.pt"
        torch.save(global_best_state_dict, final_path)
        aggregate.best_checkpoint_path = str(final_path)
        if global_best_vs_strong_state_dict is not None:
            best_vs_strong_path = checkpoint_root / "dqn_best_vs_strong.pt"
            torch.save(global_best_vs_strong_state_dict, best_vs_strong_path)
            aggregate.best_vs_strong_checkpoint_path = str(best_vs_strong_path)
            aggregate.best_vs_strong_win_rate = global_best_vs_strong_win_rate
        write_metrics_snapshot(aggregate, checkpoint_root / "metrics_latest.json")
        write_metrics_snapshot(aggregate, checkpoint_root / "metrics_final.json")
    else:
        aggregate.best_checkpoint_path = global_best_checkpoint_path
        aggregate.best_vs_strong_checkpoint_path = global_best_vs_strong_checkpoint_path
        aggregate.best_vs_strong_win_rate = global_best_vs_strong_win_rate

    best_network = build_network_from_state_dict(
        global_best_state_dict,
        device=config.resolve_device(),
        hidden_dim=config.dqn.hidden_dim,
        channel_sizes=config.dqn.channel_sizes,
        kernel_sizes=config.dqn.kernel_sizes,
        stride_sizes=config.dqn.stride_sizes,
        head_hidden_sizes=config.dqn.head_hidden_sizes,
        use_dueling_head=config.dqn.use_dueling_head,
    )
    return DQNAgent(best_network, device=config.resolve_device(), epsilon=0.0, seed=config.global_.seed), aggregate


def train_dqn_lesson_population(
    *,
    lesson: DQNLessonConfig,
    config: Config,
    checkpoint_dir: str | Path | None = None,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[DQNAgent, DQNTrainingMetrics, dict[str, torch.Tensor]]:
    set_all_seeds(config.global_.seed)
    rng = random.Random(config.global_.seed)
    device = torch.device(config.resolve_device())

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    population = create_population(config, device=device, initial_state_dict=initial_state_dict, pretrained_path=lesson.pretrained_path)
    if lesson.learning_rate_scale != 1.0:
        for member in population:
            member.learning_rate = float(max(member.learning_rate * lesson.learning_rate_scale, 1e-6))
            member.optimizer = torch.optim.AdamW(member.network.parameters(), lr=member.learning_rate)
    shared_memory = ReplayBuffer(config.dqn.replay_buffer_size)

    metrics = DQNTrainingMetrics(
        config={**asdict(config.dqn), "seed": config.global_.seed},
        curriculum_name=lesson.name,
        curriculum_description=f"Tutorial lesson vs {lesson.opponent} with population-based DQN training.",
        phase_summary=[
            {
                "phase_name": lesson.name,
                "opponent_kind": lesson.opponent,
                "episodes": lesson.max_train_episodes,
                "start_episode": 1,
                "end_episode": lesson.max_train_episodes,
                "lesson_name": lesson.name,
            }
        ],
    )

    if lesson.buffer_warm_up:
        warmup_opponent = build_lesson_opponent(
            lesson.warm_up_opponent or lesson.opponent,
            config=config,
            episode=0,
            rng=rng,
            opponent_pool=deque(),
            block_vert_coef=lesson.block_vert_coef,
        )
        fill_replay_buffer(
            replay=shared_memory,
            opponent_agent=warmup_opponent,
            rewards=lesson.rewards,
            config=config,
            rng=rng,
        )
        if lesson.agent_warm_up > 0:
            base_member = population[0]
            for _ in range(lesson.agent_warm_up):
                loss = maybe_update_member(base_member, shared_memory, rng, config, device)
                if loss is not None:
                    metrics.losses.append(loss)
            population = clone_population_from_base(
                base_member,
                config,
                device=device,
                population_size=config.dqn.population_size,
            )

    opponent_pool: deque[dict[str, torch.Tensor]] = deque(maxlen=max(lesson.opponent_pool_size, 1))
    if lesson.opponent == "self_play":
        elite_member = population[0]
        for _ in range(max(lesson.opponent_pool_size, 1)):
            opponent_pool.append(clone_state_dict(elite_member.network))

    total_episodes = 0
    elite = population[0]
    best_score = float("-inf")
    best_state_dict = clone_state_dict(elite.network)
    best_checkpoint = ""
    previous_eval_state_dict: dict[str, torch.Tensor] | None = None
    lesson_epsilon_start = lesson.epsilon_start if lesson.epsilon_start is not None else config.dqn.epsilon_start
    lesson_epsilon_end = lesson.epsilon_end if lesson.epsilon_end is not None else config.dqn.epsilon_end
    epsilon = lesson_epsilon_start
    lesson_decay_rate = compute_lesson_decay_rate(
        config,
        lesson.max_train_episodes,
        epsilon_start=lesson_epsilon_start,
        epsilon_end=lesson_epsilon_end,
    )
    best_vs_strong_win_rate = float("-inf")
    best_vs_strong_checkpoint = ""
    no_improve_vs_strong_evals = 0

    if lesson.max_train_episodes == 0:
        evaluation = evaluate_member(elite, lesson, config, previous_eval_state_dict)
        evaluation["episode"] = 0.0
        evaluation["lesson_name"] = lesson.name
        metrics.evaluation.append(evaluation)
        best_score = float(evaluation["eval_mean_outcome"])
        best_state_dict = clone_state_dict(elite.network)
        best_checkpoint = write_best_checkpoint(best_state_dict, checkpoint_path, lesson.save_path)
        metrics.best_checkpoint_path = best_checkpoint
        metrics.best_score = best_score
        metrics.best_vs_strong_checkpoint_path = best_checkpoint
        metrics.best_vs_strong_win_rate = float(evaluation.get("vs_strong_heuristic_win_rate", 0.0))
        if checkpoint_path is not None:
            write_metrics_snapshot(metrics, checkpoint_path / "metrics_latest.json")
            write_metrics_snapshot(metrics, checkpoint_path / "metrics_final.json")
        return DQNAgent(elite.network, device=config.resolve_device(), epsilon=0.0, seed=config.global_.seed), metrics, best_state_dict

    episodes_per_epoch = max(config.dqn.episodes_per_epoch, 1)
    evo_epochs = max(config.dqn.evo_epochs, 1)
    max_steps = config.dqn.max_steps_per_episode
    max_epochs = max(
        1,
        int(np.ceil(lesson.max_train_episodes / max(episodes_per_epoch * len(population), 1))),
    )
    next_eval_episode = min(max(config.dqn.eval_interval, 1), lesson.max_train_episodes)

    for epoch in range(max_epochs):
        turns_per_epoch = []
        for member in population:
            for _ in range(episodes_per_epoch):
                if total_episodes >= lesson.max_train_episodes:
                    break
                opponent = select_training_opponent(lesson, config, rng, opponent_pool, total_episodes, lesson.block_vert_coef)
                score, turns, episode_reward = run_training_episode(
                    member=member,
                    lesson=lesson,
                    config=config,
                    rng=rng,
                    replay=shared_memory,
                    opponent_agent=opponent,
                    epsilon=epsilon,
                    device=device,
                    max_steps=max_steps,
                    metrics=metrics,
                )
                member.scores.append(score)
                metrics.episode_rewards.append(episode_reward)
                metrics.episode_lengths.append(turns)
                metrics.epsilons.append(epsilon)
                metrics.replay_sizes.append(len(shared_memory))
                metrics.opponent_kinds.append(lesson.opponent)
                turns_per_epoch.append(turns)
                total_episodes += 1
                epsilon = max(lesson_epsilon_end, epsilon * lesson_decay_rate)

                if lesson.opponent == "self_play" and lesson.opponent_upgrade > 0 and total_episodes % lesson.opponent_upgrade == 0:
                    elite_snapshot = select_elite(population)
                    opponent_pool.append(clone_state_dict(elite_snapshot.network))
            if total_episodes >= lesson.max_train_episodes:
                break

        should_evaluate = total_episodes >= next_eval_episode or total_episodes >= lesson.max_train_episodes
        if should_evaluate:
            elite, previous_eval_state_dict, elite_score, elite_eval = evaluate_population(
                population=population,
                lesson=lesson,
                config=config,
                previous_eval_state_dict=previous_eval_state_dict,
                metrics=metrics,
                epoch=epoch + 1,
                total_episodes=total_episodes,
            )
            if elite_score >= best_score:
                best_score = elite_score
                best_state_dict = clone_state_dict(elite.network)
                best_checkpoint = write_best_checkpoint(best_state_dict, checkpoint_path, lesson.save_path)
                metrics.best_checkpoint_path = best_checkpoint

            current_vs_strong = float(elite_eval.get("vs_strong_heuristic_win_rate", 0.0))
            if current_vs_strong > best_vs_strong_win_rate:
                best_vs_strong_win_rate = current_vs_strong
                no_improve_vs_strong_evals = 0
                if checkpoint_path is not None:
                    best_vs_strong_path = checkpoint_path / "dqn_best_vs_strong.pt"
                    torch.save(clone_state_dict(elite.network), best_vs_strong_path)
                    best_vs_strong_checkpoint = str(best_vs_strong_path)
                elif best_checkpoint:
                    best_vs_strong_checkpoint = best_checkpoint
            else:
                no_improve_vs_strong_evals += 1

            if checkpoint_path is not None:
                torch.save(elite.network.state_dict(), checkpoint_path / f"dqn_episode_{total_episodes:04d}.pt")
                write_metrics_snapshot(metrics, checkpoint_path / "metrics_latest.json")

            while next_eval_episode <= total_episodes:
                next_eval_episode += max(config.dqn.eval_interval, 1)

        if ((epoch + 1) % evo_epochs == 0 or total_episodes >= lesson.max_train_episodes) and any(member.fitness for member in population):
            population = tournament_select_and_mutate(population, rng, device=device, config=config)

        if (
            normalize_opponent(lesson.opponent) == "self_play"
            and total_episodes >= config.dqn.self_play_min_episodes_before_early_stop
            and no_improve_vs_strong_evals >= config.dqn.self_play_early_stop_patience_evals
        ):
            break

        if total_episodes >= lesson.max_train_episodes:
            break

    metrics.best_score = best_score
    metrics.best_vs_strong_checkpoint_path = best_vs_strong_checkpoint
    metrics.best_vs_strong_win_rate = best_vs_strong_win_rate
    if checkpoint_path is not None:
        write_metrics_snapshot(metrics, checkpoint_path / "metrics_final.json")
    final_network = build_network_from_state_dict(
        best_state_dict,
        device=config.resolve_device(),
        hidden_dim=config.dqn.hidden_dim,
        channel_sizes=config.dqn.channel_sizes,
        kernel_sizes=config.dqn.kernel_sizes,
        stride_sizes=config.dqn.stride_sizes,
        head_hidden_sizes=config.dqn.head_hidden_sizes,
        use_dueling_head=config.dqn.use_dueling_head,
    )
    final_agent = DQNAgent(final_network, device=config.resolve_device(), epsilon=0.0, seed=config.global_.seed)
    return final_agent, metrics, best_state_dict


def create_population(
    config: Config,
    *,
    device: torch.device,
    initial_state_dict: dict[str, torch.Tensor] | None,
    pretrained_path: str | None,
) -> list[EvoDQNMember]:
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=device)
    else:
        state_dict = initial_state_dict

    population = []
    for idx in range(config.dqn.population_size):
        network = build_dqn_network_from_config(config).to(device)
        if state_dict is not None:
            network.load_state_dict(state_dict)
        target_network = build_dqn_network_from_config(config).to(device)
        target_network.load_state_dict(network.state_dict())
        optimizer = torch.optim.AdamW(network.parameters(), lr=config.dqn.learning_rate)
        population.append(
            EvoDQNMember(
                member_id=idx,
                network=network,
                target_network=target_network,
                optimizer=optimizer,
                learning_rate=config.dqn.learning_rate,
                batch_size=config.dqn.batch_size,
                learn_step=config.dqn.learn_step,
            )
        )
    return population


def clone_population_from_base(base_member: EvoDQNMember, config: Config, *, device: torch.device, population_size: int) -> list[EvoDQNMember]:
    base_state = clone_state_dict(base_member.network)
    population = []
    for idx in range(population_size):
        network = build_network_from_state_dict(
            base_state,
            device=str(device),
            hidden_dim=config.dqn.hidden_dim,
            channel_sizes=config.dqn.channel_sizes,
            kernel_sizes=config.dqn.kernel_sizes,
            stride_sizes=config.dqn.stride_sizes,
            head_hidden_sizes=config.dqn.head_hidden_sizes,
            use_dueling_head=config.dqn.use_dueling_head,
        )
        target_network = build_network_from_state_dict(
            base_state,
            device=str(device),
            hidden_dim=config.dqn.hidden_dim,
            channel_sizes=config.dqn.channel_sizes,
            kernel_sizes=config.dqn.kernel_sizes,
            stride_sizes=config.dqn.stride_sizes,
            head_hidden_sizes=config.dqn.head_hidden_sizes,
            use_dueling_head=config.dqn.use_dueling_head,
        )
        optimizer = torch.optim.AdamW(network.parameters(), lr=base_member.learning_rate)
        population.append(
            EvoDQNMember(
                member_id=idx,
                network=network,
                target_network=target_network,
                optimizer=optimizer,
                learning_rate=base_member.learning_rate,
                batch_size=base_member.batch_size,
                learn_step=base_member.learn_step,
            )
        )
    return population


def select_training_opponent(
    lesson: DQNLessonConfig,
    config: Config,
    rng: random.Random,
    opponent_pool: deque[dict[str, torch.Tensor]],
    episode: int,
    block_vert_coef: float,
):
    if normalize_opponent(lesson.opponent) == "self_play":
        draw = rng.random()
        if draw < config.dqn.random_opponent_fraction:
            return build_lesson_opponent(
                "random",
                config=config,
                episode=episode,
                rng=rng,
                opponent_pool=opponent_pool,
                block_vert_coef=block_vert_coef,
            )
        if draw < config.dqn.random_opponent_fraction + config.dqn.heuristic_opponent_fraction:
            return build_lesson_opponent(
                "strong",
                config=config,
                episode=episode,
                rng=rng,
                opponent_pool=opponent_pool,
                block_vert_coef=block_vert_coef,
            )
    return build_lesson_opponent(
        lesson.opponent,
        config=config,
        episode=episode,
        rng=rng,
        opponent_pool=opponent_pool,
        block_vert_coef=block_vert_coef,
    )


def run_training_episode(
    *,
    member: EvoDQNMember,
    lesson: DQNLessonConfig,
    config: Config,
    rng: random.Random,
    replay: ReplayBuffer,
    opponent_agent,
    epsilon: float,
    device: torch.device,
    max_steps: int,
    metrics: DQNTrainingMetrics,
) -> tuple[float, int, float]:
    controlled_player = 2 if rng.random() > 0.5 else 1
    state = initial_state()
    pending: tuple[np.ndarray, int] | None = None
    episode_reward = 0.0
    score = 0.0
    turns = 0
    last_online_action: int | None = None
    acting_agent = DQNAgent(member.network, device=config.resolve_device(), epsilon=epsilon, seed=config.global_.seed + replay.counter)

    for _ in range(max_steps):
        if state.current_player == controlled_player:
            legal = legal_actions(state)
            state_array = state_to_numpy(state, controlled_player)
            action = acting_agent.get_action(state_array, epsilon=epsilon, action_mask=legal_actions_to_mask(legal))[0]
            next_state = apply_action(state, action)
            turns += 1
            last_online_action = action
            if is_terminal(next_state):
                reward = shaped_reward(next_state, controlled_player, lesson.rewards, done=True)
                episode_reward = reward
                score = outcome_for_player(next_state, controlled_player)
                add_transition(
                    replay=replay,
                    config=config,
                    state=state_array,
                    action=action,
                    reward=reward,
                    next_state=np.zeros((2, 6, 7), dtype=np.float32),
                    done=True,
                    next_action_mask=np.zeros(7, dtype=np.float32),
                )
                maybe_learn_member(member, replay, rng, config, device, metrics)
                break
            pending = (state_array, action)
            state = next_state
        else:
            legal = legal_actions(state)
            action = select_opponent_action(
                opponent_agent=opponent_agent,
                state=state,
                legal=legal,
                last_online_action=last_online_action,
                block_vert_coef=lesson.block_vert_coef,
            )
            next_state = apply_action(state, action)
            turns += 1
            if pending is not None:
                done = is_terminal(next_state)
                reward = shaped_reward(next_state, controlled_player, lesson.rewards, done=done)
                episode_reward = reward if done else episode_reward
                score = outcome_for_player(next_state, controlled_player) if done else score
                next_state_array = np.zeros((2, 6, 7), dtype=np.float32) if done else state_to_numpy(next_state, controlled_player)
                next_mask = np.zeros(7, dtype=np.float32) if done else legal_actions_to_mask(legal_actions(next_state))
                add_transition(
                    replay=replay,
                    config=config,
                    state=pending[0],
                    action=pending[1],
                    reward=reward,
                    next_state=next_state_array,
                    done=done,
                    next_action_mask=next_mask,
                )
                pending = None
                maybe_learn_member(member, replay, rng, config, device, metrics)
                if done:
                    break
            state = next_state
    return score, turns, episode_reward


def maybe_learn_member(
    member: EvoDQNMember,
    replay: ReplayBuffer,
    rng: random.Random,
    config: Config,
    device: torch.device,
    metrics: DQNTrainingMetrics,
) -> None:
    if replay.counter % max(member.learn_step, 1) != 0:
        return
    loss = maybe_update_member(member, replay, rng, config, device)
    if loss is not None:
        metrics.losses.append(loss)


def maybe_update_member(
    member: EvoDQNMember,
    replay: ReplayBuffer,
    rng: random.Random,
    config: Config,
    device: torch.device,
) -> float | None:
    if len(replay) < max(config.dqn.min_replay_size, member.batch_size):
        return None

    states, actions, rewards, next_states, dones, next_masks = replay.sample(member.batch_size, rng)
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
    next_masks_t = torch.tensor(next_masks, dtype=torch.bool, device=device)

    q_values = member.network(states_t).gather(1, actions_t).squeeze(1)
    with torch.no_grad():
        next_online_q = member.network(next_states_t).masked_fill(~next_masks_t, -1e9)
        next_actions = next_online_q.argmax(dim=1, keepdim=True)
        next_target_q = member.target_network(next_states_t).gather(1, next_actions).squeeze(1)
        next_target_q = torch.where(dones_t > 0.5, torch.zeros_like(next_target_q), next_target_q)
        targets = rewards_t + config.dqn.gamma * next_target_q

    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    member.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(member.network.parameters(), 5.0)
    member.optimizer.step()
    soft_update_target_network(member.network, member.target_network, tau=config.dqn.tau)
    return float(loss.item())


def select_elite(population: list[EvoDQNMember]) -> EvoDQNMember:
    return max(population, key=lambda member: member.fitness[-1] if member.fitness else float("-inf"))


def evaluate_population(
    *,
    population: list[EvoDQNMember],
    lesson: DQNLessonConfig,
    config: Config,
    previous_eval_state_dict: dict[str, torch.Tensor] | None,
    metrics: DQNTrainingMetrics,
    epoch: int,
    total_episodes: int,
) -> tuple[EvoDQNMember, dict[str, torch.Tensor], float]:
    for member in population:
        evaluation = evaluate_member(member, lesson, config, previous_eval_state_dict)
        mean_fit = float(evaluation["eval_mean_outcome"])
        member.fitness.append(mean_fit)
        metrics.population_history.append(
            {
                "lesson_name": lesson.name,
                "epoch": epoch,
                "member_id": member.member_id,
                "fitness": mean_fit,
                "learning_rate": member.learning_rate,
                "batch_size": member.batch_size,
                "learn_step": member.learn_step,
            }
        )

    elite = select_elite(population)
    elite_eval = evaluate_member(elite, lesson, config, previous_eval_state_dict)
    elite_eval["episode"] = float(total_episodes)
    elite_eval["lesson_name"] = lesson.name
    elite_eval["selection_score"] = score_evaluation_for_checkpoint(elite_eval, config)
    metrics.evaluation.append(elite_eval)
    return elite, clone_state_dict(elite.network), float(elite_eval["selection_score"]), elite_eval


def compute_lesson_decay_rate(
    config: Config,
    lesson_episodes: int,
    *,
    epsilon_start: float,
    epsilon_end: float,
) -> float:
    if lesson_episodes <= 1:
        return epsilon_end
    if epsilon_start <= epsilon_end:
        return 1.0
    target_rate = (epsilon_end / epsilon_start) ** (1.0 / max(lesson_episodes - 1, 1))
    return min(config.dqn.epsilon_decay_rate, target_rate)


def score_evaluation_for_checkpoint(evaluation: dict[str, float | str], config: Config) -> float:
    eval_mean_outcome = float(evaluation.get("eval_mean_outcome", 0.0))
    vs_random = float(evaluation.get("vs_random_win_rate", 0.0))
    vs_weak = float(evaluation.get("vs_weak_heuristic_win_rate", 0.0))
    vs_strong = float(evaluation.get("vs_strong_heuristic_win_rate", 0.0))
    return eval_mean_outcome + (0.5 * vs_random) + vs_weak + (config.dqn.checkpoint_score_heuristic_weight * vs_strong)


def tournament_select_and_mutate(
    population: list[EvoDQNMember],
    rng: random.Random,
    *,
    device: torch.device,
    config: Config,
) -> list[EvoDQNMember]:
    elite = select_elite(population)
    next_population = [elite.clone(device=device)]
    next_population[0].member_id = 0

    while len(next_population) < config.dqn.population_size:
        contenders = rng.sample(population, k=min(config.dqn.tournament_size, len(population)))
        winner = max(contenders, key=lambda member: member.fitness[-1] if member.fitness else float("-inf"))
        child = winner.clone(device=device)
        mutate_member(child, rng, config)
        child.member_id = len(next_population)
        next_population.append(child)
    return next_population


def mutate_member(member: EvoDQNMember, rng: random.Random, config: Config) -> None:
    mutation_draw = rng.random()
    if mutation_draw < config.dqn.no_mutation_prob:
        return
    if mutation_draw < config.dqn.no_mutation_prob + config.dqn.mutation_lr_prob:
        scale = rng.choice([config.dqn.mutation_shrink_factor, config.dqn.mutation_grow_factor])
        member.learning_rate = float(np.clip(member.learning_rate * scale, config.dqn.mutation_min_lr, config.dqn.mutation_max_lr))
    elif mutation_draw < config.dqn.no_mutation_prob + config.dqn.mutation_lr_prob + config.dqn.mutation_batch_prob:
        scale = rng.choice([config.dqn.mutation_shrink_factor, config.dqn.mutation_grow_factor])
        candidate = int(round(member.batch_size * scale))
        member.batch_size = int(np.clip(candidate, config.dqn.mutation_min_batch_size, config.dqn.mutation_max_batch_size))
    else:
        scale = rng.choice([config.dqn.mutation_shrink_factor, config.dqn.mutation_grow_factor])
        candidate = int(round(member.learn_step * scale))
        member.learn_step = int(np.clip(candidate, config.dqn.mutation_min_learn_step, config.dqn.mutation_max_learn_step))
    member.optimizer = torch.optim.AdamW(member.network.parameters(), lr=member.learning_rate)


def build_tutorial_dqn_lessons(total_episodes: int | None = None, *, lessons_dir: str | Path | None = None) -> list[DQNLessonConfig]:
    lessons = load_dqn_lessons(lessons_dir)
    if total_episodes is None or total_episodes <= 0:
        return lessons

    positive_lessons = [lesson for lesson in lessons if lesson.max_train_episodes > 0]
    weights = [lesson.max_train_episodes for lesson in positive_lessons]
    raw_counts = [weight * total_episodes / sum(weights) for weight in weights]
    counts = [int(value) for value in raw_counts]
    remainder = total_episodes - sum(counts)
    order = sorted(range(len(raw_counts)), key=lambda idx: raw_counts[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1

    adjusted = []
    count_idx = 0
    for lesson in lessons:
        copied = copy.deepcopy(lesson)
        if copied.max_train_episodes > 0:
            copied.max_train_episodes = counts[count_idx]
            count_idx += 1
        adjusted.append(copied)
    return adjusted


def load_dqn_lessons(lessons_dir: str | Path | None = None) -> list[DQNLessonConfig]:
    directory = Path(lessons_dir) if lessons_dir is not None else default_dqn_lessons_dir()
    lesson_files = sorted(directory.glob("lesson*.yaml"))
    if not lesson_files:
        raise FileNotFoundError(f"No DQN lesson YAML files found in {directory}")

    lessons = []
    for lesson_path in lesson_files:
        data = yaml.safe_load(lesson_path.read_text(encoding="utf-8")) or {}
        lessons.append(
            DQNLessonConfig(
                name=str(data.get("name") or lesson_path.stem),
                opponent=normalize_opponent(data.get("opponent", "random")),
                eval_opponent=normalize_opponent(data.get("eval_opponent") or data.get("opponent", "random")),
                max_train_episodes=int(data.get("max_train_episodes", 0)),
                pretrained_path=data.get("pretrained_path"),
                save_path=data.get("save_path"),
                opponent_pool_size=int(data.get("opponent_pool_size") or 6),
                opponent_upgrade=int(data.get("opponent_upgrade") or 6000),
                buffer_warm_up=bool(data.get("buffer_warm_up", False)),
                warm_up_opponent=normalize_opponent(data.get("warm_up_opponent")) if data.get("warm_up_opponent") else None,
                agent_warm_up=int(data.get("agent_warm_up", 0)),
                block_vert_coef=float(data.get("block_vert_coef", 1.0)),
                learning_rate_scale=float(data.get("learning_rate_scale", 1.0)),
                epsilon_start=float(data["epsilon_start"]) if data.get("epsilon_start") is not None else None,
                epsilon_end=float(data["epsilon_end"]) if data.get("epsilon_end") is not None else None,
                rewards=RewardProfile(**(data.get("rewards") or {})),
            )
        )
    return lessons


def default_dqn_lessons_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "curriculums" / "connect_four_dqn"


def normalize_opponent(kind: str | None) -> str:
    normalized = (kind or "random").strip().lower()
    aliases = {"self": "self_play", "self-play": "self_play", "heuristic": "strong"}
    return aliases.get(normalized, normalized)


def build_dqn_network_from_config(config: Config) -> ConnectFourQNetwork:
    return ConnectFourQNetwork(
        hidden_dim=config.dqn.hidden_dim,
        channel_sizes=config.dqn.channel_sizes,
        kernel_sizes=config.dqn.kernel_sizes,
        stride_sizes=config.dqn.stride_sizes,
        head_hidden_sizes=config.dqn.head_hidden_sizes,
        use_dueling_head=config.dqn.use_dueling_head,
    )


def fill_replay_buffer(
    *,
    replay: ReplayBuffer,
    opponent_agent,
    rewards: RewardProfile,
    config: Config,
    rng: random.Random,
) -> None:
    while len(replay) < replay.capacity:
        controlled_player = 2 if rng.random() > 0.5 else 1
        state = initial_state()
        pending: tuple[np.ndarray, int] | None = None
        last_online_action: int | None = None

        while not is_terminal(state) and len(replay) < replay.capacity:
            if state.current_player == controlled_player:
                action = rng.choice(legal_actions(state))
                next_state = apply_action(state, action)
                state_array = state_to_numpy(state, controlled_player)
                last_online_action = action
                if is_terminal(next_state):
                    reward = shaped_reward(next_state, controlled_player, rewards, done=True)
                    add_transition(
                        replay=replay,
                        config=config,
                        state=state_array,
                        action=action,
                        reward=reward,
                        next_state=np.zeros((2, 6, 7), dtype=np.float32),
                        done=True,
                        next_action_mask=np.zeros(7, dtype=np.float32),
                    )
                    break
                pending = (state_array, action)
                state = next_state
            else:
                action = select_opponent_action(
                    opponent_agent=opponent_agent,
                    state=state,
                    legal=legal_actions(state),
                    last_online_action=last_online_action,
                    block_vert_coef=1.0,
                )
                next_state = apply_action(state, action)
                if pending is not None:
                    done = is_terminal(next_state)
                    reward = shaped_reward(next_state, controlled_player, rewards, done=done)
                    next_state_array = np.zeros((2, 6, 7), dtype=np.float32) if done else state_to_numpy(next_state, controlled_player)
                    next_mask = np.zeros(7, dtype=np.float32) if done else legal_actions_to_mask(legal_actions(next_state))
                    add_transition(
                        replay=replay,
                        config=config,
                        state=pending[0],
                        action=pending[1],
                        reward=reward,
                        next_state=next_state_array,
                        done=done,
                        next_action_mask=next_mask,
                    )
                    pending = None
                state = next_state


def select_opponent_action(
    *,
    opponent_agent,
    state: ConnectFourState,
    legal: list[int],
    last_online_action: int | None,
    block_vert_coef: float,
) -> int:
    if isinstance(opponent_agent, CurriculumRandomAgent):
        if last_online_action is not None:
            temp_state = ConnectFourState(
                board=state.board,
                current_player=state.current_player,
                winner=state.winner,
                moves_played=state.moves_played,
                last_action=last_online_action,
            )
            return opponent_agent.select_action(temp_state, legal)
        return opponent_agent.select_action(state, legal)
    del block_vert_coef
    return opponent_agent.select_action(state, legal)


def build_lesson_opponent(
    opponent_kind: str,
    *,
    config: Config,
    episode: int,
    rng: random.Random,
    opponent_pool: deque[dict[str, torch.Tensor]],
    block_vert_coef: float,
):
    kind = normalize_opponent(opponent_kind)
    if kind == "random":
        return CurriculumRandomAgent(seed=config.global_.seed + episode, block_vertical_bias=block_vert_coef)
    if kind == "weak":
        return WeakHeuristicAgent(seed=config.global_.seed + episode)
    if kind == "strong":
        return StrongHeuristicAgent(seed=config.global_.seed + episode)
    if kind == "self_play":
        state_dict = rng.choice(list(opponent_pool))
        network = build_network_from_state_dict(
            state_dict,
            device=config.resolve_device(),
            hidden_dim=config.dqn.hidden_dim,
            channel_sizes=config.dqn.channel_sizes,
            kernel_sizes=config.dqn.kernel_sizes,
            stride_sizes=config.dqn.stride_sizes,
            head_hidden_sizes=config.dqn.head_hidden_sizes,
            use_dueling_head=config.dqn.use_dueling_head,
        )
        return DQNAgent(network, device=config.resolve_device(), epsilon=config.dqn.opponent_epsilon, seed=config.global_.seed + episode, name="snapshot")
    raise ValueError(f"Unsupported lesson opponent '{opponent_kind}'")


def evaluate_member(
    member: EvoDQNMember,
    lesson: DQNLessonConfig,
    config: Config,
    previous_eval_state_dict: dict[str, torch.Tensor] | None,
) -> dict[str, float | str]:
    agent = DQNAgent(member.network, device=config.resolve_device(), epsilon=0.0, seed=config.global_.seed)
    random_wr = evaluate_against_agent(agent, lambda idx: RandomAgent(seed=config.global_.seed + 10_000 + idx), games=config.dqn.eval_games)
    weak_wr = evaluate_against_agent(agent, lambda idx: WeakHeuristicAgent(seed=config.global_.seed + 20_000 + idx), games=config.dqn.eval_games)
    strong_wr = evaluate_against_agent(agent, lambda idx: StrongHeuristicAgent(seed=config.global_.seed + 30_000 + idx), games=config.dqn.eval_games)
    previous_wr = 0.0
    if previous_eval_state_dict is not None:
        previous_wr = evaluate_against_agent(
            agent,
            lambda idx: DQNAgent(
                build_network_from_state_dict(
                    previous_eval_state_dict,
                    device=config.resolve_device(),
                    hidden_dim=config.dqn.hidden_dim,
                    channel_sizes=config.dqn.channel_sizes,
                    kernel_sizes=config.dqn.kernel_sizes,
                    stride_sizes=config.dqn.stride_sizes,
                    head_hidden_sizes=config.dqn.head_hidden_sizes,
                    use_dueling_head=config.dqn.use_dueling_head,
                ),
                device=config.resolve_device(),
                epsilon=0.0,
                seed=config.global_.seed + 40_000 + idx,
                name="previous_snapshot",
            ),
            games=config.dqn.eval_games,
        )
    eval_outcome = evaluate_mean_outcome(agent, lesson.eval_opponent, games=config.dqn.evo_loop, base_seed=config.global_.seed)
    return {
        "eval_opponent": lesson.eval_opponent,
        "eval_mean_outcome": eval_outcome,
        "vs_random_win_rate": random_wr,
        "vs_weak_heuristic_win_rate": weak_wr,
        "vs_strong_heuristic_win_rate": strong_wr,
        "vs_heuristic_win_rate": strong_wr,
        "vs_previous_win_rate": previous_wr,
    }


def evaluate_mean_outcome(dqn_agent: DQNAgent, opponent_kind: str, *, games: int, base_seed: int) -> float:
    scores = []
    for game_idx in range(games):
        controlled_player = 1 if game_idx % 2 == 0 else 2
        opponent = build_reference_opponent(opponent_kind, seed=base_seed + 50_000 + game_idx)
        scores.append(play_dqn_match(dqn_agent, opponent, controlled_player=controlled_player))
    return float(np.mean(scores)) if scores else 0.0


def build_reference_opponent(name: str, *, seed: int):
    kind = normalize_opponent(name)
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "weak":
        return WeakHeuristicAgent(seed=seed)
    if kind == "strong":
        return StrongHeuristicAgent(seed=seed)
    raise ValueError(f"Unsupported reference opponent '{name}'")


def evaluate_against_agent(dqn_agent: DQNAgent, opponent_factory, *, games: int = 20) -> float:
    wins = 0
    for game_idx in range(games):
        controlled_player = 1 if game_idx % 2 == 0 else 2
        try:
            opponent = opponent_factory(game_idx)
        except TypeError:
            opponent = opponent_factory()
        if play_dqn_match(dqn_agent, opponent, controlled_player=controlled_player) > 0:
            wins += 1
    return wins / games


def play_dqn_match(dqn_agent: DQNAgent, opponent, *, controlled_player: int = 1) -> float:
    state = initial_state()
    while not is_terminal(state):
        if state.current_player == controlled_player:
            action = dqn_agent.select_action(state, legal_actions(state))
        else:
            action = opponent.select_action(state, legal_actions(state))
        state = apply_action(state, action)
    return outcome_for_player(state, controlled_player)


def shaped_reward(state: ConnectFourState, perspective_player: int, rewards: RewardProfile, *, done: bool) -> float:
    if done:
        outcome = outcome_for_player(state, perspective_player)
        if outcome > 0:
            return rewards.vertical_win if check_vertical_win(state, perspective_player) else rewards.win
        if outcome < 0:
            return rewards.lose
        return rewards.play_continues
    opponent = 2 if perspective_player == 1 else 1
    own_three = count_winnable_windows(state, perspective_player)
    opp_three = count_winnable_windows(state, opponent)
    if own_three + opp_three == 0:
        return rewards.play_continues
    return (rewards.three_in_row * own_three) + (rewards.opp_three_in_row * opp_three)


def write_best_checkpoint(best_state_dict: dict[str, torch.Tensor], checkpoint_path: Path | None, save_path: str | None) -> str:
    saved_path = ""
    if checkpoint_path is not None:
        best_path = checkpoint_path / "dqn_best.pt"
        torch.save(best_state_dict, best_path)
        saved_path = str(best_path)
    if save_path:
        lesson_save_path = Path(save_path)
        lesson_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state_dict, lesson_save_path)
        if not saved_path:
            saved_path = str(lesson_save_path)
    return saved_path


def write_metrics_snapshot(metrics: DQNTrainingMetrics, path: Path) -> None:
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def soft_update_target_network(online_net: nn.Module, target_net: nn.Module, *, tau: float) -> None:
    if tau <= 0.0:
        return
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)


def add_transition(
    replay: ReplayBuffer,
    config: Config,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    next_action_mask: np.ndarray,
) -> None:
    replay.add(state, action, reward, next_state, done, next_action_mask)
    if not config.dqn.use_horizontal_symmetry_augmentation:
        return
    replay.add(
        flip_state_horizontally(state),
        flip_action_horizontally(action),
        reward,
        flip_state_horizontally(next_state),
        done,
        flip_action_mask_horizontally(next_action_mask),
    )
