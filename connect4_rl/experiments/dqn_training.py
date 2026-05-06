from __future__ import annotations

import json
import random
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
    epsilon_by_step,
    flip_action_horizontally,
    flip_action_mask_horizontally,
    flip_state_horizontally,
    legal_actions_to_mask,
    state_to_numpy,
)
from connect4_rl.config import Config
from connect4_rl.envs.connect_four import ConnectFourState, apply_action, initial_state, is_terminal, legal_actions, outcome_for_player
from connect4_rl.experiments.dqn_curriculum_utils import (
    CurriculumDefinition,
    CurriculumPhase,
    RewardProfile,
    expand_curriculum_schedule,
    normalize_dqn_opponent_kind,
    shaped_reward,
)
from connect4_rl.utils.seed_utils import set_all_seeds


@dataclass
class DQNTrainingMetrics:
    config: dict[str, object]
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    evaluation: list[dict[str, float]] = field(default_factory=list)
    replay_sizes: list[int] = field(default_factory=list)
    opponent_kinds: list[str] = field(default_factory=list)
    curriculum_name: str = ""
    curriculum_description: str = ""
    phase_sequence: list[str] = field(default_factory=list)
    phase_summary: list[dict[str, object]] = field(default_factory=list)
    best_checkpoint_path: str = ""
    best_score: float = float("-inf")


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
            idx = legal.index(state.last_action)
            weights[idx] *= max(self.block_vertical_bias, 1.0)
        return int(self._rng.choices(legal, weights=weights, k=1)[0])


def train_dqn_self_play(
    config: Config | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
    lessons_dir: str | Path | None = None,
) -> tuple[DQNAgent, DQNTrainingMetrics]:
    from connect4_rl.config import get_default_config

    config = config or get_default_config()
    definition = load_dqn_lesson_definition(lessons_dir)
    return train_dqn_with_curriculum(definition, config, checkpoint_dir=checkpoint_dir)


def train_dqn_with_curriculum(
    definition: CurriculumDefinition,
    config: Config | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
) -> tuple[DQNAgent, DQNTrainingMetrics]:
    from connect4_rl.config import get_default_config

    config = config or get_default_config()
    set_all_seeds(config.global_.seed)
    rng = random.Random(config.global_.seed)

    device = torch.device(config.resolve_device())
    online_net = build_dqn_network_from_config(config).to(device)
    target_net = build_dqn_network_from_config(config).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.AdamW(online_net.parameters(), lr=config.dqn.learning_rate)
    replay = ReplayBuffer(config.dqn.replay_buffer_size)

    opponent_pool = [clone_state_dict(online_net)]
    update_steps = 0
    policy_steps = 0
    metrics = DQNTrainingMetrics(
        config=asdict(config.dqn),
        curriculum_name=definition.name,
        curriculum_description=definition.description,
    )
    best_state_dict = clone_state_dict(online_net)
    previous_eval_state_dict: dict[str, torch.Tensor] | None = None

    schedule, phase_summary = expand_curriculum_schedule(config.dqn.max_episodes, definition, seed=config.global_.seed)
    metrics.phase_sequence = [phase.opponent_kind or "mixed" for phase in schedule]
    metrics.phase_summary = phase_summary

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    warmed_up_phase_names: set[str] = set()

    for episode in range(1, config.dqn.max_episodes + 1):
        phase = schedule[episode - 1]
        if phase.buffer_warm_up and phase.name not in warmed_up_phase_names:
            replay = fill_replay_buffer_for_phase(
                replay,
                online_net,
                config,
                phase,
                rng,
                target_size=max(phase.warmup_replay_fill, max(config.dqn.min_replay_size, config.dqn.batch_size)),
            )
            if phase.agent_warmup_updates > 0 and len(replay) >= config.dqn.batch_size:
                update_steps += warmup_q_network(
                    online_net,
                    target_net,
                    optimizer,
                    replay,
                    config,
                    device,
                    rng,
                    phase.agent_warmup_updates,
                    metrics,
                )
            opponent_pool = [clone_state_dict(online_net)]
            warmed_up_phase_names.add(phase.name)

        opponent_kind, opponent_agent = build_phase_opponent(
            config=config,
            phase=phase,
            episode=episode,
            rng=rng,
            opponent_pool=opponent_pool,
        )
        online_player = 1 if rng.random() < 0.5 else 2
        state = initial_state()
        pending: tuple[np.ndarray, int] | None = None
        episode_reward = 0.0
        episode_steps = 0

        while not is_terminal(state):
            if state.current_player == online_player:
                epsilon = epsilon_by_step(config.dqn, policy_steps)
                online_agent = DQNAgent(
                    online_net,
                    device=config.resolve_device(),
                    epsilon=epsilon,
                    seed=config.global_.seed + policy_steps,
                    name="online",
                )
                action = online_agent.select_action(state, legal_actions(state))
                state_after_online = apply_action(state, action)
                state_array = state_to_numpy(state, online_player)
                policy_steps += 1
                episode_steps += 1

                if is_terminal(state_after_online):
                    reward = shaped_reward(state_after_online, online_player, phase.rewards, done=True)
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
                    episode_reward = reward
                    state = state_after_online
                    pending = None
                    loss = maybe_update_q_network(online_net, target_net, optimizer, replay, rng, config, device)
                    if loss is not None:
                        metrics.losses.append(loss)
                        update_steps += 1
                        soft_update_target_network(online_net, target_net, tau=config.dqn.tau)
                        if update_steps % config.dqn.target_update_freq == 0:
                            target_net.load_state_dict(online_net.state_dict())
                    break

                pending = (state_array, action)
                state = state_after_online
            else:
                action = opponent_agent.select_action(state, legal_actions(state))
                next_state = apply_action(state, action)
                episode_steps += 1

                if pending is not None:
                    done = is_terminal(next_state)
                    reward = shaped_reward(next_state, online_player, phase.rewards, done=done)
                    next_state_array = np.zeros((2, 6, 7), dtype=np.float32) if done else state_to_numpy(next_state, online_player)
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
                    if done:
                        episode_reward = reward
                    pending = None

                    for _ in range(config.dqn.gradient_updates_per_step):
                        loss = maybe_update_q_network(online_net, target_net, optimizer, replay, rng, config, device)
                        if loss is not None:
                            metrics.losses.append(loss)
                            update_steps += 1
                            soft_update_target_network(online_net, target_net, tau=config.dqn.tau)
                            if update_steps % config.dqn.target_update_freq == 0:
                                target_net.load_state_dict(online_net.state_dict())

                state = next_state

        metrics.episode_rewards.append(episode_reward)
        metrics.episode_lengths.append(episode_steps)
        metrics.epsilons.append(epsilon_by_step(config.dqn, policy_steps))
        metrics.replay_sizes.append(len(replay))
        metrics.opponent_kinds.append(opponent_kind)

        refresh_interval = phase.opponent_upgrade_interval or config.dqn.opponent_refresh_interval
        pool_size = phase.opponent_pool_size or config.dqn.opponent_pool_size
        if refresh_interval > 0 and episode % refresh_interval == 0:
            opponent_pool.append(clone_state_dict(online_net))
            opponent_pool = opponent_pool[-pool_size:]

        if checkpoint_path is not None and (episode % config.dqn.eval_interval == 0 or episode == config.dqn.max_episodes):
            torch.save(online_net.state_dict(), checkpoint_path / f"dqn_episode_{episode:04d}.pt")

        if episode % config.dqn.eval_interval == 0 or episode == config.dqn.max_episodes:
            eval_agent = DQNAgent(online_net, device=config.resolve_device(), epsilon=0.0, seed=config.global_.seed)
            random_wr = evaluate_against_agent(
                eval_agent,
                lambda game_idx: RandomAgent(seed=config.global_.seed + 10_000 + game_idx),
                games=config.dqn.eval_games,
            )
            weak_wr = evaluate_against_agent(
                eval_agent,
                lambda game_idx: WeakHeuristicAgent(seed=config.global_.seed + 15_000 + game_idx),
                games=config.dqn.eval_games,
            )
            strong_wr = evaluate_against_agent(
                eval_agent,
                lambda game_idx: StrongHeuristicAgent(seed=config.global_.seed + 20_000 + game_idx),
                games=config.dqn.eval_games,
            )
            previous_wr = 0.0
            if previous_eval_state_dict is not None:
                previous_wr = evaluate_against_agent(
                    eval_agent,
                    lambda game_idx: DQNAgent(
                        build_network_from_state_dict(
                            previous_eval_state_dict,
                            device=config.resolve_device(),
                            hidden_dim=config.dqn.hidden_dim,
                            channel_sizes=config.dqn.channel_sizes,
                            kernel_sizes=config.dqn.kernel_sizes,
                            stride_sizes=config.dqn.stride_sizes,
                            head_hidden_sizes=config.dqn.head_hidden_sizes,
                        ),
                        device=config.resolve_device(),
                        epsilon=0.0,
                        seed=config.global_.seed + 30_000 + game_idx,
                        name="previous_snapshot",
                    ),
                    games=config.dqn.eval_games,
                )
            metrics.evaluation.append(
                {
                    "episode": float(episode),
                    "vs_random_win_rate": random_wr,
                    "vs_weak_heuristic_win_rate": weak_wr,
                    "vs_strong_heuristic_win_rate": strong_wr,
                    "vs_heuristic_win_rate": strong_wr,
                    "vs_previous_win_rate": previous_wr,
                }
            )
            score = random_wr + (2.0 * weak_wr) + (3.0 * strong_wr)
            if score >= metrics.best_score:
                metrics.best_score = score
                best_state_dict = clone_state_dict(online_net)
                if checkpoint_path is not None:
                    best_path = checkpoint_path / "dqn_best.pt"
                    torch.save(best_state_dict, best_path)
                    metrics.best_checkpoint_path = str(best_path)
            if checkpoint_path is not None:
                write_metrics_snapshot(metrics, checkpoint_path / "metrics_latest.json")
            previous_eval_state_dict = clone_state_dict(online_net)

    online_net.load_state_dict(best_state_dict)
    final_agent = DQNAgent(online_net, device=config.resolve_device(), epsilon=0.0, seed=config.global_.seed)
    if checkpoint_path is not None:
        write_metrics_snapshot(metrics, checkpoint_path / "metrics_final.json")
    return final_agent, metrics


def load_dqn_lesson_definition(lessons_dir: str | Path | None = None) -> CurriculumDefinition:
    directory = Path(lessons_dir) if lessons_dir is not None else default_dqn_lessons_dir()
    lesson_files = sorted(directory.glob("lesson*.yaml"))
    if not lesson_files:
        raise FileNotFoundError(f"No lesson YAML files found in {directory}")

    phases: list[CurriculumPhase] = []
    for lesson_path in lesson_files:
        data = yaml.safe_load(lesson_path.read_text(encoding="utf-8")) or {}
        rewards = RewardProfile(**(data.get("rewards") or {}))
        phases.append(
            CurriculumPhase(
                name=str(data.get("name") or lesson_path.stem),
                opponent_kind=normalize_dqn_opponent_kind(data.get("opponent")),
                fraction=float(data.get("fraction", 0.0)),
                rewards=rewards,
                buffer_warm_up=bool(data.get("buffer_warm_up", False)),
                warm_up_opponent=normalize_dqn_opponent_kind(data.get("warm_up_opponent")),
                warmup_replay_fill=int(data.get("warmup_replay_fill", 0)),
                agent_warmup_updates=int(data.get("agent_warmup_updates", data.get("agent_warm_up", 0))),
                block_vertical_bias=float(data.get("block_vertical_bias", data.get("block_vert_coef", 1.0))),
                opponent_pool_size=_optional_int(data.get("opponent_pool_size")),
                opponent_upgrade_interval=_optional_int(data.get("opponent_upgrade")),
            )
        )
    return CurriculumDefinition(
        name="dqn_self_play_lessons",
        description="Sequencia base de licoes do DQN: random, weak, strong e self-play com reward shaping.",
        phases=tuple(phases),
    )


def default_dqn_lessons_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "curriculums" / "connect_four_dqn"


def build_network(config: Config) -> nn.Module:
    return build_dqn_network_from_config(config)


def build_dqn_network_from_config(config: Config) -> ConnectFourQNetwork:
    return ConnectFourQNetwork(
        hidden_dim=config.dqn.hidden_dim,
        channel_sizes=config.dqn.channel_sizes,
        kernel_sizes=config.dqn.kernel_sizes,
        stride_sizes=config.dqn.stride_sizes,
        head_hidden_sizes=config.dqn.head_hidden_sizes,
    )


def maybe_update_q_network(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    rng: random.Random,
    config: Config,
    device: torch.device,
) -> float | None:
    if len(replay) < max(config.dqn.min_replay_size, config.dqn.batch_size):
        return None

    states, actions, rewards, next_states, dones, next_masks = replay.sample(config.dqn.batch_size, rng)
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
    next_masks_t = torch.tensor(next_masks, dtype=torch.bool, device=device)

    online_net.train()
    q_values = online_net(states_t).gather(1, actions_t).squeeze(1)

    with torch.no_grad():
        next_online_q_values = online_net(next_states_t)
        next_online_q_values = next_online_q_values.masked_fill(~next_masks_t, -1e9)
        next_actions = next_online_q_values.argmax(dim=1, keepdim=True)

        next_target_q_values = target_net(next_states_t)
        max_next_q_values = next_target_q_values.gather(1, next_actions).squeeze(1)
        max_next_q_values = torch.where(dones_t > 0.5, torch.zeros_like(max_next_q_values), max_next_q_values)
        targets = rewards_t + config.dqn.gamma * max_next_q_values

    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), 5.0)
    optimizer.step()
    online_net.eval()
    return float(loss.item())


def evaluate_against_agent(
    dqn_agent: DQNAgent,
    opponent_factory,
    *,
    games: int = 20,
) -> float:
    wins = 0
    for game_idx in range(games):
        controlled_player = 1 if game_idx % 2 == 0 else 2
        try:
            opponent = opponent_factory(game_idx)
        except TypeError:
            opponent = opponent_factory()
        result = play_dqn_match(dqn_agent, opponent, controlled_player=controlled_player)
        if result > 0:
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


def build_phase_opponent(
    *,
    config: Config,
    phase: CurriculumPhase,
    episode: int,
    rng: random.Random,
    opponent_pool: list[dict[str, torch.Tensor]],
) -> tuple[str, object]:
    kind = phase.opponent_kind or "random"
    if kind == "random":
        return (
            "random",
            CurriculumRandomAgent(
                seed=config.global_.seed + episode,
                block_vertical_bias=phase.block_vertical_bias,
            ),
        )
    if kind == "weak":
        return "weak", WeakHeuristicAgent(seed=config.global_.seed + episode)
    if kind == "strong":
        return "strong", StrongHeuristicAgent(seed=config.global_.seed + episode)
    if kind == "heuristic":
        return "strong", StrongHeuristicAgent(seed=config.global_.seed + episode)
    if kind == "self_play":
        opponent_net = build_network_from_state_dict(
            rng.choice(opponent_pool),
            device=config.resolve_device(),
            hidden_dim=config.dqn.hidden_dim,
            channel_sizes=config.dqn.channel_sizes,
            kernel_sizes=config.dqn.kernel_sizes,
            stride_sizes=config.dqn.stride_sizes,
            head_hidden_sizes=config.dqn.head_hidden_sizes,
        )
        return (
            "snapshot",
            DQNAgent(
                opponent_net,
                device=config.resolve_device(),
                epsilon=config.dqn.opponent_epsilon,
                seed=config.global_.seed + episode,
                name="snapshot",
            ),
        )
    raise ValueError(f"Unsupported DQN opponent kind '{kind}'")


def warmup_q_network(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: Config,
    device: torch.device,
    rng: random.Random,
    updates: int,
    metrics: DQNTrainingMetrics,
) -> int:
    update_steps = 0
    for _ in range(updates):
        loss = maybe_update_q_network(online_net, target_net, optimizer, replay, rng, config, device)
        if loss is None:
            continue
        metrics.losses.append(loss)
        update_steps += 1
        soft_update_target_network(online_net, target_net, tau=config.dqn.tau)
        if update_steps % config.dqn.target_update_freq == 0:
            target_net.load_state_dict(online_net.state_dict())
    return update_steps


def fill_replay_buffer_for_phase(
    replay: ReplayBuffer,
    online_net: nn.Module,
    config: Config,
    phase: CurriculumPhase,
    rng: random.Random,
    *,
    target_size: int,
) -> ReplayBuffer:
    warmup_opponent_kind = phase.warm_up_opponent or phase.opponent_kind or "random"
    while len(replay) < min(target_size, replay.capacity):
        episode_idx = len(replay) + 1
        _opponent_kind, opponent_agent = build_phase_opponent(
            config=config,
            phase=CurriculumPhase(
                name=f"{phase.name}_warmup",
                opponent_kind=warmup_opponent_kind,
                fraction=phase.fraction,
                rewards=phase.rewards,
                block_vertical_bias=phase.block_vertical_bias,
            ),
            episode=episode_idx,
            rng=rng,
            opponent_pool=[clone_state_dict(online_net)],
        )
        play_random_warmup_episode(replay, opponent_agent, phase.rewards, config, rng)
    return replay


def play_random_warmup_episode(
    replay: ReplayBuffer,
    opponent_agent,
    rewards: RewardProfile,
    config: Config,
    rng: random.Random,
) -> None:
    online_player = 1 if rng.random() < 0.5 else 2
    state = initial_state()
    pending: tuple[np.ndarray, int] | None = None

    while not is_terminal(state):
        if state.current_player == online_player:
            action = rng.choice(legal_actions(state))
            next_state = apply_action(state, action)
            state_array = state_to_numpy(state, online_player)
            if is_terminal(next_state):
                reward = shaped_reward(next_state, online_player, rewards, done=True)
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
            action = opponent_agent.select_action(state, legal_actions(state))
            next_state = apply_action(state, action)
            if pending is not None:
                done = is_terminal(next_state)
                reward = shaped_reward(next_state, online_player, rewards, done=done)
                next_state_array = np.zeros((2, 6, 7), dtype=np.float32) if done else state_to_numpy(next_state, online_player)
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


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
