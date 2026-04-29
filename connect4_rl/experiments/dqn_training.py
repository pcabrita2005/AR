from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from connect4_rl.agents.baselines import HeuristicAgent, RandomAgent
from connect4_rl.agents.learning.dqn import (
    ConnectFourQNetwork,
    DQNAgent,
    ReplayBuffer,
    clone_state_dict,
    epsilon_by_step,
    flip_action_horizontally,
    flip_action_mask_horizontally,
    flip_state_horizontally,
    legal_actions_to_mask,
    state_to_numpy,
)
from connect4_rl.config import Config, DQNConfig
from connect4_rl.utils.seed_utils import set_all_seeds
from connect4_rl.envs.connect_four import (
    ConnectFourState,
    apply_action,
    initial_state,
    is_terminal,
    legal_actions,
    outcome_for_player,
)


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
    best_checkpoint_path: str = ""
    best_score: float = float("-inf")


def train_dqn_self_play(
    config: Config | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
) -> tuple[DQNAgent, DQNTrainingMetrics]:
    from connect4_rl.config import get_default_config
    config = config or get_default_config()
    set_all_seeds(config.global_.seed)
    rng = random.Random(config.global_.seed)

    device = torch.device(config.resolve_device())
    online_net = ConnectFourQNetwork(hidden_dim=config.dqn.hidden_dim).to(device)
    target_net = ConnectFourQNetwork(hidden_dim=config.dqn.hidden_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.AdamW(online_net.parameters(), lr=config.dqn.learning_rate)
    replay = ReplayBuffer(config.dqn.replay_buffer_size)

    opponent_pool = [clone_state_dict(online_net)]
    update_steps = 0
    policy_steps = 0
    metrics = DQNTrainingMetrics(config=asdict(config.dqn))
    best_state_dict = clone_state_dict(online_net)
    previous_eval_state_dict: dict[str, torch.Tensor] | None = None

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    for episode in range(1, config.dqn.episodes + 1):
        opponent_kind, opponent_agent = build_training_opponent(
            config=config,
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
                    reward = outcome_for_player(state_after_online, online_player)
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
                    pending = None  # Ensure pending is cleared when episode terminates
                    loss = maybe_update_q_network(
                        online_net,
                        target_net,
                        optimizer,
                        replay,
                        rng,
                        config,
                        device,
                    )
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
                    reward = outcome_for_player(next_state, online_player) if is_terminal(next_state) else 0.0
                    next_state_array = (
                        np.zeros((2, 6, 7), dtype=np.float32)
                        if is_terminal(next_state)
                        else state_to_numpy(next_state, online_player)
                    )
                    next_mask = (
                        np.zeros(7, dtype=np.float32)
                        if is_terminal(next_state)
                        else legal_actions_to_mask(legal_actions(next_state))
                    )
                    add_transition(
                        replay=replay,
                        config=config,
                        state=pending[0],
                        action=pending[1],
                        reward=reward,
                        next_state=next_state_array,
                        done=is_terminal(next_state),
                        next_action_mask=next_mask,
                    )
                    episode_reward = reward if is_terminal(next_state) else episode_reward
                    pending = None

                    for _ in range(config.dqn.gradient_updates_per_step):
                        loss = maybe_update_q_network(
                            online_net,
                            target_net,
                            optimizer,
                            replay,
                            rng,
                            config,
                            device,
                        )
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

        if episode % config.dqn.opponent_refresh_interval == 0:
            opponent_pool.append(clone_state_dict(online_net))
            opponent_pool = opponent_pool[-config.dqn.opponent_pool_size :]

        if checkpoint_path is not None and (episode % config.dqn.eval_interval == 0 or episode == config.dqn.max_episodes):
            torch.save(online_net.state_dict(), checkpoint_path / f"dqn_episode_{episode:04d}.pt")

        if episode % config.dqn.eval_interval == 0 or episode == config.dqn.max_episodes:
            eval_agent = DQNAgent(online_net, device=config.resolve_device(), epsilon=0.0, seed=config.global_.seed)
            random_wr = evaluate_against_agent(
                eval_agent,
                lambda game_idx: RandomAgent(seed=config.global_.seed + 10_000 + game_idx),
                games=config.dqn.eval_games,
            )
            heuristic_wr = evaluate_against_agent(
                eval_agent,
                lambda game_idx: HeuristicAgent(seed=config.global_.seed + 20_000 + game_idx),
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
                            hidden_dim=config.dqn.fc_hidden,
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
                    "vs_heuristic_win_rate": heuristic_wr,
                    "vs_previous_win_rate": previous_wr,
                }
            )
            score = heuristic_wr * config.dqn.checkpoint_score_heuristic_weight + random_wr
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


def build_network(config: Config) -> nn.Module:
    from connect4_rl.agents.learning.dqn import ConnectFourQNetwork

    return ConnectFourQNetwork(hidden_dim=config.dqn.fc_hidden)


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


def build_training_opponent(
    *,
    config: Config,
    episode: int,
    rng: random.Random,
    opponent_pool: list[dict[str, torch.Tensor]],
) -> tuple[str, object]:
    warmup = config.dqn.warmup_episodes
    if episode <= warmup:
        if episode % 2 == 0:
            return "random", RandomAgent(seed=config.global_.seed + episode)
        return "heuristic", HeuristicAgent(seed=config.global_.seed + episode)

    draw = rng.random()
    random_frac = config.dqn.random_opponent_fraction
    heuristic_frac = config.dqn.heuristic_opponent_fraction
    if draw < random_frac:
        return "random", RandomAgent(seed=config.global_.seed + episode)
    if draw < random_frac + heuristic_frac:
        return "heuristic", HeuristicAgent(seed=config.global_.seed + episode)

    opponent_net = build_network_from_state_dict(
        rng.choice(opponent_pool),
        device=config.resolve_device(),
        hidden_dim=config.dqn.fc_hidden,
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
