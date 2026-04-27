from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from connect4_rl.agents.baselines import HeuristicAgent, RandomAgent
from connect4_rl.agents.baselines.heuristic_agent import score_position
from connect4_rl.agents.learning.ppo import ConnectFourActorCritic, PPOAgent, PPOConfig
from connect4_rl.envs.connect_four import apply_action, encode_state, initial_state, is_terminal, legal_actions, outcome_for_player


@dataclass
class PPOMetrics:
    config: dict[str, object]
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    evaluation: list[dict[str, float]] = field(default_factory=list)
    opponent_kinds: list[str] = field(default_factory=list)
    best_checkpoint_path: str = ""
    best_score: float = float("-inf")


def train_ppo_self_play(
    config: PPOConfig | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
) -> tuple[PPOAgent, PPOMetrics]:
    config = config or PPOConfig()
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    network = ConnectFourActorCritic(hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate)

    metrics = PPOMetrics(config=asdict(config))
    best_state_dict = clone_state_dict(network)
    previous_eval_state_dict: dict[str, torch.Tensor] | None = None
    rollout_buffer: list[dict[str, object]] = []

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    for episode in range(1, config.episodes + 1):
        opponent_kind = build_training_mode(config, episode, rng)

        if opponent_kind == "self_play":
            trajectories, episode_reward, episode_steps = collect_self_play_episode(network, device, config)
            for trajectory in trajectories:
                if trajectory:
                    rollout_buffer.extend(
                        augment_trajectory(trajectory) if config.use_horizontal_symmetry_augmentation else trajectory
                    )
        else:
            opponent_agent = build_fixed_opponent(opponent_kind, config.seed + episode)
            controlled_player = 1 if rng.random() < 0.5 else 2
            trajectory, episode_reward, episode_steps = collect_policy_episode_against_opponent(
                network,
                device,
                opponent_agent,
                controlled_player,
                config,
            )
            if trajectory:
                rollout_buffer.extend(augment_trajectory(trajectory) if config.use_horizontal_symmetry_augmentation else trajectory)

        if rollout_buffer and (episode % config.rollout_episodes_per_update == 0 or episode == config.episodes):
            maybe_anneal_learning_rate(optimizer, config, episode)
            policy_loss, value_loss, entropy = update_ppo(network, optimizer, rollout_buffer, config, device)
            metrics.policy_losses.append(policy_loss)
            metrics.value_losses.append(value_loss)
            metrics.entropies.append(entropy)
            rollout_buffer = []

        metrics.episode_rewards.append(episode_reward)
        metrics.episode_lengths.append(episode_steps)
        metrics.opponent_kinds.append(opponent_kind)

        if checkpoint_path is not None and (episode % config.eval_interval == 0 or episode == config.episodes):
            torch.save(network.state_dict(), checkpoint_path / f"ppo_episode_{episode:04d}.pt")

        if episode % config.eval_interval == 0 or episode == config.episodes:
            eval_agent = PPOAgent(network, device=config.device, sample_actions=False, seed=config.seed)
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
                    lambda game_idx: PPOAgent(
                        _load_previous_ppo_network(previous_eval_state_dict, config.hidden_dim),
                        device=config.device,
                        sample_actions=False,
                        seed=config.seed + 30_000 + game_idx,
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
            score = heuristic_wr * config.checkpoint_score_heuristic_weight + random_wr
            if score >= metrics.best_score:
                metrics.best_score = score
                best_state_dict = clone_state_dict(network)
                if checkpoint_path is not None:
                    best_path = checkpoint_path / "ppo_best.pt"
                    torch.save(best_state_dict, best_path)
                    metrics.best_checkpoint_path = str(best_path)
            if checkpoint_path is not None:
                write_metrics_snapshot(metrics, checkpoint_path / "metrics_latest.json")
            previous_eval_state_dict = clone_state_dict(network)

    network.load_state_dict(best_state_dict)
    final_agent = PPOAgent(network, device=config.device, sample_actions=False, seed=config.seed)
    if checkpoint_path is not None:
        write_metrics_snapshot(metrics, checkpoint_path / "metrics_final.json")
    return final_agent, metrics


def update_ppo(
    network: ConnectFourActorCritic,
    optimizer: torch.optim.Optimizer,
    trajectory: list[dict[str, object]],
    config: PPOConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    states = torch.tensor(np.stack([step["state"] for step in trajectory]), dtype=torch.float32, device=device)
    actions = torch.tensor([step["action"] for step in trajectory], dtype=torch.int64, device=device)
    action_masks = torch.tensor(np.stack([step["action_mask"] for step in trajectory]), dtype=torch.bool, device=device)
    old_log_probs = torch.tensor([step["log_prob"] for step in trajectory], dtype=torch.float32, device=device)
    values = torch.tensor([step["value"] for step in trajectory], dtype=torch.float32, device=device)
    rewards = torch.tensor([step["reward"] for step in trajectory], dtype=torch.float32, device=device)
    dones = torch.tensor([step["done"] for step in trajectory], dtype=torch.float32, device=device)

    returns, advantages = compute_gae(rewards, values, dones, config.gamma, config.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []

    network.train()
    batch_size = states.shape[0]
    for _ in range(config.update_epochs):
        indices = torch.randperm(batch_size, device=device)
        for start in range(0, batch_size, config.minibatch_size):
            end = min(start + config.minibatch_size, batch_size)
            batch_indices = indices[start:end]

            logits, predicted_values = network(states[batch_indices])
            masked_logits = logits.masked_fill(~action_masks[batch_indices], -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(actions[batch_indices])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
            unclipped = ratio * advantages[batch_indices]
            clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages[batch_indices]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.smooth_l1_loss(predicted_values, returns[batch_indices])
            loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

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
    config: PPOConfig,
    episode: int,
) -> None:
    if not config.anneal_learning_rate:
        return
    progress = max(0.0, 1.0 - ((episode - 1) / max(config.episodes, 1)))
    current_lr = config.learning_rate * progress
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr


def collect_self_play_episode(
    network: ConnectFourActorCritic,
    device: torch.device,
    config: PPOConfig,
) -> tuple[list[list[dict[str, object]]], float, int]:
    trajectories: dict[int, list[dict[str, object]]] = {1: [], 2: []}
    state = initial_state()
    episode_steps = 0

    while not is_terminal(state):
        player = state.current_player
        action, log_prob, value, entropy, action_mask = sample_policy_action(network, state, player, device)
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
    config: PPOConfig,
) -> tuple[list[dict[str, object]], float, int]:
    trajectory: list[dict[str, object]] = []
    state = initial_state()
    episode_reward = 0.0
    episode_steps = 0

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
        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[legal] = logits[legal]
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = int(dist.sample().item())
        log_prob = float(dist.log_prob(torch.tensor(action, device=device)).item())
        entropy = float(dist.entropy().item())
    return action, log_prob, float(value.item()), entropy, legal_actions_to_mask(legal)


def compute_step_reward(
    state,
    next_state,
    player: int,
    config: PPOConfig,
) -> float:
    if is_terminal(next_state):
        return outcome_for_player(next_state, player)
    if not config.use_reward_shaping:
        return 0.0
    previous_score = score_position(state, player)
    next_score = score_position(next_state, player)
    delta = float(next_score - previous_score)
    return float(config.reward_shaping_scale * np.tanh(delta / 100.0))


def build_training_mode(
    config: PPOConfig,
    episode: int,
    rng: random.Random,
) -> str:
    if episode <= config.warmup_episodes:
        if episode <= max(1, (config.warmup_episodes * 3) // 4):
            return "random"
        return "heuristic"

    draw = rng.random()
    if draw < config.random_opponent_fraction:
        return "random"
    if draw < config.random_opponent_fraction + config.heuristic_opponent_fraction:
        return "heuristic"
    return "self_play"


def build_fixed_opponent(kind: str, seed: int):
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(seed=seed)
    raise ValueError(f"Unsupported fixed opponent kind: {kind}")


def evaluate_against_agent(agent: PPOAgent, opponent_factory, *, games: int = 20) -> float:
    wins = 0
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
    return wins / games


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _load_previous_ppo_network(
    state_dict: dict[str, torch.Tensor],
    hidden_dim: int,
) -> ConnectFourActorCritic:
    network = ConnectFourActorCritic(hidden_dim=hidden_dim)
    network.load_state_dict(state_dict)
    return network


def write_metrics_snapshot(metrics: PPOMetrics, path: Path) -> None:
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
