from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import time

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
from connect4_rl.envs.connect_four import ConnectFourState, apply_action, initial_state, is_terminal, legal_actions, outcome_for_player


@dataclass
class AlphaZeroMetrics:
    config: dict[str, object]
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    evaluation: list[dict[str, float]] = field(default_factory=list)
    tactical_accuracy: list[dict[str, float]] = field(default_factory=list)
    best_checkpoint_path: str = ""
    best_score: float = float("-inf")


def train_alphazero_self_play(
    config: AlphaZeroConfig | None = None,
    *,
    checkpoint_dir: str | Path | None = None,
) -> tuple[AlphaZeroAgent, AlphaZeroMetrics]:
    config = config or AlphaZeroConfig()
    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Resolve device: if config.device is "auto", resolve to cuda or cpu; otherwise use as-is.
    if config.device == "auto":
        resolved_device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device_str = config.device
    device = torch.device(resolved_device_str)
    if config.device != resolved_device_str:
        print(f"Resolved device from '{config.device}' to '{resolved_device_str}'")
    print(f"Using device: {device} | CUDA available: {torch.cuda.is_available()}")
    network = ConnectFourPolicyValueNet(
        n_filters=config.n_filters,
        n_res_blocks=config.n_res_blocks,
    ).to(device)
    print(f"Network moved to {device}")
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    replay_buffer: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=config.replay_capacity)

    metrics = AlphaZeroMetrics(config=asdict(config))
    best_state_dict = clone_state_dict(network)
    previous_eval_state_dict: dict[str, torch.Tensor] | None = None

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    eval_simulations = config.eval_mcts_simulations or config.mcts_simulations
    # episodes_per_batch controls how many self-play episodes are grouped
    # together before performing network updates. This replaces the old
    # `num_workers` semantic which attempted parallelism.
    episodes_per_batch = max(1, int(getattr(config, "episodes_per_batch", 1)))
    episode = 1
    # lightweight timing
    training_start_time = time.time()
    print(f"Training with episode batch size {episodes_per_batch} for self-play | Main network on {device}")
    print(f"Will start training after episode {config.replay_warmup_games} (when buffer has at least {min(config.batch_size, 8)} examples)")

    while episode <= config.episodes:
        batch_end = min(config.episodes, episode + episodes_per_batch - 1)
        batch_episodes = list(range(episode, batch_end + 1))
        print(f"[LOOP] Starting batch: episodes {batch_episodes[0]}-{batch_end}", flush=True)
        print(f"[CLONE-PRE] About to clone state_dict", flush=True)
        batch_snapshot = clone_state_dict(network)
        print(f"[CLONE-POST] State dict cloned successfully", flush=True)
        config_snapshot = asdict(config)
        batch_results: list[tuple[int, tuple[list[tuple[np.ndarray, np.ndarray, float]], float, int]]] = []

        # TEMPORARY FIX: Force serial execution to debug threading deadlock
        print(f"[BATCH] Processing episodes {batch_episodes[0]}-{batch_episodes[-1]} (SERIAL MODE due to deadlock)", flush=True)
        for batch_episode in batch_episodes:
            print(f"[SERIAL] Episode {batch_episode}: generating self-play", flush=True)
            result = _generate_self_play_episode_from_state_dict(
                batch_snapshot,
                config_snapshot,
                batch_episode,
                get_training_mcts_simulations(config, batch_episode),
                "cpu",
            )
            batch_results.append((batch_episode, result))
            print(f"[SERIAL] Episode {batch_episode}: done", flush=True)

        print(f"[RESULTS] Got {len(batch_results)} episode results, extending replay buffer", flush=True)
        for idx, (_, (examples, final_reward, episode_steps)) in enumerate(batch_results):
            print(f"[BUFFER] Episode {_}: adding {len(examples)} examples", flush=True)
            replay_buffer.extend(examples)
            metrics.episode_rewards.append(final_reward)
            metrics.episode_lengths.append(episode_steps)
        print(f"[BUFFER] Replay buffer now has {len(replay_buffer)} examples", flush=True)

        if batch_episodes and batch_episodes[-1] >= config.replay_warmup_games and len(replay_buffer) >= min(config.batch_size, 8):
            print(f"[UPDATE] Starting training: {config.updates_per_episode} updates x {len(batch_episodes)} episodes", flush=True)
            maybe_anneal_learning_rate(optimizer, config, batch_end)
            batch_policy_losses: list[float] = []
            batch_value_losses: list[float] = []
            total_updates = config.updates_per_episode * len(batch_episodes)
            for update_idx in range(total_updates):
                print(f"[UPDATE] Step {update_idx + 1}/{total_updates}", flush=True)
                step_policy_loss, step_value_loss = update_policy_value_network(
                    network,
                    optimizer,
                    replay_buffer,
                    config,
                    device,
                    rng,
                )
                batch_policy_losses.append(step_policy_loss)
                batch_value_losses.append(step_value_loss)
            print(f"[UPDATE] Finished all {total_updates} updates", flush=True)
            metrics.policy_losses.append(float(np.mean(batch_policy_losses)))
            metrics.value_losses.append(float(np.mean(batch_value_losses)))

        if checkpoint_path is not None and (batch_end % config.eval_interval == 0 or batch_end == config.episodes):
            torch.save(network.state_dict(), checkpoint_path / f"alphazero_episode_{batch_end:04d}.pt")

        if batch_end % config.eval_interval == 0 or batch_end == config.episodes:
            eval_state_dict = clone_state_dict(network)
            random_wr = evaluate_against_agent(
                lambda game_idx, state_dict=eval_state_dict: _build_eval_agent_from_state_dict(
                    state_dict,
                    config,
                    simulations=eval_simulations,
                    seed=config.seed + 10_000 + game_idx,
                    temperature=0.0,
                    name="alphazero_eval",
                    device="cpu",
                ),
                lambda game_idx: RandomAgent(seed=config.seed + 20_000 + game_idx),
                games=config.eval_games,
            )
            heuristic_wr = evaluate_against_agent(
                lambda game_idx, state_dict=eval_state_dict: _build_eval_agent_from_state_dict(
                    state_dict,
                    config,
                    simulations=eval_simulations,
                    seed=config.seed + 30_000 + game_idx,
                    temperature=0.0,
                    name="alphazero_eval",
                    device="cpu",
                ),
                lambda game_idx: HeuristicAgent(seed=config.seed + 40_000 + game_idx),
                games=config.eval_games,
            )
            previous_wr = 0.0
            if previous_eval_state_dict is not None:
                previous_wr = evaluate_against_agent(
                    lambda game_idx, state_dict=eval_state_dict: _build_eval_agent_from_state_dict(
                        state_dict,
                        config,
                        simulations=eval_simulations,
                        seed=config.seed + 50_000 + game_idx,
                        temperature=0.0,
                        name="alphazero_eval",
                        device="cpu",
                    ),
                    lambda game_idx, state_dict=previous_eval_state_dict: _build_eval_agent_from_state_dict(
                        state_dict,
                        config,
                        simulations=eval_simulations,
                        seed=config.seed + 60_000 + game_idx,
                        temperature=0.0,
                        name="previous_snapshot",
                        device="cpu",
                    ),
                    games=config.eval_games,
                )
            metrics.evaluation.append(
                {
                    "episode": float(batch_end),
                    "vs_random_win_rate": random_wr,
                    "vs_heuristic_win_rate": heuristic_wr,
                    "vs_previous_win_rate": previous_wr,
                }
            )
            metrics.tactical_accuracy.append(
                {
                    "episode": float(batch_end),
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

        # lightweight heartbeat every 10 episodes, with a fuller timing summary every 100.
        if batch_end % 10 == 0 or batch_end == config.episodes:
            now = time.time()
            elapsed = now - training_start_time
            episodes_done = batch_end
            eps_per_min = episodes_done / (elapsed / 60.0) if elapsed > 0 else float("inf")
            print(f"[PROGRESS] episode={episodes_done}/{config.episodes} | elapsed={elapsed:.1f}s | {eps_per_min:.2f} ep/min")

        if batch_end % 100 == 0 or batch_end == config.episodes:
            now = time.time()
            elapsed = now - training_start_time
            episodes_done = batch_end
            eps_per_min = episodes_done / (elapsed / 60.0) if elapsed > 0 else float("inf")
            avg_s_per_ep = elapsed / episodes_done if episodes_done > 0 else 0.0
            print(f"[TIMING] episode={episodes_done}/{config.episodes} | elapsed={elapsed:.1f}s | avg={avg_s_per_ep:.2f}s/ep | {eps_per_min:.2f} ep/min | GPU mem={torch.cuda.memory_allocated(device) / 1e9:.2f}GB")

        print(f"[LOOP-END] Batch {batch_end} finished, moving to next batch", flush=True)
        episode = batch_end + 1

    network.load_state_dict(best_state_dict)
    final_agent = AlphaZeroAgent(
        network,
        simulations=eval_simulations,
        c_puct=config.c_puct,
        device=str(device),
        seed=config.seed,
        temperature=0.0,
    )
    if checkpoint_path is not None:
        write_metrics_snapshot(metrics, checkpoint_path / "metrics_final.json")
    return final_agent, metrics


def generate_self_play_episode(
    network: ConnectFourPolicyValueNet,
    config: AlphaZeroConfig,
    rng: random.Random,
    *,
    simulations: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, int]:
    examples: list[tuple[np.ndarray, np.ndarray, int]] = []
    state = initial_state()
    episode_steps = 0

    while not is_terminal(state):
        root_noise = config.root_noise_each_move or episode_steps == 0
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
        temperature = config.temperature if episode_steps < config.temperature_drop_move else 0.0
        action = sample_action_from_policy(visit_policy, legal_actions(state), temperature=temperature, rng=rng)
        examples.append((encode_alphazero_state(state, state.current_player), visit_policy.copy(), state.current_player))
        state = apply_action(state, action)
        episode_steps += 1

    final_examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    for encoded_state, policy_target, player in examples:
        final_value = outcome_for_player(state, player)
        final_examples.append((encoded_state, policy_target, final_value))
        if config.use_horizontal_symmetry_augmentation:
            final_examples.append((np.flip(encoded_state, axis=2).copy(), np.flip(policy_target, axis=0).copy(), final_value))

    return final_examples, outcome_for_player(state, 1), episode_steps

def update_policy_value_network(
    network: ConnectFourPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: deque[tuple[np.ndarray, np.ndarray, float]],
    config: AlphaZeroConfig,
    device: torch.device,
    rng: random.Random,
) -> tuple[float, float]:
    batch_size = min(config.batch_size, len(replay_buffer))
    policy_losses: list[float] = []
    value_losses: list[float] = []
    network.train()

    for _ in range(config.update_epochs):
        batch = rng.sample(replay_buffer, batch_size)
        states, target_policies, target_values = zip(*batch)
        states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
        target_policies_t = torch.tensor(np.stack(target_policies), dtype=torch.float32, device=device)
        target_values_t = torch.tensor(np.array(target_values, dtype=np.float32), dtype=torch.float32, device=device)

        logits, values = network(states_t)
        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -(target_policies_t * log_probs).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(values, target_values_t)
        loss = policy_loss + config.value_loss_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.max_grad_norm)
        optimizer.step()

        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))

    network.eval()
    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def get_training_mcts_simulations(config: AlphaZeroConfig, episode: int) -> int:
    if config.mcts_start_search_iter is None:
        return config.mcts_simulations

    simulations = config.mcts_start_search_iter + max(episode - 1, 0) * config.mcts_search_increment
    if config.mcts_max_search_iter is not None:
        simulations = min(simulations, config.mcts_max_search_iter)
    return max(1, simulations)


def maybe_anneal_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: AlphaZeroConfig,
    episode: int,
) -> None:
    if not config.anneal_learning_rate:
        return
    progress = max(0.0, 1.0 - ((episode - 1) / max(config.episodes, 1)))
    current_lr = config.learning_rate * progress
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr


def evaluate_against_agent(agent_factory, opponent_factory, *, games: int = 20, num_workers: int = 1) -> float:
    """Evaluate agent against opponent in serial mode (no threading due to PyTorch thread-safety issues)."""
    if games <= 0:
        return 0.0

    def play_single_game(game_idx: int) -> int:
        controlled_player = 1 if game_idx % 2 == 0 else 2
        agent = agent_factory(game_idx)
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
        return 1 if state.winner == controlled_player else 0

    # FORCED SERIAL: PyTorch is not thread-safe for concurrent forward passes
    # ThreadPoolExecutor was causing 120s+ hangs in WSL2/Jupyter environment
    wins = [play_single_game(game_idx) for game_idx in range(games)]
    return float(sum(wins) / games)


def _build_eval_network_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    config: AlphaZeroConfig,
    device: str = "cpu",
) -> ConnectFourPolicyValueNet:
    network = ConnectFourPolicyValueNet(n_filters=config.n_filters, n_res_blocks=config.n_res_blocks).to(device)
    network.load_state_dict(state_dict)
    network.eval()
    return network


def _build_eval_agent_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: AlphaZeroConfig,
    *,
    simulations: int,
    seed: int,
    temperature: float,
    name: str,
    device: str = "auto",
) -> AlphaZeroAgent:
    """Build evaluation agent from state dict, optionally respecting a specific device."""
    if device == "auto":
        device = config.device
    network = _build_eval_network_from_state_dict(state_dict, config=config, device=device)
    return AlphaZeroAgent(
        network,
        simulations=simulations,
        c_puct=config.c_puct,
        device=device,
        seed=seed,
        temperature=temperature,
        name=name,
    )


def _generate_self_play_episode_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    config_snapshot: dict[str, object],
    episode: int,
    simulations: int,
    device: str = "auto",
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, int]:
    """Generate a self-play episode from a network state dict snapshot.
    
    If device='auto' or the config has device='auto', will resolve to cuda/cpu based on availability.
    Otherwise uses config's device (respecting user's choice).
    """
    worker_config = AlphaZeroConfig(**config_snapshot)
    # Resolve device for workers - keep MCTS on GPU unless explicitly CPU
    if device == "auto":
        device = worker_config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    worker_config.device = device
    worker_config.num_workers = 1  # Nested workers don't make sense
    worker_network = _build_eval_network_from_state_dict(state_dict, config=worker_config, device=device)
    worker_rng = random.Random(worker_config.seed + episode)
    return generate_self_play_episode(worker_network, worker_config, worker_rng, simulations=simulations)


def write_metrics_snapshot(metrics: AlphaZeroMetrics, path: Path) -> None:
    path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")


def _load_previous_alphazero_network(
    state_dict: dict[str, torch.Tensor],
    *,
    n_filters: int,
    n_res_blocks: int,
) -> ConnectFourPolicyValueNet:
    network = ConnectFourPolicyValueNet(n_filters=n_filters, n_res_blocks=n_res_blocks)
    network.load_state_dict(state_dict)
    return network


def evaluate_tactical_accuracy(
    network: ConnectFourPolicyValueNet,
    config: AlphaZeroConfig,
    *,
    examples_per_type: int,
) -> float:
    states: list[np.ndarray] = []
    targets: list[int] = []

    for condition in ("win", "block"):
        generated = 0
        seed_offset = 1_000 if condition == "win" else 2_000
        rng = random.Random(config.seed + seed_offset)
        while generated < examples_per_type:
            state = initial_state()
            while not is_terminal(state):
                legal = legal_actions(state)
                tactical_action = find_tactical_action(state, legal, condition)
                if tactical_action is not None:
                    states.append(encode_alphazero_state(state, state.current_player))
                    targets.append(tactical_action)
                    generated += 1
                    break
                action = rng.choice(legal)
                state = apply_action(state, action)

    with torch.no_grad():
        network.eval()
        device = next(network.parameters()).device
        x_target = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
        logits, _values = network(x_target)
        predictions = logits.argmax(dim=1).detach().cpu().numpy()
    return float(np.mean(predictions == np.asarray(targets)))


def find_tactical_action(state: ConnectFourState, legal: list[int], condition: str) -> int | None:
    current_player = state.current_player
    opponent = 2 if current_player == 1 else 1

    if condition == "win":
        for action in legal:
            next_state = apply_action(state, action)
            if next_state.winner == current_player:
                return action
        return None

    opponent_turn_state = ConnectFourState(
        board=state.board,
        current_player=opponent,
        winner=state.winner,
        moves_played=state.moves_played,
        last_action=state.last_action,
    )
    for action in legal:
        reply_state = apply_action(opponent_turn_state, action)
        if reply_state.winner == opponent:
            return action
    return None
