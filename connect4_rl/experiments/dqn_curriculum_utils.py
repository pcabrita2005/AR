from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from connect4_rl.envs.connect_four import COLUMNS, ROWS, ConnectFourState, outcome_for_player


@dataclass(frozen=True)
class RewardProfile:
    win: float = 1.0
    vertical_win: float = 1.0
    three_in_row: float = 0.0
    opp_three_in_row: float = 0.0
    lose: float = -1.0
    play_continues: float = 0.0


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    opponent_kind: str | None
    fraction: float
    rewards: RewardProfile
    opponent_weights: dict[str, float] | None = None
    buffer_warm_up: bool = False
    warm_up_opponent: str | None = None
    warmup_replay_fill: int = 0
    agent_warmup_updates: int = 0
    block_vertical_bias: float = 1.0
    opponent_pool_size: int | None = None
    opponent_upgrade_interval: int | None = None


@dataclass(frozen=True)
class CurriculumDefinition:
    name: str
    description: str
    phases: tuple[CurriculumPhase, ...]


def normalize_dqn_opponent_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    normalized = kind.strip().lower()
    aliases = {
        "self": "self_play",
        "self-play": "self_play",
        "snapshot": "self_play",
    }
    return aliases.get(normalized, normalized)


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
) -> tuple[list[CurriculumPhase], list[dict[str, object]]]:
    schedule: list[CurriculumPhase] = []
    summary: list[dict[str, object]] = []
    cursor = 1
    rng = random.Random(seed)
    for phase, count in allocate_phase_episodes(total_episodes, definition.phases):
        start_episode = cursor
        end_episode = cursor + count - 1
        if phase.opponent_weights:
            phase_schedule = _sample_phase_schedule(phase, count, rng)
            opponent_kind = "mixed"
            realized_counts = dict(Counter(item.opponent_kind for item in phase_schedule))
        else:
            phase_schedule = [phase] * count
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


def _sample_phase_schedule(phase: CurriculumPhase, count: int, rng: random.Random) -> list[CurriculumPhase]:
    assert phase.opponent_weights is not None
    kinds = list(phase.opponent_weights.keys())
    weights = [float(phase.opponent_weights[kind]) for kind in kinds]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("opponent_weights must sum to a positive value")
    normalized = [weight / total_weight for weight in weights]
    return [
        CurriculumPhase(
            name=phase.name,
            opponent_kind=normalize_dqn_opponent_kind(kind),
            fraction=phase.fraction,
            rewards=phase.rewards,
            block_vertical_bias=phase.block_vertical_bias,
            opponent_pool_size=phase.opponent_pool_size,
            opponent_upgrade_interval=phase.opponent_upgrade_interval,
        )
        for kind in rng.choices(kinds, weights=normalized, k=count)
    ]


def shaped_reward(
    state: ConnectFourState,
    perspective_player: int,
    rewards: RewardProfile,
    *,
    done: bool,
) -> float:
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


def check_vertical_win(state: ConnectFourState, player: int) -> bool:
    board = state.board
    for col in range(COLUMNS):
        for row in range(ROWS - 3):
            if all(board[row + offset][col] == player for offset in range(4)):
                return True
    return False


def count_winnable_windows(state: ConnectFourState, player: int) -> int:
    board = state.board
    windows = []
    for row in range(ROWS):
        for col in range(COLUMNS - 3):
            windows.append([board[row][col + offset] for offset in range(4)])
    for row in range(ROWS - 3):
        for col in range(COLUMNS):
            windows.append([board[row + offset][col] for offset in range(4)])
    for row in range(ROWS - 3):
        for col in range(COLUMNS - 3):
            windows.append([board[row + offset][col + offset] for offset in range(4)])
    for row in range(3, ROWS):
        for col in range(COLUMNS - 3):
            windows.append([board[row - offset][col + offset] for offset in range(4)])
    return sum(1 for window in windows if window.count(player) == 3 and window.count(0) == 1)
