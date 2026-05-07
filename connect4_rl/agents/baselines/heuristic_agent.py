from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from connect4_rl.envs.connect_four import COLUMNS, ConnectFourState, ROWS, apply_action


def score_position(state: ConnectFourState, player: int) -> int:
    opponent = 2 if player == 1 else 1
    board = state.board
    score = 0
    center_column = COLUMNS // 2
    center_count = sum(1 for row in range(ROWS) if board[row][center_column] == player)
    score += center_count * 6
    for window in all_windows(board):
        score += score_window(window, player, opponent)
    return score


def all_windows(board: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    windows: list[list[int]] = []
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
    return windows


def score_window(window: list[int], player: int, opponent: int) -> int:
    player_count = window.count(player)
    opponent_count = window.count(opponent)
    empty_count = window.count(0)
    if player_count == 4:
        return 10_000
    if player_count == 3 and empty_count == 1:
        return 80
    if player_count == 2 and empty_count == 2:
        return 12
    if player_count == 1 and empty_count == 3:
        return 2
    if opponent_count == 3 and empty_count == 1:
        return -100
    if opponent_count == 2 and empty_count == 2:
        return -10
    return 0


class BaseHeuristicAgent:
    def __init__(self, seed: int | None = None, name: str = "heuristic") -> None:
        self.name = name
        self._rng = random.Random(seed)
        self.num_cols = COLUMNS
        self.num_rows = ROWS
        self.length = 4

    def select_action(self, state: ConnectFourState, legal_actions_now: Sequence[int]) -> int:
        raise NotImplementedError

    def _update_top(self, state: ConnectFourState) -> np.ndarray:
        board = np.array(state.board, dtype=int)
        top = np.full(self.num_cols, self.num_rows, dtype=int)
        for col in range(self.num_cols):
            for row in range(self.num_rows - 1, -1, -1):
                if board[row, col] == 0:
                    top[col] = row
                    break
        return top

    def _simulate_player_move(self, state: ConnectFourState, action: int, player: int) -> ConnectFourState:
        if state.current_player == player:
            return apply_action(state, action)
        swapped_state = ConnectFourState(
            board=state.board,
            current_player=player,
            winner=state.winner,
            moves_played=state.moves_played,
            last_action=state.last_action,
        )
        return apply_action(swapped_state, action)

    def _direction_lengths(self, board: np.ndarray, row: int, col: int, piece: int) -> np.ndarray:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        lengths = []
        for d_row, d_col in directions:
            count = 1
            count += self._count_direction(board, row, col, d_row, d_col, piece)
            count += self._count_direction(board, row, col, -d_row, -d_col, piece)
            lengths.append(count)
        return np.asarray(lengths, dtype=int)

    def _count_direction(self, board: np.ndarray, row: int, col: int, d_row: int, d_col: int, piece: int) -> int:
        count = 0
        r = row + d_row
        c = col + d_col
        while 0 <= r < self.num_rows and 0 <= c < self.num_cols and board[r, c] == piece:
            count += 1
            r += d_row
            c += d_col
        return count

    def _outcome(
        self,
        state: ConnectFourState,
        action: int,
        player: int,
        *,
        return_length: bool = False,
    ) -> tuple[bool, float | None, bool, np.ndarray | None]:
        top = self._update_top(state)
        row = int(top[action])
        if row >= self.num_rows:
            return (False, None, None, None) if return_length else (False, None, None, None)

        next_state = self._simulate_player_move(state, action, player)
        board = np.array(next_state.board, dtype=int)
        lengths = self._direction_lengths(board, row, action, player)
        draw = next_state.winner == 0 and next_state.moves_played == self.num_rows * self.num_cols
        ended = next_state.winner == player or draw
        reward = 1.0 if next_state.winner == player else 0.0

        if return_length:
            return True, reward, ended, lengths
        return True, reward, ended, None


class WeakHeuristicAgent(BaseHeuristicAgent):
    def __init__(self, seed: int | None = None, name: str = "weak_heuristic") -> None:
        super().__init__(seed=seed, name=name)

    def select_action(self, state: ConnectFourState, legal_actions_now: Sequence[int]) -> int:
        return self._pick_weak_action(state, legal_actions_now)

    def _pick_weak_action(self, state: ConnectFourState, legal_actions_now: Sequence[int]) -> int:
        candidate_actions = list(legal_actions_now)
        max_length = -1
        best_actions: list[int] = []
        player = state.current_player

        for action in candidate_actions:
            possible, _reward, _ended, lengths = self._outcome(state, action, player, return_length=True)
            if not possible or lengths is None:
                continue
            total_length = int(lengths.sum())
            if total_length > max_length:
                max_length = total_length
                best_actions = [action]
            elif total_length == max_length:
                best_actions.append(action)

        if not best_actions:
            return self._rng.choice(candidate_actions)
        return self._rng.choice(best_actions)


class StrongHeuristicAgent(BaseHeuristicAgent):
    def __init__(self, seed: int | None = None, name: str = "heuristic") -> None:
        super().__init__(seed=seed, name=name)

    def select_action(self, state: ConnectFourState, legal_actions_now: Sequence[int]) -> int:
        candidate_actions = list(legal_actions_now)
        player = state.current_player
        opponent = 1 if player == 2 else 2

        winning_actions = []
        for action in candidate_actions:
            possible, _reward, ended, _lengths = self._outcome(state, action, player, return_length=False)
            if possible and ended:
                winning_actions.append(action)
        if winning_actions:
            return self._rng.choice(winning_actions)

        blocking_actions = []
        for action in candidate_actions:
            possible, _reward, ended, _lengths = self._outcome(state, action, opponent, return_length=False)
            if possible and ended:
                blocking_actions.append(action)
        if blocking_actions:
            return self._rng.choice(blocking_actions)

        return WeakHeuristicAgent._pick_weak_action(self, state, candidate_actions)


class HeuristicAgent(StrongHeuristicAgent):
    pass
