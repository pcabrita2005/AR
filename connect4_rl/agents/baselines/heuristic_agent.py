from __future__ import annotations

import random
from typing import Sequence

from connect4_rl.envs.connect_four import COLUMNS, ConnectFourState, ROWS, apply_action, legal_actions


def score_position(state: ConnectFourState, player: int) -> int:
    opponent = 2 if player == 1 else 1
    board = state.board
    score = 0

    center_column = COLUMNS // 2
    center_count = sum(1 for row in range(ROWS) if board[row][center_column] == player)
    score += center_count * 6

    windows = all_windows(board)
    for window in windows:
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

    def select_action(self, state: ConnectFourState, legal_actions_now: Sequence[int]) -> int:
        raise NotImplementedError

    def _score_state(self, state: ConnectFourState, player: int) -> int:
        return score_position(state, player)

    def _pick_best_scored_action(
        self,
        state: ConnectFourState,
        legal_actions_now: Sequence[int],
        *,
        avoid_immediate_reply: bool,
    ) -> int:
        player = state.current_player
        opponent = 2 if player == 1 else 1

        if avoid_immediate_reply:
            safe_actions = [
                action for action in legal_actions_now if not self._allows_immediate_reply_win(state, action, opponent)
            ]
            candidate_actions = safe_actions if safe_actions else list(legal_actions_now)
        else:
            candidate_actions = list(legal_actions_now)

        scored_actions = []
        for action in candidate_actions:
            next_state = apply_action(state, action)
            score = self._score_state(next_state, player)
            scored_actions.append((score, action))

        best_score = max(score for score, _action in scored_actions)
        best_actions = [action for score, action in scored_actions if score == best_score]
        return self._rng.choice(best_actions)

    def _find_immediate_win(
        self,
        state: ConnectFourState,
        legal_actions_now: Sequence[int],
        target_player: int,
    ) -> int | None:
        for action in legal_actions_now:
            next_state = self._simulate_player_move(state, action, target_player)
            if next_state.winner == target_player:
                return action
        return None

    def _allows_immediate_reply_win(self, state: ConnectFourState, action: int, opponent: int) -> bool:
        next_state = apply_action(state, action)
        if next_state.winner != 0:
            return False
        opponent_winning_reply = self._find_immediate_win(next_state, legal_actions(next_state), opponent)
        return opponent_winning_reply is not None

    @staticmethod
    def _simulate_player_move(state: ConnectFourState, action: int, player: int) -> ConnectFourState:
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


class WeakHeuristicAgent(BaseHeuristicAgent):
    def __init__(self, seed: int | None = None, name: str = "weak_heuristic") -> None:
        super().__init__(seed=seed, name=name)

    def select_action(self, state: ConnectFourState, legal_actions_now: Sequence[int]) -> int:
        return self._pick_best_scored_action(state, legal_actions_now, avoid_immediate_reply=False)


class StrongHeuristicAgent(BaseHeuristicAgent):
    def __init__(self, seed: int | None = None, name: str = "heuristic") -> None:
        super().__init__(seed=seed, name=name)

    def select_action(self, state: ConnectFourState, legal_actions_now: Sequence[int]) -> int:
        player = state.current_player
        opponent = 2 if player == 1 else 1

        winning_action = self._find_immediate_win(state, legal_actions_now, player)
        if winning_action is not None:
            return winning_action

        blocking_action = self._find_immediate_win(state, legal_actions_now, opponent)
        if blocking_action is not None:
            return blocking_action

        return self._pick_best_scored_action(state, legal_actions_now, avoid_immediate_reply=True)


class HeuristicAgent(StrongHeuristicAgent):
    pass
