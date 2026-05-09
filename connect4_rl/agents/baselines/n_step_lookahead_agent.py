from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from connect4_rl.envs.connect_four import ConnectFourState, get_legal_actions, drop_piece, is_game_winner, is_board_terminal as is_terminal


def count_n_in_row(board: np.ndarray, n: int, mark: int, inrow: int = 4) -> int:
    count = 0
    check_marks = lambda ww: (ww == mark).sum(axis=1) == n
    check_spaces = lambda ww: (ww == 0).sum(axis=1) == inrow-n
    for j in range(board.shape[1] - inrow + 1):
        w = board[:, j:j+inrow]
        count += np.sum(check_spaces(w) & check_marks(w))
    for i in range(board.shape[0] - inrow + 1):
        w = board[i:i+inrow, :].T
        count += np.sum(check_spaces(w) & check_marks(w))
    check_marks_diag = lambda ww: (ww == mark).sum() == n
    check_spaces_diag = lambda ww: (ww == 0).sum() == inrow-n
    for i in range(board.shape[0] - inrow + 1):
        for j in range(board.shape[1] - inrow + 1):
            asc_diag = board[i:i+inrow, j:j+inrow].diagonal()
            des_diag = np.fliplr(board[i:i+inrow, j:j+inrow]).diagonal()
            count += np.sum(check_spaces_diag(asc_diag) & check_marks_diag(asc_diag))
            count += np.sum(check_spaces_diag(des_diag) & check_marks_diag(des_diag))
    return count


class NStepLookaheadAgent:

    _pattern_scores = {
        4: 1e10,   # four of your tokens in a row
        3: 1e4,    # three of your tokens in a row
        2: 1e2,    # two of your tokens in a row
        -2: -1,    # two of the opponent's tokens in a row
        -3: -1e6,  # three of the opponent's tokens in a row
        -4: -1e8   # four of the opponent's tokens in a row
    }

    def __init__(self, n: int = 2, name: str = None, prefer_central_columns: bool = True):
        self.n = n
        self.name = name or f"{n}-Step Lookahead"
        self.prefer_central_columns = prefer_central_columns

    def select_action(self, state: ConnectFourState, legal_actions: Sequence[int]) -> int:
        active_player = state.current_player
        opponent = 2 if active_player == 1 else 1
        
        # Convert board to 1 (me) and -1 (opponent)
        board = np.array(state.board, dtype=np.int32)
        internal_board = np.zeros_like(board)
        internal_board[board == active_player] = 1
        internal_board[board == opponent] = -1
        
        scores = self._compute_scores(internal_board)
        best_scored_actions = np.where(scores == np.amax(scores))[0]
        
        if self.prefer_central_columns:
            center = board.shape[1] // 2
            best_action = best_scored_actions[np.argmin(np.abs(best_scored_actions - center))]
        else:
            best_action = np.random.choice(best_scored_actions)
            
        return int(best_action)

    def _compute_scores(self, board: np.ndarray) -> np.ndarray:
        scores = np.full(board.shape[1], -np.inf)
        legal_actions = get_legal_actions(board)
        for action in legal_actions:
            next_board = drop_piece(board=board, column=action, mark=1)
            scores[action] = self._minmax_search(
                board=next_board, depth=self.n-1, is_max_player=False
            )
        return scores

    def _score_leaf_board(self, board: np.ndarray) -> float:
        score = 0
        for pattern, pattern_score in self._pattern_scores.items():
            mark = 1 if pattern > 0 else -1
            counts = count_n_in_row(board=board, n=abs(pattern), mark=mark)
            score += counts * pattern_score
        return score

    def _minmax_search(self, board: np.ndarray, depth: int, is_max_player: bool) -> float:
        if depth == 0 or is_terminal(board=board):
            return self._score_leaf_board(board=board)

        legal_actions = get_legal_actions(board=board)
        if is_max_player:
            value = -np.inf
            for action in legal_actions:
                child_board = drop_piece(board=board, column=action, mark=1)
                value = max(value, self._minmax_search(child_board, depth-1, False))
            return value
        else:
            value = np.inf
            for action in legal_actions:
                child_board = drop_piece(board=board, column=action, mark=-1)
                value = min(value, self._minmax_search(child_board, depth-1, True))
            return value
