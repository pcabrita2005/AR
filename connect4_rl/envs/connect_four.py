from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence, List


# --- Game Constants ---
ROWS = 6
COLUMNS = 7
CONNECT_N = 4
EMPTY = 0


# --- Core Logic (Numpy based) ---

def get_legal_actions(board: np.ndarray) -> List[int]:
    """ Returns the list of columns that are not full. """
    return np.where(board[0] == 0)[0].tolist()

def get_illegal_actions(board: np.ndarray) -> List[int]:
    """ Returns the list of columns that are full. """
    return np.where(board[0] != 0)[0].tolist()

def drop_piece(board: np.ndarray, column: int, mark: int) -> np.ndarray:
    """ Returns a copy of the board with a piece dropped in the specified column. """
    next_board = board.copy()
    row = np.where(board[:, column] == 0)[0][-1]
    next_board[row, column] = mark
    return next_board

def is_game_winner(board: np.ndarray, mark: int, inrow: int = 4) -> bool:
    """ Checks if a player has won the game. """
    target = np.array([mark] * inrow)
    for j in range(board.shape[1] - inrow + 1):
        if np.any(np.all(target == board[:, j:j+inrow], axis=1)): return True
    for i in range(board.shape[0] - inrow + 1):
        if np.any(np.all(target == board[i:i+inrow, :].T, axis=1)): return True
    for i in range(board.shape[0] - inrow + 1):
        for j in range(board.shape[1] - inrow + 1):
            if np.all(board[i:i+inrow, j:j+inrow].diagonal() == target): return True
            if np.all(np.fliplr(board[i:i+inrow, j:j+inrow]).diagonal() == target): return True
    return False

def is_board_terminal(board: np.ndarray, inrow: int = 4) -> bool:
    """ Checks if the board is terminal (win or draw). """
    return (board == 0).sum() == 0 or is_game_winner(board, 1, inrow) or \
           is_game_winner(board, 2, inrow) or is_game_winner(board, -1, inrow)

def get_winning_cols(board: np.ndarray, mark: int, inrow: int = 4) -> List[int]:
    winning_columns = []
    for column in get_legal_actions(board):
        if is_game_winner(drop_piece(board, column, mark), mark, inrow):
            winning_columns.append(column)
    return winning_columns


# --- State-based API (For tests and existing agents) ---

@dataclass(frozen=True)
class ConnectFourState:
    board: tuple[tuple[int, ...], ...]
    current_player: int = 1
    winner: int = 0
    moves_played: int = 0
    last_action: int | None = None

    @property
    def is_draw(self) -> bool:
        return self.winner == 0 and self.moves_played == ROWS * COLUMNS

def initial_state() -> ConnectFourState:
    board = tuple(tuple(EMPTY for _ in range(COLUMNS)) for _ in range(ROWS))
    return ConnectFourState(board=board)

def legal_actions(state: ConnectFourState) -> list[int]:
    if state.winner or state.moves_played == ROWS * COLUMNS: return []
    board_np = np.array(state.board)
    return get_legal_actions(board_np)

def action_mask(state: ConnectFourState) -> list[int]:
    legal = set(legal_actions(state))
    return [1 if column in legal else 0 for column in range(COLUMNS)]

def is_terminal(state: ConnectFourState) -> bool:
    return state.winner != 0 or state.moves_played == ROWS * COLUMNS

def apply_action(state: ConnectFourState, action: int) -> ConnectFourState:
    board_np = np.array(state.board)
    if action not in get_legal_actions(board_np):
        raise ValueError(f"Illegal action {action}")
    new_board_np = drop_piece(board_np, action, state.current_player)
    winner = state.current_player if is_game_winner(new_board_np, state.current_player) else 0
    return ConnectFourState(
        board=tuple(tuple(row) for row in new_board_np),
        current_player=2 if state.current_player == 1 else 1,
        winner=winner,
        moves_played=state.moves_played + 1,
        last_action=action,
    )

def outcome_for_player(state: ConnectFourState, player: int) -> float:
    if state.winner == player: return 1.0
    if state.winner == 0: return 0.0
    return -1.0

def encode_state(state: ConnectFourState, perspective_player: int | None = None) -> list[list[list[float]]]:
    """ Encode the board as a 2 x 6 x 7 binary tensor from one player's perspective. """
    player = perspective_player or state.current_player
    opponent = 2 if player == 1 else 1
    own_plane = [[1.0 if cell == player else 0.0 for cell in row] for row in state.board]
    opp_plane = [[1.0 if cell == opponent else 0.0 for cell in row] for row in state.board]
    return [own_plane, opp_plane]

def render_ascii(state: ConnectFourState) -> str:
    symbols = {0: ".", 1: "X", 2: "O"}
    lines = [" ".join(symbols[cell] for cell in row) for row in state.board]
    lines.append(" ".join(str(i) for i in range(COLUMNS)))
    return "\n".join(lines)


class ConnectFourEnv:
    def __init__(self) -> None:
        self.state = initial_state()

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
        del seed
        self.state = initial_state()
        info = {"winner": self.state.winner, "legal_actions": legal_actions(self.state)}
        return self.observe(), info

    def observe(self) -> dict[str, object]:
        return {
            "board": self.state.board,
            "current_player": self.state.current_player,
            "action_mask": action_mask(self.state),
        }

    def step(self, action: int) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        acting_player = self.state.current_player
        self.state = apply_action(self.state, action)
        terminated = is_terminal(self.state)
        reward = outcome_for_player(self.state, acting_player) if terminated else 0.0
        info = {"winner": self.state.winner, "legal_actions": legal_actions(self.state)}
        return self.observe(), reward, terminated, False, info

    def render(self) -> str:
        return render_ascii(self.state)
