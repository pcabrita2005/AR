from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


ROWS = 6
COLUMNS = 7
CONNECT_N = 4
EMPTY = 0


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
    if state.winner or state.moves_played == ROWS * COLUMNS:
        return []
    return [column for column in range(COLUMNS) if state.board[0][column] == EMPTY]


def action_mask(state: ConnectFourState) -> list[int]:
    legal = set(legal_actions(state))
    return [1 if column in legal else 0 for column in range(COLUMNS)]


def is_terminal(state: ConnectFourState) -> bool:
    return state.winner != 0 or state.moves_played == ROWS * COLUMNS


def apply_action(state: ConnectFourState, action: int) -> ConnectFourState:
    legal = legal_actions(state)
    if action not in legal:
        raise ValueError(f"Illegal action {action}; legal actions are {legal}")

    board = [list(row) for row in state.board]
    row_to_fill = _find_open_row(board, action)
    board[row_to_fill][action] = state.current_player
    winner = state.current_player if _has_connect_n(board, row_to_fill, action, state.current_player) else 0

    return ConnectFourState(
        board=tuple(tuple(row) for row in board),
        current_player=2 if state.current_player == 1 else 1,
        winner=winner,
        moves_played=state.moves_played + 1,
        last_action=action,
    )


def outcome_for_player(state: ConnectFourState, player: int) -> float:
    if state.winner == player:
        return 1.0
    if state.winner == 0:
        return 0.0
    return -1.0


def encode_state(state: ConnectFourState, perspective_player: int | None = None) -> list[list[list[float]]]:
    """Encode the board as a 2 x 6 x 7 binary tensor from one player's perspective."""

    player = perspective_player or state.current_player
    opponent = 2 if player == 1 else 1
    own_plane = [[1.0 if cell == player else 0.0 for cell in row] for row in state.board]
    opp_plane = [[1.0 if cell == opponent else 0.0 for cell in row] for row in state.board]
    return [own_plane, opp_plane]


def render_ascii(state: ConnectFourState) -> str:
    symbols = {0: ".", 1: "X", 2: "O"}
    lines = [" ".join(symbols[cell] for cell in row) for row in state.board]
    lines.append("0 1 2 3 4 5 6")
    return "\n".join(lines)


class ConnectFourEnv:
    """Minimal two-player environment with Gymnasium-like reset/step methods."""

    def __init__(self) -> None:
        self.state = initial_state()

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, object], dict[str, object]]:
        del seed
        self.state = initial_state()
        return self.observe(), {}

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
        info = {
            "winner": self.state.winner,
            "legal_actions": legal_actions(self.state),
        }
        return self.observe(), reward, terminated, False, info

    def render(self) -> str:
        return render_ascii(self.state)


def _find_open_row(board: Sequence[Sequence[int]], action: int) -> int:
    for row in range(ROWS - 1, -1, -1):
        if board[row][action] == EMPTY:
            return row
    raise ValueError(f"Column {action} is full")


def _has_connect_n(board: Sequence[Sequence[int]], row: int, column: int, player: int) -> bool:
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for d_row, d_col in directions:
        count = 1
        count += _count_direction(board, row, column, d_row, d_col, player)
        count += _count_direction(board, row, column, -d_row, -d_col, player)
        if count >= CONNECT_N:
            return True
    return False


def _count_direction(
    board: Sequence[Sequence[int]],
    row: int,
    column: int,
    d_row: int,
    d_col: int,
    player: int,
) -> int:
    count = 0
    r = row + d_row
    c = column + d_col
    while 0 <= r < ROWS and 0 <= c < COLUMNS and board[r][c] == player:
        count += 1
        r += d_row
        c += d_col
    return count
