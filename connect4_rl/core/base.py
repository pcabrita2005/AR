from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class MatchResult:
    winner: int
    moves: int
    starter: int


class Agent(Protocol):
    name: str

    def select_action(self, state: object, legal_actions: Sequence[int]) -> int:
        """Return an action for the current state."""

