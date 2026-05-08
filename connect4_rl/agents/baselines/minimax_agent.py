from __future__ import annotations

import math
import random
from typing import Sequence

from connect4_rl.agents.baselines.heuristic_agent import score_position
from connect4_rl.envs.connect_four import ConnectFourState, apply_action, is_terminal, legal_actions, outcome_for_player


class MinimaxAgent:
    def __init__(self, depth: int = 2, seed: int | None = None, name: str | None = None) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth
        self.name = name or f"minimax_d{depth}"
        self._rng = random.Random(seed)

    def select_action(self, state: ConnectFourState, legal_actions_now: Sequence[int]) -> int:
        legal = list(legal_actions_now)
        if len(legal) == 1:
            return legal[0]

        player = state.current_player
        best_score = -math.inf
        best_actions: list[int] = []
        for action in self._ordered_actions(legal):
            next_state = apply_action(state, action)
            score = self._min_value(next_state, player, self.depth - 1, -math.inf, math.inf)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return self._rng.choice(best_actions)

    def _max_value(
        self,
        state: ConnectFourState,
        player: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        if depth <= 0 or is_terminal(state):
            return self._evaluate(state, player, depth)
        value = -math.inf
        for action in self._ordered_actions(legal_actions(state)):
            value = max(value, self._min_value(apply_action(state, action), player, depth - 1, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def _min_value(
        self,
        state: ConnectFourState,
        player: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        if depth <= 0 or is_terminal(state):
            return self._evaluate(state, player, depth)
        value = math.inf
        for action in self._ordered_actions(legal_actions(state)):
            value = min(value, self._max_value(apply_action(state, action), player, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _evaluate(self, state: ConnectFourState, player: int, depth: int) -> float:
        if is_terminal(state):
            outcome = outcome_for_player(state, player)
            # Prefer faster wins and slower losses.
            return (10_000.0 + depth) if outcome > 0 else (-10_000.0 - depth) if outcome < 0 else 0.0
        return float(score_position(state, player) - score_position(state, 1 if player == 2 else 2))

    @staticmethod
    def _ordered_actions(actions: Sequence[int]) -> list[int]:
        preferred_order = [3, 2, 4, 1, 5, 0, 6]
        return sorted(actions, key=lambda action: preferred_order.index(action))
