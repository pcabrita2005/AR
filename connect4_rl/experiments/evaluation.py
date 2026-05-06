from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Callable

from connect4_rl.core.base import MatchResult
from connect4_rl.envs.connect_four import ConnectFourState, apply_action, initial_state, is_terminal, legal_actions


AgentFactory = Callable[[], object]


def play_match(agent_one: object, agent_two: object, starter: int = 1) -> MatchResult:
    state = _with_starting_player(initial_state(), starter)
    agents = {starter: agent_one, 2 if starter == 1 else 1: agent_two}

    while not is_terminal(state):
        agent = agents[state.current_player]
        action = agent.select_action(state, legal_actions(state))
        state = apply_action(state, action)

    return MatchResult(winner=state.winner, moves=state.moves_played, starter=starter)


def round_robin(agent_factories: dict[str, AgentFactory], games_per_pair: int = 20) -> dict[str, dict[str, float]]:
    scoreboard, _match_log = round_robin_detailed(agent_factories, games_per_pair=games_per_pair)
    return scoreboard


def round_robin_detailed(
    agent_factories: dict[str, AgentFactory],
    games_per_pair: int = 20,
) -> tuple[dict[str, dict[str, float]], list[dict[str, float | int | str]]]:
    scoreboard: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    match_log: list[dict[str, float | int | str]] = []

    for left_name, right_name in combinations(agent_factories.keys(), 2):
        for game_idx in range(games_per_pair):
            left_starts = game_idx % 2 == 0
            starter = 1 if left_starts else 2

            left_agent = agent_factories[left_name]()
            right_agent = agent_factories[right_name]()

            if left_starts:
                result = play_match(left_agent, right_agent, starter=starter)
                player_for_left = 1
            else:
                result = play_match(right_agent, left_agent, starter=starter)
                player_for_left = 1

            if result.winner == 0:
                scoreboard[left_name]["draws"] += 1
                scoreboard[right_name]["draws"] += 1
            elif result.winner == player_for_left:
                scoreboard[left_name]["wins"] += 1
                scoreboard[right_name]["losses"] += 1
            else:
                scoreboard[left_name]["losses"] += 1
                scoreboard[right_name]["wins"] += 1

            scoreboard[left_name]["games"] += 1
            scoreboard[right_name]["games"] += 1
            match_log.append(
                {
                    "left_name": left_name,
                    "right_name": right_name,
                    "winner": result.winner,
                    "starter": result.starter,
                    "moves": result.moves,
                    "left_starts": int(left_starts),
                    "left_player": player_for_left,
                }
            )

    for name, metrics in scoreboard.items():
        games = max(metrics["games"], 1.0)
        metrics["win_rate"] = metrics["wins"] / games
        metrics["draw_rate"] = metrics["draws"] / games

    return {name: dict(metrics) for name, metrics in scoreboard.items()}, match_log


def compute_elo_ratings(
    match_log: list[dict[str, float | int | str]],
    *,
    initial_rating: float = 1200.0,
    k_factor: float = 24.0,
) -> dict[str, float]:
    ratings: dict[str, float] = {}

    for match in match_log:
        left_name = str(match["left_name"])
        right_name = str(match["right_name"])
        winner = int(match["winner"])
        left_starts = bool(match["left_starts"])

        ratings.setdefault(left_name, initial_rating)
        ratings.setdefault(right_name, initial_rating)

        left_rating = ratings[left_name]
        right_rating = ratings[right_name]
        expected_left = 1.0 / (1.0 + 10 ** ((right_rating - left_rating) / 400.0))
        expected_right = 1.0 - expected_left

        if winner == 0:
            score_left = 0.5
            score_right = 0.5
        else:
            left_player = int(match.get("left_player", 1 if left_starts else 2))
            if winner == left_player:
                score_left = 1.0
                score_right = 0.0
            else:
                score_left = 0.0
                score_right = 1.0

        ratings[left_name] = left_rating + k_factor * (score_left - expected_left)
        ratings[right_name] = right_rating + k_factor * (score_right - expected_right)

    return {name: round(value, 2) for name, value in sorted(ratings.items(), key=lambda item: item[1], reverse=True)}


def _with_starting_player(state: ConnectFourState, starter: int) -> ConnectFourState:
    return ConnectFourState(
        board=state.board,
        current_player=starter,
        winner=state.winner,
        moves_played=state.moves_played,
        last_action=state.last_action,
    )
