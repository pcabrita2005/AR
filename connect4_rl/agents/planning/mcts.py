from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from connect4_rl.agents.baselines.heuristic_agent import HeuristicAgent
from connect4_rl.envs.connect_four import ConnectFourState, apply_action, is_terminal, legal_actions, outcome_for_player


@dataclass
class MCTSNode:
    state: ConnectFourState
    parent: "MCTSNode | None" = None
    action_from_parent: int | None = None
    untried_actions: list[int] = field(default_factory=list)
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        if not self.untried_actions:
            self.untried_actions = _ordered_actions(self.state)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions

    def best_child(self, exploration_weight: float) -> "MCTSNode":
        assert self.children, "best_child called with no children"

        def score(node: MCTSNode) -> float:
            exploitation = -node.value
            exploration = exploration_weight * math.sqrt(math.log(self.visit_count) / node.visit_count)
            return exploitation + exploration

        return max(self.children.values(), key=score)

    def expand(self) -> "MCTSNode":
        action = self.untried_actions.pop()
        next_state = apply_action(self.state, action)
        child = MCTSNode(
            state=next_state,
            parent=self,
            action_from_parent=action,
        )
        self.children[action] = child
        return child

    def backpropagate(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTSAgent:
    def __init__(
        self,
        simulations: int = 200,
        exploration_weight: float = math.sqrt(2.0),
        rollout_seed: int | None = None,
        use_heuristic_rollout: bool = True,
        name: str = "mcts",
    ) -> None:
        self.name = name
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        self._rng = random.Random(rollout_seed)
        self.use_heuristic_rollout = use_heuristic_rollout
        self._heuristic_rollout_agent = HeuristicAgent(seed=rollout_seed, name="heuristic_rollout")

    def select_action(self, state: ConnectFourState, legal_actions_now: list[int]) -> int:
        if len(legal_actions_now) == 1:
            return legal_actions_now[0]

        root = MCTSNode(state=state)
        for _ in range(self.simulations):
            node = root

            while node.is_fully_expanded() and node.children and not is_terminal(node.state):
                node = node.best_child(self.exploration_weight)

            if not is_terminal(node.state) and node.untried_actions:
                node = node.expand()

            value = self._rollout(node.state)
            node.backpropagate(value)

        best_action, _best_node = max(
            root.children.items(),
            key=lambda item: (item[1].visit_count, -item[1].value),
        )
        return best_action

    def _rollout(self, state: ConnectFourState) -> float:
        current_state = state
        perspective_player = state.current_player
        while not is_terminal(current_state):
            current_legal_actions = legal_actions(current_state)
            if self.use_heuristic_rollout:
                action = self._heuristic_rollout_agent.select_action(current_state, current_legal_actions)
            else:
                action = self._rng.choice(current_legal_actions)
            current_state = apply_action(current_state, action)
        return outcome_for_player(current_state, perspective_player)


def _ordered_actions(state: ConnectFourState) -> list[int]:
    actions = legal_actions(state)
    preferred_order = [3, 2, 4, 1, 5, 0, 6]
    return sorted(actions, key=lambda action: preferred_order.index(action))
