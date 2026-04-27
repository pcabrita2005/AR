from __future__ import annotations

import random
from typing import Sequence


class RandomAgent:
    def __init__(self, seed: int | None = None, name: str = "random") -> None:
        self.name = name
        self._rng = random.Random(seed)

    def select_action(self, state: object, legal_actions: Sequence[int]) -> int:
        del state
        return self._rng.choice(list(legal_actions))

