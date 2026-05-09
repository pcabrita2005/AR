import numpy as np
from connect4_rl.envs.connect_four import ConnectFourState

class LeftmostAgent:
    def __init__(self, name="Leftmost Agent"):
        self.name = name

    def select_action(self, state: ConnectFourState, legal_actions: list[int]) -> int:
        return min(legal_actions)
