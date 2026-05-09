import random
from typing import Tuple, List
import numpy as np
import torch
from torch.distributions import Categorical
from connect4_rl.envs.connect_four import get_legal_actions, get_illegal_actions

class Agent:
    """ Base Agent class for Connect4. """

    def __init__(self,
                 name: str = "Agent",
                 exploration_rate: float = 0.,
                 allow_illegal_actions: bool = False) -> None:
        self.name = name
        self.exploration_rate = exploration_rate
        self.allow_illegal_actions = allow_illegal_actions

    def get_exploration_policy(self, obs: np.ndarray) -> Categorical:
        """ Uniform policy over legal actions. """
        n_actions = obs.shape[-1]
        if self.allow_illegal_actions:
            probs = torch.full([n_actions], fill_value=1 / n_actions)
        else:
            probs = torch.zeros(n_actions)
            legal_actions = get_legal_actions(board=obs)
            probs[legal_actions] = 1 / len(legal_actions)
        return Categorical(probs=probs)

    def get_exploitation_policy(self, obs: np.array) -> Categorical:
        """ To be implemented by children. """
        pass

    def get_policy_scores_to_visualize(self, obs: np.ndarray) -> Tuple:
        """ Returns probabilities or Q-values for visualization. """
        policy = self.get_exploitation_policy(obs=obs)
        return tuple(policy.probs.cpu().tolist())

    def _process_exploration_rate(self, exploration_rate: float) -> float:
        return exploration_rate if exploration_rate is not None else self.exploration_rate

    def choose_action(self, obs: np.ndarray, exploration_rate: float = None, active_player: int = 1) -> int:
        """ Epsilon-greedy action selection. """
        internal_obs = np.zeros_like(obs, dtype=np.float32)
        internal_obs[obs == active_player] = 1.0
        opponent = 2 if active_player == 1 else 1 # Assuming marks are 1 and 2
        internal_obs[obs == opponent] = -1.0
        
        exploration_rate_ = self._process_exploration_rate(exploration_rate)
        if random.random() < exploration_rate_:
            policy = self.get_exploration_policy(obs=internal_obs)
        else:
            policy = self.get_exploitation_policy(obs=internal_obs)
        return policy.sample().item()

    def get_transition(self, state: np.ndarray, exploration_rate: float = None) -> dict:
        """ Returns basic (s, a) transition. """
        with torch.no_grad():
            action = self.choose_action(obs=state, exploration_rate=exploration_rate)
        return {'state': state.copy(), 'action': action, 'log_prob': None}

    @staticmethod
    def get_symmetric_transition(transition: dict) -> dict:
        """ Flips the board and action horizontally and copies all metadata. """
        symmetric = transition.copy()
        symmetric['state'] = np.flip(transition['state'], axis=-1).copy()
        symmetric['action'] = 6 - transition['action']
        if 'next_state' in transition:
            symmetric['next_state'] = np.flip(transition['next_state'], axis=-1).copy()
        return symmetric
