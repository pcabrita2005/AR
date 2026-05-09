import numpy as np
import torch
from torch.distributions import Categorical
from connect4_rl.agents.training.custom_trainable_agent import TrainableAgent
from .custom_agent_base import get_illegal_actions
from .custom_net import CustomNetwork

class PGAgent(TrainableAgent):
    """ Policy-Gradient Agent (PPO/REINFORCE). """

    def __init__(self,
                 stochastic_mode: bool = True,
                 avg_symmetric_probs: bool = True,
                 name: str = 'Policy-Gradient Agent',
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.stochastic_mode = stochastic_mode
        self.avg_symmetric_probs = avg_symmetric_probs

    def _get_probs(self, obs: np.ndarray) -> torch.Tensor:
        """ Internal helper to get masked probabilities. """
        model_input = self.model.obs_to_model_input(obs=obs)
        with torch.no_grad():
            logits, _ = self.model(model_input)
            probs = torch.softmax(logits.squeeze(), dim=0)

        if self.avg_symmetric_probs:
            sym_obs = np.flip(obs, axis=-1)
            sym_model_input = self.model.obs_to_model_input(obs=sym_obs)
            with torch.no_grad():
                sym_logits, _ = self.model(sym_model_input)
                sym_probs = torch.softmax(sym_logits.squeeze(), dim=0)
            probs = (probs + sym_probs.flip(dims=[0])) / 2

        if not self.allow_illegal_actions:
            illegal_actions = get_illegal_actions(board=obs)
            probs[illegal_actions] = 0
            if probs.sum() > 0:
                probs /= probs.sum()
            else:
                legal_actions = [i for i in range(len(probs)) if i not in illegal_actions]
                probs[legal_actions] = 1.0 / len(legal_actions)
        return probs

    def get_log_prob(self, obs: np.ndarray, action: int) -> torch.Tensor:
        """ Returns log probability of an action. """
        probs = self._get_probs(obs=obs)
        dist = Categorical(probs=probs)
        device = next(self.model.parameters()).device
        return dist.log_prob(torch.tensor(action).to(device))

    def get_exploitation_policy(self, obs: np.ndarray) -> Categorical:
        """ Returns exploitation policy (stochastic or greedy). """
        probs = self._get_probs(obs=obs)
        if not self.stochastic_mode:
            best_action = torch.argmax(probs)
            probs = torch.zeros_like(probs)
            probs[best_action] = 1.0
        return Categorical(probs=probs)

    def get_transition(self, state: np.ndarray, exploration_rate: float = None) -> dict:
        """ Returns transition (s, a, log_prob). """
        exploration_rate_ = self._process_exploration_rate(exploration_rate)
        internal_obs = state # Base class handles normalization if called from choose_action
    
        if np.random.random() < exploration_rate_:
            policy = self.get_exploration_policy(obs=state)
        else:
            policy = self.get_exploitation_policy(obs=state)
            
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return {'state': state.copy(), 'action': action.item(), 'log_prob': log_prob.item()}


if __name__ == "__main__":
    from connect4_rl.envs.connect_four import ConnectFourEnv
    model = CustomNetwork(conv_block=[[32, 4, 0], 'relu'],
                          fc_block=[64, 'relu'],
                          first_head=[64, 'relu', 7],
                          second_head=[64, 'relu', 1])
    agent = PGAgent(model=model)
    env = ConnectFourEnv()
    obs_dict, _ = env.reset()
    obs = np.array(obs_dict["board"])
    print('Action:', agent.choose_action(obs=obs))
