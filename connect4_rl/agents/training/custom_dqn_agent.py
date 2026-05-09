from typing import Tuple
import numpy as np
import torch
from torch.distributions import Categorical
from connect4_rl.agents.training.custom_trainable_agent import TrainableAgent
from .custom_agent_base import get_illegal_actions
from .custom_net import CustomNetwork

class DQNAgent(TrainableAgent):
    """ Deep Q-Network Agent. """

    def __init__(self,
                 name: str = 'DQN Agent',
                 avg_symmetric_q_vals: bool = False,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.avg_symmetric_q_vals = avg_symmetric_q_vals

    def get_q_vals(self, obs: np.ndarray) -> torch.Tensor:
        """ Returns Q-values for each action. """
        model_input = self.model.obs_to_model_input(obs=obs)
        with torch.no_grad():
            q_vals = self.model(model_input).squeeze()

        if self.avg_symmetric_q_vals:
            sym_obs = np.flip(obs, axis=-1)
            sym_model_input = self.model.obs_to_model_input(obs=sym_obs)
            with torch.no_grad():
                sym_q_vals = self.model(sym_model_input).squeeze()
            q_vals = (q_vals + sym_q_vals.flip(dims=[0])) / 2

        if not self.allow_illegal_actions:
            illegal_actions = get_illegal_actions(board=obs)
            q_vals[illegal_actions] = -torch.inf
        return q_vals

    def get_exploitation_policy(self, obs: np.ndarray) -> Categorical:
        """ Returns greedy policy based on Q-values. """
        q_vals = self.get_q_vals(obs=obs)
        best_action = q_vals.argmax()
        probs = torch.zeros_like(q_vals)
        probs[best_action] = 1.0
        return Categorical(probs=probs)

    def get_policy_scores_to_visualize(self, obs: np.ndarray) -> Tuple:
        return tuple(self.get_q_vals(obs=obs).cpu().tolist())

if __name__ == "__main__":
    from connect4_rl.envs.connect_four import ConnectFourEnv
    model = CustomNetwork(conv_block=[[32, 4, 0], 'relu'],
                          fc_block=[64, 'relu'],
                          first_head=[64, 'relu', 7])
    agent = DQNAgent(model=model)
    env = ConnectFourEnv()
    obs_dict, _ = env.reset()
    print('Action:', agent.choose_action(obs=np.array(obs_dict["board"])))
