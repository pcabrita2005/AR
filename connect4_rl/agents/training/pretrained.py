import numpy as np
import torch
from pathlib import Path
from typing import Sequence
from connect4_rl.envs.connect_four import ConnectFourState
from connect4_rl.agents.training.custom_net import CustomNetwork

class PretrainedAgent:
    """ Agent for models from the external repository. """
    
    def __init__(self, model_path: str | Path, arch_path: str | Path, name: str = "pretrained", n_heads: int = 2, agent_type: str = "ppo", device: str = "cpu"):
        self.name = name
        self.agent_type = agent_type
        self.device = torch.device(device)
        # CustomNetwork handles internal device placement automatically
        self.model = CustomNetwork.from_architecture(str(arch_path), n_heads=n_heads)
        self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def select_action(self, state: ConnectFourState, legal_actions: Sequence[int]) -> int:
        """ Selects action after normalizing board to {1, -1} perspective. """
        active_player = state.current_player
        opponent = 2 if active_player == 1 else 1
        
        obs = np.array(state.board, dtype=np.float32)
        transformed_obs = np.zeros_like(obs)
        transformed_obs[obs == active_player] = 1
        transformed_obs[obs == opponent] = -1
        
        model_input = self.model.obs_to_model_input(transformed_obs)
        
        with torch.no_grad():
            out = self.model(model_input)
            if self.agent_type == "dqn":
                q_vals = out.squeeze()
            elif self.agent_type == "dueling":
                adv, v = out
                q_vals = v + (adv - adv.mean())
                q_vals = q_vals.squeeze()
            elif self.agent_type == "ppo":
                logits, v = out
                q_vals = logits.squeeze()
            else:
                q_vals = out.squeeze()
            
        masked_q_vals = torch.full_like(q_vals, -1e9)
        masked_q_vals[list(legal_actions)] = q_vals[list(legal_actions)]
        return int(torch.argmax(masked_q_vals).item())
