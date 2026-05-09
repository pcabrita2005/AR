import copy
from collections import deque, namedtuple
from typing import List
import random
import numpy as np
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from connect4_rl.agents.training.custom_agent_base import Agent, get_illegal_actions
from connect4_rl.envs.connect_four import ConnectFourEnv, is_board_terminal as is_terminal
from connect4_rl.utils.reward_shaping import get_custom_reward

class ReplayMemory:
    """ FIFO Experience Replay Memory for training. """

    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'log_prob'))

    def __init__(self, capacity: int, reward_backprop_exponent: float = None) -> None:
        self.capacity = capacity
        self.reward_backprop_exponent = reward_backprop_exponent
        self.memory = deque(maxlen=self.capacity)
        self.original_rewards = deque(maxlen=self.capacity)

    def __getitem__(self, item): return self.memory[item]
    def __len__(self) -> int: return len(self.memory)
    def all_data(self) -> List: return list(self.memory)
    def is_empty(self) -> bool: return len(self.memory) == 0
    def sample(self, batch_size: int) -> List: return random.sample(self.memory, batch_size)
    def reset(self) -> None:
        self.memory.clear()
        self.original_rewards.clear()

    def _reward_backprop(self, turn: int, n_turns: int, last_reward: float) -> float:
        """ Calculates backpropagated reward for a specific turn. """
        if (n_turns - turn) % 2 == 1:
            win = last_reward < 0
        else:
            win = last_reward > 0
        t_eff = turn if win else turn + 1
        reward = abs(last_reward) * (t_eff/n_turns) ** self.reward_backprop_exponent
        return round(reward if win else -reward, 4)

    def _backprop_episode_rewards(self, transitions: List[dict]) -> List[dict]:
        """ Distributes final reward across previous moves. """
        if not transitions[-1]['done'] or self.reward_backprop_exponent is None:
            return transitions
            
        data = copy.deepcopy(transitions)
        max_r = abs(transitions[-1]['reward'])
        init_t = (data[0]['state'] != 0).sum() + 1
        curr_init = init_t
        
        for t in range(init_t, init_t + len(data)):
            last_r = data[t - init_t]['reward']
            if abs(abs(last_r) - max_r) < 1e-4:
                for i in range(curr_init, t):
                    if abs(data[i - init_t]['reward']) < 1e-4:
                        data[i - init_t]['reward'] = self._reward_backprop(i, t, last_r)
                curr_init = t + 1
        return data

    def push(self, transitions: List[dict]) -> None:
        """ Processes and saves an episode to memory. """
        if not transitions[-1]['done']: return
        
        data = copy.deepcopy(transitions)
        if data[-1]['action'] in get_illegal_actions(board=data[-1]['state']):
            data = [data[-1]] # Only learn from the disqualifying move

        self.original_rewards.extend([t['reward'] for t in data])
        data = self._backprop_episode_rewards(data)
        for tt in data:
            self.memory.append(self.Transition(**tt))

    def push_self_play_episode_transitions(self, agent: Agent, env: ConnectFourEnv, 
                                           init_random_obs: bool = False,
                                           push_symmetric: bool = True, exploration_rate: float = None) -> None:
        """ Plays a full episode and stores it. """
        episode = []
        obs_dict, info = env.reset()
        obs = np.array(obs_dict["board"], dtype=np.float32)
        done = False
        
        while not done:
            p = obs_dict["current_player"]
            in_obs = np.zeros_like(obs, dtype=np.float32)
            in_obs[obs == p], in_obs[obs == (3-p)] = 1.0, -1.0
            
            with torch.no_grad():
                if hasattr(agent, "choose_action"):
                    action = agent.choose_action(obs=obs, exploration_rate=exploration_rate, active_player=p)
                else:
                    from connect4_rl.envs.connect_four import ConnectFourState
                    state = ConnectFourState(board=tuple(tuple(row) for row in obs), current_player=p)
                    action = agent.select_action(state, info["legal_actions"])
                
                log_p = agent.get_log_prob(obs=in_obs, action=action) if hasattr(agent, "get_log_prob") else None
            
            obs_dict, reward, term, trunc, info = env.step(action)
            next_obs = np.array(obs_dict["board"], dtype=np.float32)
            done = term or trunc
            
            in_next = np.zeros_like(next_obs, dtype=np.float32)
            in_next[next_obs == p], in_next[next_obs == (3-p)] = 1.0, -1.0
            
            episode.append({
                'state': in_obs.copy(), 'action': action, 'reward': get_custom_reward(obs, p, action, done, info.get("winner", 0)),
                'next_state': in_next.copy(), 'done': done, 'log_prob': log_p
            })
            obs = next_obs
            
        self.push(episode)
        if push_symmetric:
            self.push([agent.get_symmetric_transition(t) for t in episode])

if __name__ == "__main__":
    from connect4_rl.agents.baselines.n_step_lookahead_agent import NStepLookaheadAgent
    env = ConnectFourEnv()
    mem = ReplayMemory(100, 3)
    mem.push_self_play_episode_transitions(NStepLookaheadAgent(n=1), env, push_symmetric=False)
    print(f"Stored {len(mem)} transitions.")
