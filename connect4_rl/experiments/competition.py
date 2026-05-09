from tqdm import trange
from typing import Tuple, List

import torch
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect4_rl.agents.training.custom_agent_base import Agent
from connect4_rl.envs.connect_four import ConnectFourEnv


def run_episode(env: ConnectFourEnv,
                agent1: Agent,
                agent2: Agent,
                exploration_rate: float = None,
                print_transitions: bool = False,
                initial_actions: List[int] = ()) -> Tuple[dict, List]:
    """
    Runs an entire episode: agent1 versus agent2 in the given environment.
    Returns the episode information and the list of observations (game boards)
    """

    obs_dict, info = env.reset()
    obs = np.array(obs_dict['board'])
    done = False
    # run the initial actions (if any)
    for init_action in initial_actions:
        obs_dict, _, terminated, truncated, _ = env.step(action=init_action)
        obs = np.array(obs_dict['board'])
        done = terminated or truncated
        if done:
            raise ValueError("initial_actions lead to a terminal board")

    obs_list = [obs]
    if print_transitions:
        print('obs:', obs)

    active_player = agent1 if len(initial_actions) % 2 == 0 else agent2

    while not done:
        with torch.no_grad():
            # Handle both types of select_action/choose_action
            if hasattr(active_player, "select_action"):
                # Our existing agents
                from connect4_rl.envs.connect_four import ConnectFourState
                state = ConnectFourState(board=tuple(tuple(row) for row in obs), 
                                         current_player=obs_dict["current_player"])
                action = active_player.select_action(state, info["legal_actions"])
            else:
                # The "training" agents
                action = active_player.choose_action(obs=obs, exploration_rate=exploration_rate, 
                                                   active_player=obs_dict["current_player"])
                                                   
        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = np.array(obs_dict['board'])
        done = terminated or truncated
        obs_list.append(obs)
        active_player = agent2 if active_player == agent1 else agent1
        if print_transitions:
            print(f'action: {action}, reward: {reward}\n'+'-'*30+f'\nobs: {obs}')

    # Add game_len and is_a_draw for competition statistics
    info['game_len'] = len(obs_list) - 1
    info['is_a_draw'] = (info['winner'] == 0)
    # We don't have cumulative rewards easily here, but we can fake them for the competition loop
    info['rewards1'] = [1.0 if info['winner'] == 1 else (-1.0 if info['winner'] == 2 else 0.0)]
    info['rewards2'] = [1.0 if info['winner'] == 2 else (-1.0 if info['winner'] == 1 else 0.0)]
    
    return info, obs_list


def _get_initial_actions(game_id: int, ncols: int = 7) -> List[int]:
    """ game_id -> initial_actions mapping for fair initialization. """
    game_id_ = game_id % 50
    if game_id_ == 49:
        return []  # the empty board
    return [game_id_ % ncols, game_id_ // ncols]  # [for player1, for player2]


@torch.no_grad()
def competition(
        env: ConnectFourEnv,
        agent1: Agent,
        agent2: Agent,
        progress_bar: bool = True) -> Tuple[dict, List]:
    """
    Runs a fair competition between agent1 and agent2 in the given environment.
    100 games, balancing starting player and using initial board variations.
    """

    n_episodes = 100

    # Define the keys that will be tracked regardless of who plays first
    general_info = {
        'avg_cum_reward1': 0, 'avg_cum_reward2': 0, 'n_wins1': 0,  'n_wins2': 0
    }
    # Define the keys tracked separately depending on who plays first
    starting_player_dependent_vars = (
        'win_rate1', 'win_rate2', 'draw_rate', 'avg_game_len'
    )

    for key_ in starting_player_dependent_vars:
        general_info[key_] = 0
        general_info[key_ + '_s1'] = 0
        general_info[key_ + '_s2'] = 0

    last_obs_list = []

    if progress_bar:  # display tdqm progress bar
        iter_ = trange(n_episodes, desc=f'{agent1.name} vs {agent2.name}')
    else:
        iter_ = range(n_episodes)

    for i in iter_:
        initial_actions = _get_initial_actions(game_id=i)
        if i < n_episodes // 2:
            info, obs_list = run_episode(env=env, agent1=agent1, agent2=agent2,
                                         initial_actions=initial_actions)
            general_info['draw_rate_s1'] += int(info['is_a_draw'])
            general_info['win_rate1_s1'] += int(info['winner'] == 1)
            general_info['win_rate2_s1'] += int(info['winner'] == 2)
            general_info['avg_game_len_s1'] += info['game_len']
            general_info['avg_cum_reward1'] += sum(info['rewards1'])
            general_info['avg_cum_reward2'] += sum(info['rewards2'])

            last_obs_list.append(obs_list[-1])

        else:
            info, obs_list = run_episode(env=env, agent1=agent2, agent2=agent1,
                                         initial_actions=initial_actions)
            general_info['draw_rate_s2'] += int(info['is_a_draw'])
            general_info['win_rate1_s2'] += int(info['winner'] == 2)
            general_info['win_rate2_s2'] += int(info['winner'] == 1)
            general_info['avg_game_len_s2'] += info['game_len']
            general_info['avg_cum_reward1'] += sum(info['rewards2'])
            general_info['avg_cum_reward2'] += sum(info['rewards1'])

            last_obs_list.append(obs_list[-1])

    # gather partial information to compute the general results
    for k in starting_player_dependent_vars:
        general_info[k] = (general_info[k+'_s1'] + general_info[k+'_s2'])
    
    general_info['n_wins1'] = general_info['win_rate1']
    general_info['n_wins2'] = general_info['win_rate2']

    # finish the win_rates computation (add draw games)
    for s in (1, 2):  # starting_player
        general_info[f'win_rate1_s{s}'] += 0.5*general_info[f'draw_rate_s{s}']
        general_info[f'win_rate2_s{s}'] += 0.5*general_info[f'draw_rate_s{s}']

    general_info['avg_game_len_s1'] /= n_episodes//2
    general_info['avg_game_len_s2'] /= n_episodes//2
    general_info['win_rate1_s1'] /= n_episodes//2
    general_info['win_rate1_s2'] /= n_episodes//2
    general_info['win_rate2_s1'] /= n_episodes//2
    general_info['win_rate2_s2'] /= n_episodes//2
    general_info['draw_rate_s1'] /= n_episodes//2
    general_info['draw_rate_s2'] /= n_episodes//2

    general_info['avg_game_len'] /= n_episodes
    general_info['avg_cum_reward1'] /= n_episodes
    general_info['avg_cum_reward2'] /= n_episodes
    general_info['draw_rate'] /= n_episodes
    general_info['win_rate1'] /= n_episodes
    general_info['win_rate2'] /= n_episodes

    for k in general_info.keys():
        general_info[k] = round(general_info[k], 5)

    general_info['n_episodes'] = n_episodes
    general_info['agent1_name'] = agent1.name
    general_info['agent2_name'] = agent2.name

    return general_info, last_obs_list


if __name__ == "__main__":
    from connect4_rl.agents.training.custom_dqn_agent import DQNAgent
    from connect4_rl.agents.training.custom_net import CustomNetwork
    from connect4_rl.agents.baselines import RandomAgent

    # 1. Setup Environment
    env = ConnectFourEnv()

    # 2. Setup a Training Agent (with random weights for testing)
    model = CustomNetwork(conv_block=[[32, 3, 1]], fc_block=[64], first_head=[7])
    agent1 = DQNAgent(model=model, name="Training_DQN_Test")
    
    # 3. Setup a Baseline Agent
    agent2 = RandomAgent()

    print(f"Starting quick test: {agent1.name} vs {agent2.name}")
    
    # 4. Run a single episode
    info, _ = run_episode(env, agent1, agent2, print_transitions=True)
    print(f"\nEpisode Finished! Winner: {info['winner']}")

    # 5. Run a small competition
    print("\nRunning a mini-competition (10 games)...")
    results, _ = competition(env, agent1, agent2, progress_bar=False)
    print(f"Results: {results}")
