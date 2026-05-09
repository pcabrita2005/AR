import numpy as np
from connect4_rl.envs.connect_four import get_legal_actions, drop_piece, is_game_winner, get_winning_cols

def get_custom_reward(board: np.ndarray, mark: int, action: int, done: bool, winner: int) -> float:
    """ Computes win/loss reward + intermediate penalties for missed wins/opponent wins. """
    if done:
        if winner == mark: return 1.0
        if winner == 0: return 0.0
        return -1.0
        
    reward = 0.0
    # Penalty for wasted win
    if len(get_winning_cols(board, mark)) > 0:
        reward -= 0.5
        
    # Penalty for letting opponent win
    opp = 3 - mark
    next_board = drop_piece(board, action, mark)
    if len(get_winning_cols(next_board, opp)) > 0:
        reward -= 1.0
        
    return reward
