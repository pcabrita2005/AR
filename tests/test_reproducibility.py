import unittest
import numpy as np
import torch
from connect4_rl.utils.seed_utils import set_all_seeds


class TestSeedReproducibility(unittest.TestCase):
    """Test that seeds ensure reproducible random behavior."""

    def test_numpy_seed_reproducibility(self):
        set_all_seeds(42)
        arr1 = np.random.randn(10)
        
        set_all_seeds(42)
        arr2 = np.random.randn(10)
        
        np.testing.assert_array_equal(arr1, arr2, "Same numpy seed should produce same arrays")

    def test_torch_seed_reproducibility(self):
        set_all_seeds(42)
        tensor1 = torch.randn(10)
        
        set_all_seeds(42)
        tensor2 = torch.randn(10)
        
        torch.testing.assert_close(tensor1, tensor2, msg="Same torch seed should produce same tensors")

    def test_python_random_seed_reproducibility(self):
        import random
        
        set_all_seeds(42)
        values1 = [random.random() for _ in range(10)]
        
        set_all_seeds(42)
        values2 = [random.random() for _ in range(10)]
        
        self.assertEqual(values1, values2, "Same seed should produce same random.random() values")

    def test_different_seeds_different_output(self):
        set_all_seeds(42)
        arr1 = np.random.randn(10)
        
        set_all_seeds(99)
        arr2 = np.random.randn(10)
        
        self.assertFalse(np.allclose(arr1, arr2), "Different seeds should produce different arrays")


class TestConfigReproducibility(unittest.TestCase):

    def test_full_experiment_reproducibility(self):
        from connect4_rl.agents.baselines import RandomAgent
        from connect4_rl.envs.connect_four import (
            initial_state,
            apply_action,
            is_terminal,
            legal_actions,
        )

        def play_game(seed):
            set_all_seeds(seed)
            agent = RandomAgent(seed=seed)
            state = initial_state()
            moves = []
            
            while not is_terminal(state):
                legal = legal_actions(state)
                action = agent.select_action(state, legal)
                moves.append(action)
                state = apply_action(state, action)
            
            return moves

        # A mesma seed deve produzir o mesmo jogo
        game1 = play_game(42)
        game2 = play_game(42)
        self.assertEqual(game1, game2, "Same seed should produce identical games")
        
        # Seeds diferentes devem produzir jogos diferentes
        game3 = play_game(99)
        self.assertNotEqual(game1, game3, "Different seeds should produce different games")


if __name__ == "__main__":
    unittest.main()
