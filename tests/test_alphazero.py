import random
import unittest

import numpy as np
import torch

from connect4_rl.agents.learning.alphazero import ConnectFourPolicyValueNet, encode_alphazero_state, run_policy_value_mcts
from connect4_rl.config import AlphaZeroConfig
from connect4_rl.experiments.alphazero_training import get_training_mcts_simulations
from connect4_rl.envs.connect_four import apply_action, initial_state, legal_actions


class TestAlphaZeroEncoding(unittest.TestCase):
    def test_encoder_returns_three_binary_planes(self):
        state = initial_state()
        state = apply_action(state, 3)
        encoded = encode_alphazero_state(state, perspective_player=state.current_player)

        self.assertEqual(encoded.shape, (3, 6, 7))
        self.assertTrue(np.all((encoded == 0.0) | (encoded == 1.0)))
        self.assertTrue(np.allclose(encoded.sum(axis=0), 1.0))


class TestAlphaZeroNetwork(unittest.TestCase):
    def test_network_output_shapes(self):
        network = ConnectFourPolicyValueNet(n_filters=64, n_res_blocks=2)
        batch = torch.zeros((4, 3, 6, 7), dtype=torch.float32)
        logits, values = network(batch)

        self.assertEqual(tuple(logits.shape), (4, 7))
        self.assertEqual(tuple(values.shape), (4,))
        self.assertTrue(torch.all(values <= 1.0))
        self.assertTrue(torch.all(values >= -1.0))


class TestAlphaZeroMCTS(unittest.TestCase):
    def test_mcts_policy_respects_legal_actions(self):
        network = ConnectFourPolicyValueNet(n_filters=64, n_res_blocks=1)
        state = initial_state()
        for action in [0, 0, 0, 0, 0, 0]:
            state = apply_action(state, action)

        policy = run_policy_value_mcts(
            network,
            state,
            simulations=8,
            c_puct=1.5,
            device="cpu",
            root_dirichlet_alpha=None,
            root_dirichlet_epsilon=0.0,
            rng=random.Random(0),
        )
        legal = set(legal_actions(state))

        self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
        for action, probability in enumerate(policy):
            if action not in legal:
                self.assertEqual(float(probability), 0.0)


class TestAlphaZeroMCTSSchedule(unittest.TestCase):
    def test_training_simulations_follow_progressive_schedule(self):
        config = AlphaZeroConfig(
            mcts_simulations=120,
            mcts_start_search_iter=30,
            mcts_max_search_iter=35,
            mcts_search_increment=2,
        )
        self.assertEqual(get_training_mcts_simulations(config, 1), 30)
        self.assertEqual(get_training_mcts_simulations(config, 2), 32)
        self.assertEqual(get_training_mcts_simulations(config, 3), 34)
        self.assertEqual(get_training_mcts_simulations(config, 4), 35)
        self.assertEqual(get_training_mcts_simulations(config, 10), 35)

    def test_training_simulations_fall_back_to_fixed_value(self):
        config = AlphaZeroConfig(mcts_simulations=120, mcts_start_search_iter=None)
        self.assertEqual(get_training_mcts_simulations(config, 1), 120)
        self.assertEqual(get_training_mcts_simulations(config, 50), 120)


if __name__ == "__main__":
    unittest.main()
