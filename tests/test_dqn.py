import unittest

import torch

from connect4_rl.agents.learning.dqn import ConnectFourQNetwork


class TestDQNNetwork(unittest.TestCase):
    def test_tutorial_style_network_output_shape(self):
        network = ConnectFourQNetwork(
            channel_sizes=[128],
            kernel_sizes=[4],
            stride_sizes=[1],
            head_hidden_sizes=[64, 64],
        )
        batch = torch.zeros((4, 2, 6, 7), dtype=torch.float32)
        q_values = network(batch)
        self.assertEqual(tuple(q_values.shape), (4, 7))


if __name__ == "__main__":
    unittest.main()
