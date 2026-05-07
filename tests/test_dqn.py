import unittest

import torch

from connect4_rl.agents.learning.dqn import ConnectFourQNetwork
from connect4_rl.experiments.dqn_training import build_tutorial_dqn_lessons


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

    def test_tutorial_lessons_cover_total_episode_budget(self):
        lessons = build_tutorial_dqn_lessons(180)
        self.assertEqual(sum(lesson.max_train_episodes for lesson in lessons), 180)
        self.assertEqual([lesson.name for lesson in lessons], [
            "lesson1_random",
            "lesson2_weak",
            "lesson3_strong",
            "lesson4_self_play",
        ])


if __name__ == "__main__":
    unittest.main()
