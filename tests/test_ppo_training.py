import unittest

import torch

from connect4_rl.agents.learning.ppo import ConnectFourActorCritic
from connect4_rl.experiments.ppo_notebook_variants import VARIANT_SPECS
from connect4_rl.experiments.ppo_training import build_tutorial_ppo_lessons


class TestPPONetwork(unittest.TestCase):
    def test_tutorial_style_network_output_shape(self):
        network = ConnectFourActorCritic(
            hidden_dim=256,
            channel_sizes=[64, 128],
            kernel_sizes=[4, 3],
            stride_sizes=[1, 1],
            head_hidden_sizes=[256, 128],
        )
        batch = torch.zeros((4, 2, 6, 7), dtype=torch.float32)
        logits, values = network(batch)
        self.assertEqual(tuple(logits.shape), (4, 7))
        self.assertEqual(tuple(values.shape), (4,))

    def test_tutorial_lessons_cover_total_episode_budget(self):
        lessons = build_tutorial_ppo_lessons(180)
        self.assertEqual(sum(lesson.max_train_episodes for lesson in lessons), 180)
        self.assertEqual(
            [lesson.name for lesson in lessons],
            ["lesson1_random", "lesson2_weak", "lesson3_minimax", "lesson4_strong", "lesson5_self_play"],
        )

    def test_notebook_variants_include_baseline_and_safe_paths(self):
        self.assertIn("baseline", VARIANT_SPECS)
        self.assertIn("robust_selection", VARIANT_SPECS)
        self.assertIn("safer_self_play", VARIANT_SPECS)
        self.assertIn("final_push", VARIANT_SPECS)
        self.assertIn("final_push_midlevel_bc", VARIANT_SPECS)
        self.assertIn("final_push_minimax2_tune", VARIANT_SPECS)


if __name__ == "__main__":
    unittest.main()
