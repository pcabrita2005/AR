import unittest

from connect4_rl.experiments.alphazero_curriculum import (
    allocate_phase_episodes,
    build_default_alphazero_curricula,
    expand_curriculum_schedule,
)


class TestAlphaZeroCurriculum(unittest.TestCase):
    def test_allocate_phase_episodes_preserves_total(self):
        curricula = build_default_alphazero_curricula()
        definition = curricula["curriculum_mid_self"]
        allocated = allocate_phase_episodes(37, definition.phases)
        self.assertEqual(sum(count for _phase, count in allocated), 37)

    def test_expand_curriculum_schedule_matches_total_and_order(self):
        curricula = build_default_alphazero_curricula()
        definition = curricula["curriculum_basic"]
        schedule, summary = expand_curriculum_schedule(20, definition, seed=42)
        self.assertEqual(len(schedule), 20)
        self.assertEqual(summary[0]["opponent_kind"], "random")
        self.assertEqual(summary[-1]["opponent_kind"], "self_play")

    def test_probabilistic_curriculum_expands_to_allowed_opponents(self):
        curricula = build_default_alphazero_curricula()
        definition = curricula["curriculum_probabilistic_bridge"]
        schedule, summary = expand_curriculum_schedule(40, definition, seed=42)
        self.assertEqual(len(schedule), 40)
        self.assertTrue(set(schedule).issubset({"random", "heuristic", "self_play"}))
        mixed_phases = [phase for phase in summary if phase["opponent_kind"] == "mixed"]
        self.assertEqual(len(mixed_phases), 1)
        self.assertGreater(sum(mixed_phases[0]["realized_opponents"].values()), 0)


if __name__ == "__main__":
    unittest.main()
