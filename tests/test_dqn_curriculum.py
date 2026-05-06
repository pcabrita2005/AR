import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from connect4_rl.envs.connect_four import ConnectFourState
from connect4_rl.experiments.dqn_training import load_dqn_lesson_definition
from connect4_rl.experiments.dqn_curriculum import (
    RewardProfile,
    allocate_phase_episodes,
    build_default_dqn_curricula,
    check_vertical_win,
    count_winnable_windows,
    expand_curriculum_schedule,
    shaped_reward,
)


class TestDQNCurriculum(unittest.TestCase):
    def test_default_dqn_lessons_load_from_yaml(self):
        definition = load_dqn_lesson_definition()
        self.assertEqual(len(definition.phases), 4)
        self.assertEqual(definition.phases[0].opponent_kind, "random")
        self.assertEqual(definition.phases[-1].opponent_kind, "self_play")

    def test_tutorial_style_yaml_aliases_are_supported(self):
        with TemporaryDirectory() as tmp_dir:
            lesson_path = Path(tmp_dir) / "lesson1.yaml"
            lesson_path.write_text(
                "\n".join(
                    [
                        "name: lesson1_random",
                        "fraction: 1.0",
                        "opponent: self",
                        "agent_warm_up: 25",
                        "block_vert_coef: 3",
                        "opponent_pool_size: 6",
                        "opponent_upgrade: 40",
                    ]
                ),
                encoding="utf-8",
            )
            definition = load_dqn_lesson_definition(tmp_dir)
        phase = definition.phases[0]
        self.assertEqual(phase.opponent_kind, "self_play")
        self.assertEqual(phase.agent_warmup_updates, 25)
        self.assertEqual(phase.block_vertical_bias, 3.0)
        self.assertEqual(phase.opponent_pool_size, 6)
        self.assertEqual(phase.opponent_upgrade_interval, 40)

    def test_allocate_phase_episodes_preserves_total(self):
        curricula = build_default_dqn_curricula()
        definition = curricula["curriculum_classic"]
        allocated = allocate_phase_episodes(37, definition.phases)
        self.assertEqual(sum(count for _phase, count in allocated), 37)

    def test_expand_curriculum_schedule_contains_expected_opponents(self):
        curricula = build_default_dqn_curricula()
        definition = curricula["curriculum_probabilistic_mix"]
        schedule, summary = expand_curriculum_schedule(40, definition, seed=42)
        self.assertEqual(len(schedule), 40)
        self.assertTrue({phase.opponent_kind for phase in schedule}.issubset({"random", "weak", "strong", "self_play"}))
        mixed_phases = [phase for phase in summary if phase["opponent_kind"] == "mixed"]
        self.assertEqual(len(mixed_phases), 1)

    def test_vertical_win_detection(self):
        state = ConnectFourState(
            board=(
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0),
            ),
            current_player=2,
            winner=1,
            moves_played=4,
            last_action=0,
        )
        self.assertTrue(check_vertical_win(state, 1))

    def test_count_winnable_windows_and_shaping(self):
        state = ConnectFourState(
            board=(
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (1, 1, 1, 0, 2, 2, 0),
            ),
            current_player=1,
            winner=0,
            moves_played=5,
            last_action=5,
        )
        rewards = RewardProfile(three_in_row=0.05, opp_three_in_row=-0.05)
        self.assertGreaterEqual(count_winnable_windows(state, 1), 1)
        self.assertGreater(shaped_reward(state, 1, rewards, done=False), 0.0)

    def test_losing_terminal_reward_uses_profile(self):
        state = ConnectFourState(
            board=(
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0),
                (2, 2, 2, 2, 0, 0, 0),
            ),
            current_player=1,
            winner=2,
            moves_played=4,
            last_action=3,
        )
        rewards = RewardProfile(lose=-1.0)
        self.assertEqual(shaped_reward(state, 1, rewards, done=True), -1.0)


if __name__ == "__main__":
    unittest.main()
