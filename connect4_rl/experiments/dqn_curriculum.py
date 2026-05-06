from __future__ import annotations

from connect4_rl.experiments.dqn_curriculum_utils import (
    CurriculumDefinition,
    CurriculumPhase,
    RewardProfile,
    allocate_phase_episodes,
    check_vertical_win,
    count_winnable_windows,
    expand_curriculum_schedule,
    normalize_dqn_opponent_kind,
    shaped_reward,
)
from connect4_rl.experiments.dqn_training import (
    DQNTrainingMetrics as DQNCurriculumMetrics,
    train_dqn_with_curriculum,
)


def build_default_dqn_curricula() -> dict[str, CurriculumDefinition]:
    early_rewards = RewardProfile(
        win=1.0,
        vertical_win=0.7,
        three_in_row=0.05,
        opp_three_in_row=-0.05,
        lose=-1.0,
        play_continues=0.0,
    )
    mid_rewards = RewardProfile(
        win=1.0,
        vertical_win=1.0,
        three_in_row=0.02,
        opp_three_in_row=-0.02,
        lose=-1.0,
        play_continues=0.0,
    )
    late_rewards = RewardProfile(
        win=1.0,
        vertical_win=1.0,
        three_in_row=0.01,
        opp_three_in_row=-0.01,
        lose=-1.0,
        play_continues=0.0,
    )
    return {
        "curriculum_classic": CurriculumDefinition(
            name="curriculum_classic",
            description="Licoes classicas: random com shaping forte, depois weak, strong e self-play com snapshots.",
            phases=(
                CurriculumPhase(
                    "lesson1_random",
                    normalize_dqn_opponent_kind("random"),
                    0.15,
                    rewards=early_rewards,
                    buffer_warm_up=True,
                    warm_up_opponent="random",
                    warmup_replay_fill=512,
                    agent_warmup_updates=150,
                    block_vertical_bias=4.0,
                ),
                CurriculumPhase("lesson2_weak", normalize_dqn_opponent_kind("weak"), 0.20, rewards=mid_rewards),
                CurriculumPhase("lesson3_strong", normalize_dqn_opponent_kind("strong"), 0.25, rewards=mid_rewards),
                CurriculumPhase("lesson4_self_play", normalize_dqn_opponent_kind("self"), 0.40, rewards=late_rewards),
            ),
        ),
        "curriculum_self_bridge": CurriculumDefinition(
            name="curriculum_self_bridge",
            description="Vai de random para weak e entra em self-play mais cedo, guardando strong como teste intermédio.",
            phases=(
                CurriculumPhase(
                    "lesson1_random",
                    normalize_dqn_opponent_kind("random"),
                    0.15,
                    rewards=early_rewards,
                    buffer_warm_up=True,
                    warm_up_opponent="random",
                    warmup_replay_fill=512,
                    agent_warmup_updates=120,
                    block_vertical_bias=4.0,
                ),
                CurriculumPhase("lesson2_weak", normalize_dqn_opponent_kind("weak"), 0.20, rewards=mid_rewards),
                CurriculumPhase("lesson3_self_play", normalize_dqn_opponent_kind("self"), 0.40, rewards=late_rewards),
                CurriculumPhase("lesson4_strong_probe", normalize_dqn_opponent_kind("strong"), 0.25, rewards=mid_rewards),
            ),
        ),
        "curriculum_probabilistic_mix": CurriculumDefinition(
            name="curriculum_probabilistic_mix",
            description="Fase mista com weak, strong e self-play para testar uma transicao menos rigida.",
            phases=(
                CurriculumPhase(
                    "lesson1_random",
                    normalize_dqn_opponent_kind("random"),
                    0.15,
                    rewards=early_rewards,
                    buffer_warm_up=True,
                    warm_up_opponent="weak",
                    warmup_replay_fill=512,
                    agent_warmup_updates=120,
                    block_vertical_bias=3.0,
                ),
                CurriculumPhase(
                    "mixed_bridge",
                    None,
                    0.55,
                    rewards=mid_rewards,
                    opponent_weights={"weak": 0.35, "strong": 0.25, "self": 0.40},
                ),
                CurriculumPhase("self_play_finish", normalize_dqn_opponent_kind("self"), 0.30, rewards=late_rewards),
            ),
        ),
    }


__all__ = [
    "CurriculumDefinition",
    "CurriculumPhase",
    "DQNCurriculumMetrics",
    "RewardProfile",
    "allocate_phase_episodes",
    "build_default_dqn_curricula",
    "check_vertical_win",
    "count_winnable_windows",
    "expand_curriculum_schedule",
    "shaped_reward",
    "train_dqn_with_curriculum",
]
