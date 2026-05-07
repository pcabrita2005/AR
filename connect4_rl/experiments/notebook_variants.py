from __future__ import annotations

import copy
from pathlib import Path

import yaml

from connect4_rl.config import Config
from connect4_rl.experiments.dqn_training import default_dqn_lessons_dir


VARIANT_SPECS: dict[str, dict[str, object]] = {
    "baseline": {
        "description": "Configuração atual usada como referência.",
        "config_overrides": {},
        "lesson_overrides": {},
    },
    "strong_focus": {
        "description": "Dá ainda mais peso à lesson3_strong e encurta a lesson4_self_play.",
        "config_overrides": {
            "dqn": {
                "self_play_early_stop_patience_evals": 2,
            }
        },
        "lesson_overrides": {
            "lesson3.yaml": {"max_train_episodes": 800000},
            "lesson4.yaml": {"max_train_episodes": 200000},
        },
    },
    "safer_self_play": {
        "description": "Torna a lesson4_self_play ainda mais conservadora para preservar o pico da lesson3.",
        "config_overrides": {
            "dqn": {
                "random_opponent_fraction": 0.0,
                "heuristic_opponent_fraction": 0.60,
                "self_play_early_stop_patience_evals": 2,
            }
        },
        "lesson_overrides": {
            "lesson4.yaml": {
                "learning_rate_scale": 0.15,
                "epsilon_start": 0.10,
                "epsilon_end": 0.01,
            }
        },
    },
    "robust_selection": {
        "description": "Aumenta a robustez da escolha de checkpoints com avaliação mais forte.",
        "config_overrides": {
            "dqn": {
                "eval_games": 48,
                "checkpoint_score_heuristic_weight": 6.5,
                "random_opponent_fraction": 0.0,
                "heuristic_opponent_fraction": 0.55,
                "self_play_early_stop_patience_evals": 2,
            }
        },
        "lesson_overrides": {
            "lesson4.yaml": {
                "learning_rate_scale": 0.20,
                "epsilon_start": 0.12,
                "epsilon_end": 0.01,
            }
        },
    },
}


def apply_variant_to_config(config: Config, variant_name: str) -> Config:
    variant = get_variant_spec(variant_name)
    updated = copy.deepcopy(config)
    dqn_overrides = variant.get("config_overrides", {}).get("dqn", {})
    for key, value in dqn_overrides.items():
        setattr(updated.dqn, key, value)
    return updated


def prepare_variant_lessons_dir(root: Path, variant_name: str) -> Path:
    variant = get_variant_spec(variant_name)
    target_dir = root / "tmp" / "dqn_lessons_variants" / variant_name
    target_dir.mkdir(parents=True, exist_ok=True)

    base_dir = default_dqn_lessons_dir()
    lesson_overrides: dict[str, dict[str, object]] = variant.get("lesson_overrides", {})

    for lesson_path in sorted(base_dir.glob("lesson*.yaml")):
        data = yaml.safe_load(lesson_path.read_text(encoding="utf-8")) or {}
        overrides = lesson_overrides.get(lesson_path.name, {})
        data.update(overrides)
        output_path = target_dir / lesson_path.name
        output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    return target_dir


def get_variant_spec(variant_name: str) -> dict[str, object]:
    if variant_name not in VARIANT_SPECS:
        raise KeyError(f"Unknown notebook DQN variant '{variant_name}'. Available: {sorted(VARIANT_SPECS)}")
    return VARIANT_SPECS[variant_name]
