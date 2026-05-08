from __future__ import annotations

import copy

from connect4_rl.config import Config


VARIANT_SPECS: dict[str, dict[str, object]] = {
    "baseline": {
        "description": "Configuracao tutorial-style base para o PPO.",
        "config_overrides": {},
    },
    "robust_selection": {
        "description": "Aumenta a robustez da escolha de checkpoints com mais peso no adversario forte.",
        "config_overrides": {
            "ppo": {
                "eval_games": 32,
                "checkpoint_score_heuristic_weight": 6.0,
                "rollout_episodes_per_update": 14,
            }
        },
    },
    "safer_self_play": {
        "description": "Torna a ultima fase mais conservadora para preservar progresso contra o heuristico forte.",
        "config_overrides": {
            "ppo": {
                "learning_rate": 1.0e-4,
                "entropy_coeff": 0.01,
                "random_opponent_fraction": 0.0,
                "heuristic_opponent_fraction": 0.40,
            }
        },
    },
    "wider_policy": {
        "description": "Aumenta a capacidade do ator-critico para testar se a rede limita a performance.",
        "config_overrides": {
            "ppo": {
                "hidden_dim": 320,
                "channel_sizes": [64, 128, 128],
                "kernel_sizes": [3, 3, 2],
                "stride_sizes": [1, 1, 1],
                "head_hidden_sizes": [320, 160],
            }
        },
    },
    "anti_strong": {
        "description": "Foca o PPO em sobrevivencia tatico-posicional contra a heuristica forte.",
        "config_overrides": {
            "ppo": {
                "learning_rate": 8.0e-5,
                "entropy_coeff": 0.004,
                "eval_games": 40,
                "heuristic_opponent_fraction": 0.50,
                "threat_bonus_scale": 0.50,
                "opponent_threat_penalty_scale": 0.85,
                "blocked_threat_bonus_scale": 1.00,
                "allowed_threat_penalty_scale": 1.40,
                "center_control_scale": 0.12,
                "imitation_loss_coeff": 0.70,
            }
        },
    },
    "wider_safer_strong": {
        "description": "Combina rede maior com PPO mais conservador para preservar o pico contra o strong.",
        "config_overrides": {
            "ppo": {
                "hidden_dim": 320,
                "channel_sizes": [64, 128, 128],
                "kernel_sizes": [3, 3, 2],
                "stride_sizes": [1, 1, 1],
                "head_hidden_sizes": [320, 160],
                "learning_rate": 7.0e-5,
                "entropy_coeff": 0.003,
                "eval_games": 40,
                "checkpoint_score_heuristic_weight": 6.5,
                "rollout_episodes_per_update": 10,
                "heuristic_opponent_fraction": 0.50,
                "threat_bonus_scale": 0.45,
                "opponent_threat_penalty_scale": 0.90,
                "blocked_threat_bonus_scale": 1.10,
                "allowed_threat_penalty_scale": 1.50,
                "center_control_scale": 0.12,
                "imitation_loss_coeff": 0.80,
                "self_play_min_episodes_before_early_stop": 40,
                "self_play_early_stop_patience_evals": 2,
            }
        },
    },
    "final_push": {
        "description": "Perfil mais agressivo para fechar um PPO final: bootstrap mais forte, foco longo no strong e endgame curto.",
        "config_overrides": {
            "ppo": {
                "curriculum_profile": "final_push",
                "hidden_dim": 320,
                "channel_sizes": [64, 128, 128],
                "kernel_sizes": [3, 3, 2],
                "stride_sizes": [1, 1, 1],
                "head_hidden_sizes": [320, 160],
                "learning_rate": 1.0e-4,
                "entropy_coeff": 0.01,
                "n_epochs": 8,
                "eval_games": 32,
                "bootstrap_samples": 24000,
                "bootstrap_epochs": 12,
                "bootstrap_learning_rate": 3.0e-4,
                "rollout_episodes_per_update": 14,
                "imitation_loss_coeff": 0.55,
                "checkpoint_score_heuristic_weight": 6.0,
                "freeze_feature_extractor_lessons": 4,
                "self_play_min_episodes_before_early_stop": 30,
                "self_play_early_stop_patience_evals": 2,
            }
        },
    },
    "final_push_midlevel_bc": {
        "description": "Fecha o PPO a partir de um pretreino supervisionado mais forte com teacher minimax_1 e extractor congelado.",
        "config_overrides": {
            "ppo": {
                "curriculum_profile": "final_push",
                "hidden_dim": 320,
                "channel_sizes": [64, 128, 128],
                "kernel_sizes": [3, 3, 2],
                "stride_sizes": [1, 1, 1],
                "head_hidden_sizes": [320, 160],
                "learning_rate": 1.0e-4,
                "entropy_coeff": 0.01,
                "n_epochs": 8,
                "eval_games": 32,
                "bootstrap_samples": 60000,
                "bootstrap_epochs": 10,
                "bootstrap_learning_rate": 3.0e-4,
                "bootstrap_teacher_kind": "minimax_1",
                "rollout_episodes_per_update": 14,
                "imitation_loss_coeff": 0.55,
                "checkpoint_score_heuristic_weight": 6.0,
                "freeze_feature_extractor_lessons": 5,
                "self_play_min_episodes_before_early_stop": 30,
                "self_play_early_stop_patience_evals": 2,
            }
        },
    },
    "final_push_minimax2_tune": {
        "description": "Ultima linha: preserva a base supervisionada forte e abre o extractor nas fases finais para atacar minimax_2.",
        "config_overrides": {
            "ppo": {
                "curriculum_profile": "final_push_hard_bridge",
                "hidden_dim": 320,
                "channel_sizes": [64, 128, 128],
                "kernel_sizes": [3, 3, 2],
                "stride_sizes": [1, 1, 1],
                "head_hidden_sizes": [320, 160],
                "learning_rate": 8.0e-5,
                "entropy_coeff": 0.008,
                "n_epochs": 8,
                "eval_games": 32,
                "bootstrap_samples": 60000,
                "bootstrap_epochs": 10,
                "bootstrap_learning_rate": 3.0e-4,
                "bootstrap_teacher_kind": "minimax_1",
                "rollout_episodes_per_update": 14,
                "imitation_loss_coeff": 0.50,
                "checkpoint_score_heuristic_weight": 6.0,
                "freeze_feature_extractor_lessons": 3,
                "self_play_min_episodes_before_early_stop": 30,
                "self_play_early_stop_patience_evals": 2,
            }
        },
    },
}


def apply_variant_to_config(config: Config, variant_name: str) -> Config:
    variant = get_variant_spec(variant_name)
    updated = copy.deepcopy(config)
    ppo_overrides = variant.get("config_overrides", {}).get("ppo", {})
    for key, value in ppo_overrides.items():
        setattr(updated.ppo, key, value)
    return updated


def get_variant_spec(variant_name: str) -> dict[str, object]:
    if variant_name not in VARIANT_SPECS:
        raise KeyError(f"Unknown notebook PPO variant '{variant_name}'. Available: {sorted(VARIANT_SPECS)}")
    return VARIANT_SPECS[variant_name]
