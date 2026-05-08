from __future__ import annotations

import json
from pathlib import Path


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in source.splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in source.splitlines()],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_training_notebook() -> dict:
    cells = [
        md_cell(
            "# PPO Tutorial-Style Training\n\n"
            "Este notebook e o cockpit de treino e experimentacao do `PPO`. "
            "Segue agora a mesma logica do `DQN`: treino estruturado no codigo e notebook focado em iterar variantes, correr treino e analisar a run ativa. "
            "Para apresentar a melhor versao, usa o `05_ppo_best_model_showcase.ipynb`."
        ),
        md_cell("## Passo 1: Setup"),
        code_cell(
            "from __future__ import annotations\n\n"
            "import copy\n"
            "import json\n"
            "import statistics\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import torch\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "for candidate in [ROOT, *ROOT.parents]:\n"
            "    if (candidate / 'connect4_rl').exists():\n"
            "        ROOT = candidate\n"
            "        break\n"
            "else:\n"
            "    raise RuntimeError('Nao encontrei a raiz do repositorio com a pasta connect4_rl.')\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from connect4_rl.agents.baselines import MinimaxAgent, RandomAgent, StrongHeuristicAgent, WeakHeuristicAgent\n"
            "from connect4_rl.config import load_config\n"
            "from connect4_rl.experiments import build_agent_from_run, evaluate_against_agent, find_best_run\n"
            "from connect4_rl.experiments.ppo_notebook_variants import VARIANT_SPECS, apply_variant_to_config\n"
            "from connect4_rl.experiments.ppo_training import build_tutorial_ppo_lessons, evaluate_match_summary, train_ppo_self_play\n"
        ),
        md_cell(
            "## Passo 2: Configuracao do treino\n\n"
            "O perfil `quick` usa o budget de notebook para iteracao rapida. "
            "O perfil `full` aproxima-se mais do regime final. "
            "O campo `experiment_variant` permite testar hipoteses controladas sem mexer no codigo base."
        ),
        code_cell(
            "NOTEBOOK_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "training_profile = 'quick'  # 'quick' ou 'full'\n"
            "experiment_variant = 'baseline'  # " + ", ".join(f"'{name}'" for name in ["baseline", "robust_selection", "safer_self_play", "wider_policy", "anti_strong", "wider_safer_strong", "final_push", "final_push_midlevel_bc", "final_push_minimax2_tune"]) + "\n"
            "seed = 42\n"
            "run_training = True\n\n"
            "CONFIG = load_config(str(ROOT / 'config.yaml'))\n"
            "training_config = apply_variant_to_config(copy.deepcopy(CONFIG), experiment_variant)\n"
            "training_config.global_.seed = seed\n\n"
            "if training_profile == 'quick':\n"
            "    training_config.ppo.episodes = CONFIG.notebook_settings.quick_test.episodes\n"
            "    training_config.ppo.eval_interval = CONFIG.notebook_settings.quick_test.eval_interval\n"
            "    training_config.ppo.eval_games = CONFIG.notebook_settings.quick_test.eval_games\n"
            "else:\n"
            "    training_config.ppo.episodes = max(training_config.ppo.episodes, 720)\n"
            "    training_config.ppo.eval_interval = 30\n"
            "    training_config.ppo.eval_games = 24\n\n"
            "variant_spec = VARIANT_SPECS[experiment_variant]\n"
            "run_name = f'ppo_tutorial_{training_profile}_{experiment_variant}_seed_{seed}'\n"
            "OUTPUTS = ROOT / 'notebooks' / 'ppo' / 'outputs'\n"
            "OUTPUTS.mkdir(parents=True, exist_ok=True)\n"
            "checkpoint_dir = OUTPUTS / run_name\n"
            "lessons = build_tutorial_ppo_lessons(training_config.ppo.episodes, profile=training_config.ppo.curriculum_profile)\n"
            "training_plan = {\n"
            "    'run_name': run_name,\n"
            "    'training_profile': training_profile,\n"
            "    'experiment_variant': experiment_variant,\n"
            "    'variant_description': variant_spec['description'],\n"
            "    'episodes': training_config.ppo.episodes,\n"
            "    'eval_interval': training_config.ppo.eval_interval,\n"
            "    'eval_games': training_config.ppo.eval_games,\n"
            "    'lessons': [lesson.name for lesson in lessons],\n"
            "    'checkpoint_dir': str(checkpoint_dir),\n"
            "}\n"
            "training_plan\n"
        ),
        md_cell("## Passo 3: Treino do pipeline PPO"),
        code_cell(
            "trained_agent = None\n"
            "trained_metrics = None\n\n"
            "if run_training:\n"
            "    trained_agent, trained_metrics = train_ppo_self_play(training_config, checkpoint_dir=checkpoint_dir)\n"
            "    {\n"
            "        'curriculum_name': trained_metrics.curriculum_name,\n"
            "        'experiment_variant': experiment_variant,\n"
            "        'bootstrap_summary': trained_metrics.bootstrap_summary,\n"
            "        'lessons': [item['lesson_name'] for item in trained_metrics.lesson_summaries],\n"
            "        'final_checkpoint_path': trained_metrics.best_checkpoint_path,\n"
            "        'focus_checkpoint_path': trained_metrics.best_vs_strong_checkpoint_path,\n"
            "        'last_eval': trained_metrics.evaluation[-1] if trained_metrics.evaluation else {},\n"
            "        'num_policy_updates': len(trained_metrics.policy_losses),\n"
            "    }\n"
            "else:\n"
            "    print('Treino nao executado nesta sessao. O notebook vai tentar carregar a melhor run guardada.')\n"
        ),
        md_cell("## Passo 4: Escolher a run ativa"),
        code_cell(
            "active_metrics = trained_metrics\n"
            "active_agent = trained_agent\n"
            "active_run_name = run_name if run_training else None\n"
            "focus_checkpoint_path = None\n\n"
            "if active_metrics is not None:\n"
            "    active_run_name = run_name\n"
            "else:\n"
            "    best_run = find_best_run(OUTPUTS, 'ppo')\n"
            "    if best_run is None:\n"
            "        raise RuntimeError('Nao encontrei runs PPO em outputs/. Corre primeiro o treino neste notebook.')\n"
            "    active_run_name = best_run.metrics_path.parent.name\n"
            "    active_metrics = type('MetricsProxy', (), best_run.data)()\n"
            "    active_agent = build_agent_from_run(best_run, root=ROOT, device=NOTEBOOK_DEVICE)\n\n"
            "focus_checkpoint_path = getattr(active_metrics, 'best_vs_strong_checkpoint_path', '') or getattr(active_metrics, 'best_checkpoint_path', '')\n"
            "lesson_summaries = getattr(active_metrics, 'lesson_summaries', [])\n"
            "evaluation = getattr(active_metrics, 'evaluation', [])\n"
            "episode_rewards = getattr(active_metrics, 'episode_rewards', [])\n\n"
            "print({\n"
            "    'run_name': active_run_name,\n"
            "    'bootstrap_summary': getattr(active_metrics, 'bootstrap_summary', {}),\n"
            "    'lessons': [item['lesson_name'] for item in lesson_summaries],\n"
            "    'num_evaluations': len(evaluation),\n"
            "    'focus_checkpoint_path': focus_checkpoint_path,\n"
            "    'best_vs_strong_win_rate': round(float(getattr(active_metrics, 'best_vs_strong_win_rate', 0.0)), 4),\n"
            "    'best_vs_strong_draw_rate': round(float(getattr(active_metrics, 'best_vs_strong_draw_rate', 0.0)), 4),\n"
            "    'reward_first_20_mean': round(sum(episode_rewards[:20]) / max(len(episode_rewards[:20]), 1), 4),\n"
            "    'reward_last_20_mean': round(sum(episode_rewards[-20:]) / max(len(episode_rewards[-20:]), 1), 4),\n"
            "})\n"
            "lesson_summaries\n"
        ),
        md_cell("## Passo 5: Curvas de avaliacao"),
        code_cell(
            "evaluation = list(getattr(active_metrics, 'evaluation', []))\n"
            "phase_summary = getattr(active_metrics, 'phase_summary', [])\n"
            "policy_losses = [float(value) for value in getattr(active_metrics, 'policy_losses', [])]\n"
            "value_losses = [float(value) for value in getattr(active_metrics, 'value_losses', [])]\n"
            "entropies = [float(value) for value in getattr(active_metrics, 'entropies', [])]\n"
            "rewards = [float(value) for value in getattr(active_metrics, 'episode_rewards', [])]\n\n"
            "def moving_average(values, window):\n"
            "    if len(values) < window:\n"
            "        return values\n"
            "    return [statistics.fmean(values[max(0, idx - window + 1): idx + 1]) for idx in range(len(values))]\n\n"
            "if evaluation:\n"
            "    eval_episodes = [int(item['episode']) for item in evaluation]\n"
            "    eval_outcome = [float(item.get('eval_mean_outcome', 0.0)) for item in evaluation]\n"
            "    vs_random = [float(item.get('vs_random_win_rate', 0.0)) for item in evaluation]\n"
            "    vs_weak = [float(item.get('vs_weak_heuristic_win_rate', 0.0)) for item in evaluation]\n"
            "    vs_strong = [float(item.get('vs_strong_heuristic_win_rate', 0.0)) for item in evaluation]\n"
            "    vs_minimax_1 = [float(item.get('vs_minimax_1_win_rate', 0.0)) for item in evaluation]\n"
            "    vs_minimax_2 = [float(item.get('vs_minimax_2_win_rate', 0.0)) for item in evaluation]\n"
            "    vs_previous = [float(item.get('vs_previous_win_rate', 0.0)) for item in evaluation]\n"
            "    best_vs_strong_idx = max(range(len(vs_strong)), key=lambda idx: vs_strong[idx])\n"
            "    print('Best vs strong checkpoint')\n"
            "    print({\n"
            "        'episode': eval_episodes[best_vs_strong_idx],\n"
            "        'lesson': evaluation[best_vs_strong_idx].get('lesson_name'),\n"
            "        'vs_strong': round(vs_strong[best_vs_strong_idx], 4),\n"
            "        'vs_strong_draw': round(float(evaluation[best_vs_strong_idx].get('vs_strong_draw_rate', 0.0)), 4),\n"
            "        'eval_mean_outcome': round(eval_outcome[best_vs_strong_idx], 4),\n"
            "        'vs_random': round(vs_random[best_vs_strong_idx], 4),\n"
            "        'vs_weak': round(vs_weak[best_vs_strong_idx], 4),\n"
            "        'vs_minimax_1': round(vs_minimax_1[best_vs_strong_idx], 4),\n"
            "        'vs_minimax_2': round(vs_minimax_2[best_vs_strong_idx], 4),\n"
            "        'vs_previous': round(vs_previous[best_vs_strong_idx], 4),\n"
            "    })\n\n"
            "    print('Evaluation checkpoints')\n"
            "    for item in evaluation:\n"
            "        print({\n"
            "            'episode': int(item['episode']),\n"
            "            'lesson': item.get('lesson_name'),\n"
            "            'eval_mean_outcome': round(float(item.get('eval_mean_outcome', 0.0)), 4),\n"
            "            'vs_random': round(float(item.get('vs_random_win_rate', 0.0)), 4),\n"
            "            'vs_weak': round(float(item.get('vs_weak_heuristic_win_rate', 0.0)), 4),\n"
            "            'vs_minimax_1': round(float(item.get('vs_minimax_1_win_rate', 0.0)), 4),\n"
            "            'vs_minimax_2': round(float(item.get('vs_minimax_2_win_rate', 0.0)), 4),\n"
            "            'vs_strong': round(float(item.get('vs_strong_heuristic_win_rate', 0.0)), 4),\n"
            "            'vs_strong_draw': round(float(item.get('vs_strong_draw_rate', 0.0)), 4),\n"
            "            'vs_previous': round(float(item.get('vs_previous_win_rate', 0.0)), 4),\n"
            "        })\n\n"
            "    print('Lesson summaries')\n"
            "    for item in lesson_summaries:\n"
            "        print({\n"
            "            'lesson': item.get('lesson_name'),\n"
            "            'best_score': round(float(item.get('best_score', 0.0)), 4),\n"
            "            'best_vs_strong': round(float(item.get('best_vs_strong_win_rate', 0.0)), 4),\n"
            "            'best_vs_strong_draw': round(float(item.get('best_vs_strong_draw_rate', 0.0)), 4),\n"
            "        })\n\n"
            "    fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
            "    axes[0, 0].plot(eval_episodes, vs_random, marker='o', label='vs random')\n"
            "    axes[0, 0].plot(eval_episodes, vs_weak, marker='o', label='vs weak')\n"
            "    axes[0, 0].plot(eval_episodes, vs_minimax_1, marker='o', label='vs minimax_1')\n"
            "    axes[0, 0].plot(eval_episodes, vs_minimax_2, marker='o', label='vs minimax_2')\n"
            "    axes[0, 0].plot(eval_episodes, vs_strong, marker='o', label='vs strong')\n"
            "    axes[0, 0].scatter([eval_episodes[best_vs_strong_idx]], [vs_strong[best_vs_strong_idx]], color='red', s=80, label='best vs strong')\n"
            "    axes[0, 0].set_title('Win rate por checkpoint')\n"
            "    axes[0, 0].set_xlabel('Episode')\n"
            "    axes[0, 0].set_ylabel('Win rate')\n"
            "    axes[0, 0].legend()\n"
            "    axes[0, 0].grid(alpha=0.3)\n\n"
            "    axes[0, 1].plot(eval_episodes, eval_outcome, marker='o', color='purple', label='eval mean outcome')\n"
            "    axes[0, 1].plot(eval_episodes, vs_previous, marker='o', color='gray', label='vs previous')\n"
            "    axes[0, 1].set_title('Outcome e estabilidade')\n"
            "    axes[0, 1].set_xlabel('Episode')\n"
            "    axes[0, 1].legend()\n"
            "    axes[0, 1].grid(alpha=0.3)\n\n"
            "    axes[1, 0].plot(rewards, alpha=0.35, label='Reward por episodio')\n"
            "    axes[1, 0].plot(moving_average(rewards, 20), linewidth=2, label='Media movel (20)')\n"
            "    axes[1, 0].set_title('Recompensa de treino')\n"
            "    axes[1, 0].set_xlabel('Episode')\n"
            "    axes[1, 0].legend()\n"
            "    axes[1, 0].grid(alpha=0.3)\n\n"
            "    axes[1, 1].plot(policy_losses, label='policy loss')\n"
            "    axes[1, 1].plot(value_losses, label='value loss')\n"
            "    if entropies:\n"
            "        axes[1, 1].plot(entropies, label='entropy')\n"
            "    axes[1, 1].set_title('Sinais de otimizacao')\n"
            "    axes[1, 1].set_xlabel('Update step')\n"
            "    axes[1, 1].legend()\n"
            "    axes[1, 1].grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "else:\n"
            "    print('Ainda nao ha checkpoints de avaliacao para mostrar.')\n"
        ),
        md_cell("## Passo 6: Avaliacao final contra referencias"),
        code_cell(
            "final_eval = {\n"
            "    'vs_random': evaluate_match_summary(active_agent, lambda game_idx: RandomAgent(seed=10_000 + game_idx), games=80),\n"
            "    'vs_weak': evaluate_match_summary(active_agent, lambda game_idx: WeakHeuristicAgent(seed=20_000 + game_idx), games=80),\n"
            "    'vs_minimax_1': evaluate_match_summary(active_agent, lambda game_idx: MinimaxAgent(depth=1, seed=25_000 + game_idx), games=80),\n"
            "    'vs_minimax_2': evaluate_match_summary(active_agent, lambda game_idx: MinimaxAgent(depth=2, seed=27_000 + game_idx), games=80),\n"
            "    'vs_strong': evaluate_match_summary(active_agent, lambda game_idx: StrongHeuristicAgent(seed=30_000 + game_idx), games=80),\n"
            "}\n"
            "final_eval\n"
        ),
    ]
    return notebook(cells)


def build_showcase_notebook() -> dict:
    cells = [
        md_cell(
            "# Best PPO Showcase\n\n"
            "Este notebook assume que ja tens pelo menos uma run `PPO` em `outputs/`. "
            "O objetivo e carregar o melhor checkpoint, resumir a run e mostrar uma avaliacao final mais estavel."
        ),
        md_cell("## Passo 1: Setup"),
        code_cell(
            "from __future__ import annotations\n\n"
            "import json\n"
            "import statistics\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import torch\n\n"
            "ROOT = Path.cwd().resolve()\n"
            "for candidate in [ROOT, *ROOT.parents]:\n"
            "    if (candidate / 'connect4_rl').exists():\n"
            "        ROOT = candidate\n"
            "        break\n"
            "else:\n"
            "    raise RuntimeError('Nao encontrei a raiz do repositorio com a pasta connect4_rl.')\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from connect4_rl.agents.baselines import MinimaxAgent, RandomAgent, StrongHeuristicAgent, WeakHeuristicAgent\n"
            "from connect4_rl.experiments import build_agent_from_run, find_best_run\n"
            "from connect4_rl.experiments.checkpoints import build_agent_from_checkpoint\n"
            "from connect4_rl.experiments.ppo_training import evaluate_match_summary\n"
        ),
        md_cell("## Passo 2: Escolher a run"),
        code_cell(
            "OUTPUTS = ROOT / 'notebooks' / 'ppo' / 'outputs'\n"
            "OUTPUTS.mkdir(parents=True, exist_ok=True)\n"
            "NOTEBOOK_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "preferred_run_name = 'ppo_tutorial_full_final_push_midlevel_bc_seed_42'\n"
            "preferred_metrics = OUTPUTS / preferred_run_name / 'metrics_final.json'\n"
            "if preferred_metrics.exists():\n"
            "    metrics_data = json.loads(preferred_metrics.read_text(encoding='utf-8'))\n"
            "    focus_checkpoint_path = metrics_data.get('best_vs_strong_checkpoint_path') or metrics_data.get('best_checkpoint_path')\n"
            "    agent = build_agent_from_checkpoint('ppo', ROOT / focus_checkpoint_path, dict(metrics_data.get('config', {})), device=NOTEBOOK_DEVICE)\n"
            "    run_name = preferred_run_name\n"
            "else:\n"
            "    best_run = find_best_run(OUTPUTS, 'ppo')\n"
            "    if best_run is None:\n"
            "        raise RuntimeError('Nao encontrei runs PPO em outputs/. Corre primeiro o notebook 04.')\n\n"
            "    metrics_data = dict(best_run.data)\n"
            "    agent = build_agent_from_run(best_run, root=ROOT, device=NOTEBOOK_DEVICE)\n"
            "    focus_checkpoint_path = metrics_data.get('best_vs_strong_checkpoint_path') or metrics_data.get('best_checkpoint_path')\n"
            "    run_name = best_run.metrics_path.parent.name\n"
            "{\n"
            "    'run_name': run_name,\n"
            "    'focus_checkpoint_path': focus_checkpoint_path,\n"
            "    'best_score': metrics_data.get('best_score'),\n"
            "    'best_vs_strong_win_rate': metrics_data.get('best_vs_strong_win_rate'),\n"
            "    'best_vs_strong_draw_rate': metrics_data.get('best_vs_strong_draw_rate'),\n"
            "}\n"
        ),
        md_cell("## Passo 3: Resumo da run"),
        code_cell(
            "summary = {\n"
            "    'lesson_summaries': metrics_data.get('lesson_summaries', []),\n"
            "    'num_evaluations': len(metrics_data.get('evaluation', [])),\n"
            "    'best_checkpoint_path': metrics_data.get('best_checkpoint_path'),\n"
            "    'best_vs_strong_checkpoint_path': metrics_data.get('best_vs_strong_checkpoint_path'),\n"
            "}\n"
            "summary\n"
        ),
        md_cell("## Passo 4: Tabela de avaliacoes"),
        code_cell(
            "evaluation = metrics_data.get('evaluation', [])\n"
            "evaluation\n"
        ),
        md_cell("## Passo 5: Curvas de avaliacao"),
        code_cell(
            "rewards = [float(value) for value in metrics_data.get('episode_rewards', [])]\n\n"
            "def rolling_mean(values, window=20):\n"
            "    if len(values) < window:\n"
            "        return values\n"
            "    return [statistics.fmean(values[max(0, idx - window + 1): idx + 1]) for idx in range(len(values))]\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n"
            "axes[0].plot(rewards, color='#7f5539', alpha=0.3)\n"
            "axes[0].plot(rolling_mean(rewards, 20), color='#386641', linewidth=2)\n"
            "axes[0].set_title('Recompensa por episodio')\n\n"
            "if evaluation:\n"
            "    eval_x = [item['episode'] for item in evaluation]\n"
            "    axes[1].plot(eval_x, [item.get('vs_random_win_rate', 0.0) for item in evaluation], label='vs random', color='#2a9d8f')\n"
            "    axes[1].plot(eval_x, [item.get('vs_weak_heuristic_win_rate', 0.0) for item in evaluation], label='vs weak', color='#e9c46a')\n"
            "    axes[1].plot(eval_x, [item.get('vs_minimax_1_win_rate', 0.0) for item in evaluation], label='vs minimax_1', color='#457b9d')\n"
            "    axes[1].plot(eval_x, [item.get('vs_minimax_2_win_rate', 0.0) for item in evaluation], label='vs minimax_2', color='#1d3557')\n"
            "    axes[1].plot(eval_x, [item.get('vs_strong_heuristic_win_rate', 0.0) for item in evaluation], label='vs strong', color='#bc4749')\n"
            "    axes[1].legend()\n"
            "axes[1].set_title('Win rate por avaliacao')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        ),
        md_cell("## Passo 6: Avaliacao final mais estavel"),
        code_cell(
            "showcase_eval = {\n"
            "    'vs_random': evaluate_match_summary(agent, lambda game_idx: RandomAgent(seed=10_000 + game_idx), games=120),\n"
            "    'vs_weak': evaluate_match_summary(agent, lambda game_idx: WeakHeuristicAgent(seed=20_000 + game_idx), games=120),\n"
            "    'vs_minimax_1': evaluate_match_summary(agent, lambda game_idx: MinimaxAgent(depth=1, seed=25_000 + game_idx), games=120),\n"
            "    'vs_minimax_2': evaluate_match_summary(agent, lambda game_idx: MinimaxAgent(depth=2, seed=27_000 + game_idx), games=120),\n"
            "    'vs_strong': evaluate_match_summary(agent, lambda game_idx: StrongHeuristicAgent(seed=30_000 + game_idx), games=120),\n"
            "}\n"
            "showcase_eval\n"
        ),
    ]
    return notebook(cells)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks" / "ppo"
    notebook_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = notebook_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / ".gitkeep").write_text("", encoding="utf-8")

    (notebook_dir / "04_ppo_self_play.ipynb").write_text(
        json.dumps(build_training_notebook(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (notebook_dir / "05_ppo_best_model_showcase.ipynb").write_text(
        json.dumps(build_showcase_notebook(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    for stale_name in ["07_ppo_curriculum_experiments.ipynb", "08_ppo_curriculum_focus.ipynb"]:
        stale_path = notebook_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()


if __name__ == "__main__":
    main()
