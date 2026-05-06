# Aprendizagem por Reforco

Repositorio do projeto **Aprendizagem Competitiva no Jogo 4 em Linha: Implementacao de Self-Play com Aprendizagem por Reforco**.

## Estado atual

O repositorio ja inclui:

- ambiente base de `Connect Four`;
- agentes baseline `random` e `heuristic`;
- implementacao inicial de `MCTS`;
- pipeline inicial de `DQN` em self-play;
- pipeline inicial de `PPO` em self-play;
- implementacao inicial de `AlphaZero` simplificado;
- utilitarios de avaliacao e torneio;
- testes minimos do ambiente e dos agentes.

## Estrutura

- `connect4_rl/`
  - codigo principal do projeto.
- `connect4_rl/envs/`
  - regras, estado e interface do ambiente.
- `connect4_rl/agents/baselines/`
  - agentes simples para comparacao.
- `connect4_rl/agents/planning/`
  - algoritmos de planeamento, comecando por `MCTS`.
- `connect4_rl/experiments/`
  - partidas e torneios entre agentes.
- `scripts/`
  - pontos de entrada para correr experiencias.
- `tests/`
  - testes automaticos base.
- `docs/`
  - notas de arquitetura e roadmap tecnico.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .
./.venv/bin/pip install ipykernel
```

Opcional para integrar o ambiente com Jupyter:

```bash
./.venv/bin/python -m ipykernel install --user --name connect4-rl
```

## Como correr

### Torneio baseline

```bash
./.venv/bin/python scripts/run_baseline_tournament.py --games-per-pair 12 --mcts-simulations 150
```

### Treino DQN base

```bash
./.venv/bin/python scripts/run_dqn_self_play.py --episodes 300 --eval-interval 50 --eval-games 24
./.venv/bin/python scripts/run_dqn_self_play.py --episodes 300 --eval-interval 50 --eval-games 24 --lessons-dir curriculums/connect_four_dqn
```

### Ablacao DQN

```bash
./.venv/bin/python scripts/run_dqn_ablation.py
```

### Treino PPO em self-play

```bash
./.venv/bin/python scripts/run_ppo_self_play.py --episodes 300 --eval-interval 50 --eval-games 24
```

### Ablacao PPO

```bash
./.venv/bin/python scripts/run_ppo_ablation.py
```

### Curricula PPO

```bash
./.venv/bin/python scripts/run_ppo_curriculum.py --agenda curriculum_basic --episodes 180 --eval-interval 30 --eval-games 12
./.venv/bin/python scripts/run_ppo_curriculum.py --agenda curriculum_mid_self --episodes 180 --eval-interval 30 --eval-games 12
./.venv/bin/python scripts/run_ppo_curriculum.py --agenda curriculum_late_heuristic --episodes 180 --eval-interval 30 --eval-games 12
./.venv/bin/python scripts/run_ppo_curriculum.py --agenda co_training_dual --episodes 180 --eval-interval 30 --eval-games 12
```

### Testes

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

### Treino AlphaZero simplificado

```bash
./.venv/bin/python scripts/run_alphazero_self_play.py --episodes 200 --eval-interval 25 --eval-games 24 --mcts-simulations 80 --eval-mcts-simulations 120
```

### Curricula DQN

```bash
./.venv/bin/python scripts/run_dqn_curriculum.py --agenda curriculum_classic --episodes 180 --eval-interval 30 --eval-games 12
./.venv/bin/python scripts/run_dqn_curriculum.py --agenda curriculum_self_bridge --episodes 180 --eval-interval 30 --eval-games 12
./.venv/bin/python scripts/run_dqn_curriculum.py --agenda curriculum_probabilistic_mix --episodes 180 --eval-interval 30 --eval-games 12
```

### Curricula AlphaZero

```bash
./.venv/bin/python scripts/run_alphazero_curriculum.py --agenda curriculum_basic --episodes 180 --eval-interval 30 --eval-games 12 --mcts-simulations 80 --eval-mcts-simulations 120
./.venv/bin/python scripts/run_alphazero_curriculum.py --agenda curriculum_mid_self --episodes 180 --eval-interval 30 --eval-games 12 --mcts-simulations 80 --eval-mcts-simulations 120
./.venv/bin/python scripts/run_alphazero_curriculum.py --agenda curriculum_short_heuristic_late --episodes 180 --eval-interval 30 --eval-games 12 --mcts-simulations 80 --eval-mcts-simulations 120
./.venv/bin/python scripts/run_alphazero_curriculum.py --agenda curriculum_probabilistic_bridge --episodes 180 --eval-interval 30 --eval-games 12 --mcts-simulations 80 --eval-mcts-simulations 120
```

## Documentacao util

- organizacao do repositorio: [docs/REPO_STRUCTURE.md](/home/vasco44/AR/docs/REPO_STRUCTURE.md)
- roadmap tecnico: [docs/NEXT_STEPS.md](/home/vasco44/AR/docs/NEXT_STEPS.md)
- planeamento do trabalho: [Planeamento.typ](/home/vasco44/AR/Planeamento.typ)

## Notebooks

A ideia do repositorio e usar os notebooks como camada simples de execucao e analise, em cima do codigo que vive em `connect4_rl/`.

- `notebooks/baselines/01_baselines.ipynb`
- `notebooks/planning/02_mcts.ipynb`
- `notebooks/dqn/03_dqn_self_play.ipynb`
- `notebooks/dqn/11_dqn_curriculum_experiments.ipynb`
- `notebooks/dqn/12_dqn_curriculum_focus.ipynb`
- `notebooks/ppo/04_ppo_self_play.ipynb`
- `notebooks/alphazero/05_alphazero_simplified.ipynb`
- `notebooks/06_model_comparison.ipynb`
- `notebooks/ppo/07_ppo_curriculum_experiments.ipynb`
- `notebooks/ppo/08_ppo_curriculum_focus.ipynb`
- `notebooks/alphazero/09_alphazero_curriculum_experiments.ipynb`
- `notebooks/alphazero/10_alphazero_curriculum_focus.ipynb`

## Grupo

- Afonso Sousa
- Luis Cunha
- Paulo Cabrita
- Vasco Macedo
