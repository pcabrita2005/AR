# Aprendizagem por Reforço

Repositório do projeto **Self-Play Connect Four: Implementação de Self-Play com Aprendizagem por Reforço**.

## Estado atual

O repositório já inclui:

- Ambiente base de `Connect Four`;
- Agentes baseline `random` e `heuristic`;
- Implementação inicial de `MCTS`;
- Pipeline inicial de `DQN` em self-play;
- Pipeline inicial de `PPO` em self-play;
- Implementação inicial de `AlphaZero` simplificado;
- Utilitários de avaliação e torneio;
- Testes mínimos do ambiente e dos agentes.

## Estrutura

- `connect4_rl/`
  - Código principal do projeto.
- `connect4_rl/envs/`
  - Regras, estado e interface do ambiente.
- `connect4_rl/agents/baselines/`
  - Agentes simples para comparação.
- `connect4_rl/agents/planning/`
  - Algoritmos de planeamento, começando por `MCTS`.
- `connect4_rl/experiments/`
  - Partidas e torneios entre agentes.
- `scripts/`
  - Pontos de entrada para correr experiências.
- `tests/`
  - Testes automáticos base.
- `docs/`
  - Notas de arquitetura e roadmap técnico.

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
```

O treino DQN segue um pipeline inspirado no tutorial de _curriculum learning_ com quatro lições sequenciais:

- `lesson1_random`
- `lesson2_weak`
- `lesson3_strong`
- `lesson4_self_play`

Cada lição reutiliza os melhores pesos da anterior, aplica _reward shaping_ e termina com self-play na última fase.

### Ablação DQN

```bash
./.venv/bin/python scripts/run_dqn_ablation.py
```

### Treino PPO em self-play

```bash
./.venv/bin/python scripts/run_ppo_self_play.py --episodes 300 --eval-interval 50 --eval-games 24
```

### Ablação PPO

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

## Documentação útil

- Organização do repositório: [docs/REPO_STRUCTURE.md](/home/vasco44/AR/docs/REPO_STRUCTURE.md)
- _Roadmap_ técnico: [docs/NEXT_STEPS.md](/home/vasco44/AR/docs/NEXT_STEPS.md)
- Planeamento do trabalho: [Planeamento.typ](/home/vasco44/AR/Planeamento.typ)

## Notebooks

A ideia do repositório é usar os notebooks como camada simples de execução e análise, em cima do código `connect4_rl/`.

- `notebooks/baselines/01_baselines.ipynb`
- `notebooks/planning/02_mcts.ipynb`
- `notebooks/dqn/03_dqn_self_play.ipynb`
- `notebooks/ppo/04_ppo_self_play.ipynb`
- `notebooks/alphazero/05_alphazero_simplified.ipynb`
- `notebooks/alphazero/06_alphazero_best_model_showcase.ipynb`
- `notebooks/06_model_comparison.ipynb`
- `notebooks/ppo/07_ppo_curriculum_experiments.ipynb`
- `notebooks/ppo/08_ppo_curriculum_focus.ipynb`

### Showcase DQN sem retraining

O repositório inclui checkpoints do DQN em:

- `notebooks/dqn/outputs/models/dqn/lesson1_trained_agent.pt`
- `notebooks/dqn/outputs/models/dqn/lesson2_trained_agent.pt`
- `notebooks/dqn/outputs/models/dqn/lesson3_trained_agent.pt`
- `notebooks/dqn/outputs/models/dqn/lesson4_trained_agent.pt`

Isto permite abrir diretamente o notebook:

- `notebooks/dqn/04_dqn_best_model_showcase.ipynb`

mesmo que a pasta `outputs/` com as runs completas não exista localmente. O notebook tenta primeiro usar um run completo em `outputs/`; se não encontrar, faz fallback para o checkpoint compacto versionado.

## Grupo

- Afonso Sousa
- Luís Cunha
- Paulo Cabrita
- Vasco Macedo
