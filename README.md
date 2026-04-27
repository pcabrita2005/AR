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

### Treino DQN em self-play

```bash
./.venv/bin/python scripts/run_dqn_self_play.py --episodes 300 --eval-interval 50 --eval-games 24
```

### Ablacao DQN

```bash
./.venv/bin/python scripts/run_dqn_ablation.py
```

### Testes

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

### Treino AlphaZero simplificado

```bash
./.venv/bin/python scripts/run_alphazero_self_play.py --episodes 200 --eval-interval 25 --eval-games 24 --mcts-simulations 80
```

## Documentacao util

- organizacao do repositorio: [docs/REPO_STRUCTURE.md](/home/vasco44/AR/docs/REPO_STRUCTURE.md)
- roadmap tecnico: [docs/NEXT_STEPS.md](/home/vasco44/AR/docs/NEXT_STEPS.md)
- planeamento do trabalho: [Planeamento.typ](/home/vasco44/AR/Planeamento.typ)

## Notebooks

A ideia do repositorio e usar os notebooks como camada simples de execucao e analise, em cima do codigo que vive em `connect4_rl/`.

- `notebooks/01_baselines.ipynb`
- `notebooks/02_mcts.ipynb`
- `notebooks/03_dqn_self_play.ipynb`
- `notebooks/04_ppo_self_play.ipynb`
- `notebooks/05_alphazero_simplified.ipynb`
- `notebooks/06_model_comparison.ipynb`

## Grupo

- Afonso Sousa
- Luis Cunha
- Paulo Cabrita
- Vasco Macedo
