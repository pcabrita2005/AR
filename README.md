# Aprendizagem por Reforço

Repositório do projeto **Aprendizagem Competitiva no Jogo 4 em Linha: Implementação de Self-Play com Aprendizagem por Reforço**.

## Estado atual

O repositório já inclui:

- ambiente base de `Connect Four`;
- agentes baseline `random`, `heuristic` e `minimax`;
- implementação de `MCTS`;
- pipeline final de `DQN` em self-play;
- pipeline final de `PPO` em self-play;
- implementação inicial de `AlphaZero` simplificado;
- utilitários de avaliação e torneio;
- testes automáticos do ambiente e dos agentes.

## Estrutura

- `connect4_rl/`
  - código principal do projeto.
- `connect4_rl/envs/`
  - regras, estado e interface do ambiente.
- `connect4_rl/agents/baselines/`
  - agentes simples para comparação.
- `connect4_rl/agents/planning/`
  - algoritmos de planeamento, começando por `MCTS`.
- `connect4_rl/experiments/`
  - treino, avaliação e torneios entre agentes.
- `scripts/`
  - pontos de entrada para correr experiências.
- `tests/`
  - testes automáticos.
- `docs/`
  - notas de arquitetura e roadmap técnico.

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

### Treino DQN

```bash
./.venv/bin/python scripts/run_dqn_self_play.py --episodes 300 --eval-interval 50 --eval-games 24
```

O treino `DQN` segue um pipeline de _curriculum learning_ com quatro lições:

- `lesson1_random`
- `lesson2_weak`
- `lesson3_strong`
- `lesson4_self_play`

### Treino PPO

```bash
./.venv/bin/python scripts/run_ppo_self_play.py --variant baseline --episodes 300 --eval-interval 30 --eval-games 24
```

O treino `PPO` foi reorganizado para seguir a mesma lógica de progressão do `DQN`, com bootstrap supervisionado, currículo por lições e fecho conservador em self-play.

### Treino AlphaZero simplificado

```bash
./.venv/bin/python scripts/run_alphazero_self_play.py --episodes 200 --eval-interval 25 --eval-games 24 --mcts-simulations 80 --eval-mcts-simulations 120
```

### Testes

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

## Documentação útil

- organização do repositório: [docs/REPO_STRUCTURE.md](/home/vasco44/AR/docs/REPO_STRUCTURE.md)
- roadmap técnico: [docs/NEXT_STEPS.md](/home/vasco44/AR/docs/NEXT_STEPS.md)
- planeamento do trabalho: [Planeamento.typ](/home/vasco44/AR/Planeamento.typ)

## Notebooks

Os notebooks funcionam como camada simples de execução e análise por cima do código de `connect4_rl/`.

- `notebooks/baselines/01_baselines.ipynb`
- `notebooks/planning/02_mcts.ipynb`
- `notebooks/dqn/03_dqn_self_play.ipynb`
- `notebooks/dqn/04_dqn_best_model_showcase.ipynb`
- `notebooks/ppo/04_ppo_self_play.ipynb`
- `notebooks/ppo/05_ppo_best_model_showcase.ipynb`
- `notebooks/alphazero/05_alphazero_simplified.ipynb`
- `notebooks/alphazero/06_alphazero_best_model_showcase.ipynb`
- `notebooks/06_model_comparison.ipynb`

### DQN

O repositório inclui checkpoints compactos do `DQN` em:

- `notebooks/dqn/outputs/models/dqn/lesson1_trained_agent.pt`
- `notebooks/dqn/outputs/models/dqn/lesson2_trained_agent.pt`
- `notebooks/dqn/outputs/models/dqn/lesson3_trained_agent.pt`
- `notebooks/dqn/outputs/models/dqn/lesson4_trained_agent.pt`

Isto permite abrir diretamente o notebook:

- `notebooks/dqn/04_dqn_best_model_showcase.ipynb`

mesmo que a pasta `outputs/` com as runs completas não exista localmente. O notebook tenta primeiro usar uma run completa em `outputs/`; se não encontrar, faz fallback para o checkpoint compacto versionado.

### PPO

O `PPO` foi limpo para ficar alinhado com o `DQN`:

- `notebooks/ppo/04_ppo_self_play.ipynb`
  - treino e iteração por variantes.
- `notebooks/ppo/05_ppo_best_model_showcase.ipynb`
  - leitura da melhor run e avaliação final mais estável.

## Grupo

- Afonso Sousa
- Luís Cunha
- Paulo Cabrita
- Vasco Macedo
