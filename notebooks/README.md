# Notebooks

Os notebooks foram reorganizados por família de método:

- `notebooks/baselines/`
  - `01_baselines.ipynb`
- `notebooks/planning/`
  - `02_mcts.ipynb`
- `notebooks/dqn/`
  - `03_dqn_self_play.ipynb`
  - `04_dqn_best_model_showcase.ipynb`
- `notebooks/ppo/`
  - `04_ppo_self_play.ipynb`
  - `07_ppo_curriculum_experiments.ipynb`
  - `08_ppo_curriculum_focus.ipynb`
- `notebooks/alphazero/`
  - `05_alphazero_simplified.ipynb`
  - `06_alphazero_best_model_showcase.ipynb`

O notebook transversal de comparação final fica na raiz:

- `notebooks/06_model_comparison.ipynb`

Notas praticas:

- `notebooks/dqn/04_dqn_best_model_showcase.ipynb` consegue correr mesmo sem `outputs/` completos, porque faz fallback para os checkpoints compactos em `notebooks/dqn/outputs/models/dqn/`.
- `notebooks/alphazero/06_alphazero_best_model_showcase.ipynb` depende de runs presentes em `notebooks/alphazero/outputs/`.
