import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class GlobalConfig:
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    verbose: bool = True

    def validate(self):
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"device must be 'auto', 'cpu', or 'cuda', got {self.device}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


@dataclass
class EnvironmentConfig:
    board_height: int = 6
    board_width: int = 7
    win_length: int = 4

    def validate(self):
        if self.board_height < 3 or self.board_width < 3:
            raise ValueError("board must be at least 3x3")
        if self.win_length < 3 or self.win_length > min(self.board_height, self.board_width):
            raise ValueError(f"win_length must be 3-{min(self.board_height, self.board_width)}")


@dataclass
class DQNConfig:
    # Architecture
    hidden_dim: int = 128
    channel_sizes: List[int] = field(default_factory=lambda: [128])
    kernel_sizes: List[int] = field(default_factory=lambda: [4])
    stride_sizes: List[int] = field(default_factory=lambda: [1])
    head_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    use_dueling_head: bool = False

    # Learning
    learning_rate: float = 1e-4
    batch_size: int = 256
    replay_buffer_size: int = 100000
    min_replay_size: int = 256
    gamma: float = 0.99

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_rate: float = 0.9998

    # Target network
    target_update_freq: int = 1
    tau: float = 0.01

    # Self-play
    opponent_pool_size: int = 6
    opponent_refresh_interval: int = 30
    warmup_episodes: int = 40
    random_opponent_fraction: float = 0.05
    heuristic_opponent_fraction: float = 0.45
    opponent_epsilon: float = 0.05
    self_play_min_episodes_before_early_stop: int = 100
    self_play_early_stop_patience_evals: int = 2

    # Tutorial-style population training
    population_size: int = 1
    episodes_per_epoch: int = 10
    evo_epochs: int = 1
    evo_loop: int = 50
    max_steps_per_episode: int = 500
    tournament_size: int = 1

    # Tutorial-style mutation settings
    no_mutation_prob: float = 1.0
    mutation_lr_prob: float = 0.0
    mutation_batch_prob: float = 0.0
    mutation_learn_step_prob: float = 0.0
    mutation_grow_factor: float = 1.5
    mutation_shrink_factor: float = 0.75
    mutation_min_lr: float = 1e-4
    mutation_max_lr: float = 1e-2
    mutation_min_batch_size: int = 8
    mutation_max_batch_size: int = 64
    mutation_min_learn_step: int = 1
    mutation_max_learn_step: int = 120
    
    # Evaluation
    eval_interval: int = 50
    eval_games: int = 24
    checkpoint_score_heuristic_weight: float = 6.0

    # Optimization
    learn_step: int = 1
    gradient_updates_per_step: int = 1
    use_horizontal_symmetry_augmentation: bool = True

    # Training limits
    episodes: int = 500

    def validate(self):
        if not 0 <= self.learning_rate <= 0.1:
            raise ValueError(f"learning_rate must be in [0, 0.1], got {self.learning_rate}")
        if not 1 <= self.batch_size <= 1024:
            raise ValueError(f"batch_size must be in [1, 1024], got {self.batch_size}")
        if not 0 < self.gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not (0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError(f"must have 0 <= epsilon_end <= epsilon_start <= 1, got end={self.epsilon_end}, start={self.epsilon_start}")
        if not 0 < self.epsilon_decay_rate <= 1:
            raise ValueError(f"epsilon_decay_rate must be in (0, 1], got {self.epsilon_decay_rate}")
        if self.target_update_freq < 1:
            raise ValueError(f"target_update_freq must be >= 1, got {self.target_update_freq}")
        if self.learn_step < 1:
            raise ValueError(f"learn_step must be >= 1, got {self.learn_step}")
        if self.eval_interval < 1:
            raise ValueError(f"eval_interval must be >= 1, got {self.eval_interval}")
        if self.population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {self.population_size}")
        if self.episodes_per_epoch < 1:
            raise ValueError(f"episodes_per_epoch must be >= 1, got {self.episodes_per_epoch}")
        if self.evo_epochs < 1:
            raise ValueError(f"evo_epochs must be >= 1, got {self.evo_epochs}")
        if self.evo_loop < 1:
            raise ValueError(f"evo_loop must be >= 1, got {self.evo_loop}")
        if self.max_steps_per_episode < 1:
            raise ValueError(f"max_steps_per_episode must be >= 1, got {self.max_steps_per_episode}")
        if self.tournament_size < 1:
            raise ValueError(f"tournament_size must be >= 1, got {self.tournament_size}")
        if self.random_opponent_fraction < 0 or self.heuristic_opponent_fraction < 0:
            raise ValueError("opponent mix fractions must be >= 0")
        if self.random_opponent_fraction + self.heuristic_opponent_fraction > 1:
            raise ValueError(
                "random_opponent_fraction + heuristic_opponent_fraction must be <= 1, "
                f"got {self.random_opponent_fraction + self.heuristic_opponent_fraction}"
            )
        if self.self_play_min_episodes_before_early_stop < 0:
            raise ValueError(
                "self_play_min_episodes_before_early_stop must be >= 0, "
                f"got {self.self_play_min_episodes_before_early_stop}"
            )
        if self.self_play_early_stop_patience_evals < 1:
            raise ValueError(
                "self_play_early_stop_patience_evals must be >= 1, "
                f"got {self.self_play_early_stop_patience_evals}"
            )
        if any(
            value < 0
            for value in [
                self.no_mutation_prob,
                self.mutation_lr_prob,
                self.mutation_batch_prob,
                self.mutation_learn_step_prob,
            ]
        ):
            raise ValueError("mutation probabilities must be >= 0")
        mutation_prob_sum = (
            self.no_mutation_prob
            + self.mutation_lr_prob
            + self.mutation_batch_prob
            + self.mutation_learn_step_prob
        )
        if mutation_prob_sum <= 0:
            raise ValueError("sum of mutation probabilities must be > 0")
        if self.mutation_grow_factor <= 1.0:
            raise ValueError(f"mutation_grow_factor must be > 1, got {self.mutation_grow_factor}")
        if not 0 < self.mutation_shrink_factor < 1:
            raise ValueError(
                f"mutation_shrink_factor must be in (0, 1), got {self.mutation_shrink_factor}"
            )
        if not (0 < self.mutation_min_lr <= self.mutation_max_lr):
            raise ValueError(
                f"mutation lr bounds invalid: min={self.mutation_min_lr}, max={self.mutation_max_lr}"
            )
        if not (1 <= self.mutation_min_batch_size <= self.mutation_max_batch_size):
            raise ValueError(
                "mutation batch size bounds invalid: "
                f"min={self.mutation_min_batch_size}, max={self.mutation_max_batch_size}"
            )
        if not (1 <= self.mutation_min_learn_step <= self.mutation_max_learn_step):
            raise ValueError(
                "mutation learn_step bounds invalid: "
                f"min={self.mutation_min_learn_step}, max={self.mutation_max_learn_step}"
            )
        if not self.channel_sizes:
            raise ValueError("channel_sizes must not be empty")
        if not (len(self.channel_sizes) == len(self.kernel_sizes) == len(self.stride_sizes)):
            raise ValueError("channel_sizes, kernel_sizes and stride_sizes must have the same length")
        if any(value < 1 for value in self.channel_sizes + self.kernel_sizes + self.stride_sizes):
            raise ValueError("channel_sizes, kernel_sizes and stride_sizes must all be >= 1")
        if not self.head_hidden_sizes:
            raise ValueError("head_hidden_sizes must not be empty")
        if any(value < 1 for value in self.head_hidden_sizes):
            raise ValueError("head_hidden_sizes must all be >= 1")


@dataclass
class PPOConfig:
    # Architecture
    hidden_dim: int = 256
    channel_sizes: List[int] = field(default_factory=lambda: [64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [4, 3])
    stride_sizes: List[int] = field(default_factory=lambda: [1, 1])
    head_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])

    # Learning
    learning_rate: float = 2.5e-4
    batch_size: int = 64  # Total batch size (rollout_episodes * steps)
    minibatch_size: int = 32
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    # Rollout collection
    rollout_length: int = 128
    rollout_episodes_per_update: int = 8
    use_horizontal_symmetry_augmentation: bool = True
    anneal_learning_rate: bool = True

    # Self-play
    opponent_pool_size: int = 5
    warmup_episodes: int = 40
    random_opponent_fraction: float = 0.20
    heuristic_opponent_fraction: float = 0.20
    reward_shaping: bool = True
    reward_shaping_scale: float = 10.0
    threat_bonus_scale: float = 0.35
    opponent_threat_penalty_scale: float = 0.60
    blocked_threat_bonus_scale: float = 0.75
    allowed_threat_penalty_scale: float = 1.10
    center_control_scale: float = 0.10
    imitation_loss_coeff: float = 0.40
    enable_policy_bootstrap: bool = True
    # Bootstrapping / curriculum options (may be absent in older configs)
    bootstrap_samples: int = 0
    bootstrap_batch_size: int = 64
    bootstrap_epochs: int = 1
    bootstrap_learning_rate: float = 1e-4
    bootstrap_teacher_kind: str = "strong"
    curriculum_profile: str = "tutorial"
    freeze_feature_extractor_lessons: int = 0
    # Early-stopping parameters for self-play phase
    self_play_min_episodes_before_early_stop: int = 0
    self_play_early_stop_patience_evals: int = 2
    
    # Evaluation
    eval_interval: int = 50
    eval_games: int = 24
    checkpoint_score_heuristic_weight: float = 2.0

    # Training limits
    episodes: int = 500

    def validate(self):
        if self.learning_rate < 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")
        if not 1 <= self.minibatch_size <= 1024:
            raise ValueError(f"minibatch_size must be in [1, 1024], got {self.minibatch_size}")
        if not 0 < self.gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0 < self.gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be in (0, 1], got {self.gae_lambda}")
        if not 0 < self.clip_ratio <= 1:
            raise ValueError(f"clip_ratio must be in (0, 1], got {self.clip_ratio}")
        if not 0 <= self.entropy_coeff <= 1:
            raise ValueError(f"entropy_coeff must be in [0, 1], got {self.entropy_coeff}")
        if not self.channel_sizes:
            raise ValueError("channel_sizes must not be empty")
        if not (len(self.channel_sizes) == len(self.kernel_sizes) == len(self.stride_sizes)):
            raise ValueError("channel_sizes, kernel_sizes and stride_sizes must have the same length")
        if any(value < 1 for value in self.channel_sizes + self.kernel_sizes + self.stride_sizes):
            raise ValueError("channel_sizes, kernel_sizes and stride_sizes must all be >= 1")
        if not self.head_hidden_sizes:
            raise ValueError("head_hidden_sizes must not be empty")
        if any(value < 1 for value in self.head_hidden_sizes):
            raise ValueError("head_hidden_sizes must all be >= 1")
        if self.bootstrap_samples < 0:
            raise ValueError(f"bootstrap_samples must be >= 0, got {self.bootstrap_samples}")
        if self.bootstrap_batch_size < 1:
            raise ValueError(f"bootstrap_batch_size must be >= 1, got {self.bootstrap_batch_size}")
        if self.bootstrap_epochs < 0:
            raise ValueError(f"bootstrap_epochs must be >= 0, got {self.bootstrap_epochs}")
        if self.freeze_feature_extractor_lessons < 0:
            raise ValueError(f"freeze_feature_extractor_lessons must be >= 0, got {self.freeze_feature_extractor_lessons}")
        if self.self_play_min_episodes_before_early_stop < 0:
            raise ValueError(
                f"self_play_min_episodes_before_early_stop must be >= 0, got {self.self_play_min_episodes_before_early_stop}"
            )
        if self.self_play_early_stop_patience_evals < 1:
            raise ValueError(
                f"self_play_early_stop_patience_evals must be >= 1, got {self.self_play_early_stop_patience_evals}"
            )


@dataclass
class AlphaZeroConfig:
    episodes: int = 800
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    # `num_workers` deprecated: use `episodes_per_batch` to control how many
    # self-play episodes are grouped into a single training batch. Kept for
    # backward compatibility with older config files but ignored by training
    # logic in favour of `episodes_per_batch`.
    num_workers: int = 1
    episodes_per_batch: int = 1
    replay_capacity: int = 20_000
    replay_warmup_games: int = 12
    update_epochs: int = 1
    updates_per_episode: int = 2
    n_filters: int = 64
    n_res_blocks: int = 4
    mcts_simulations: int = 48
    mcts_start_search_iter: int | None = 12
    mcts_max_search_iter: int | None = 48
    mcts_search_increment: int = 1
    eval_mcts_simulations: int | None = 80
    c_puct: float = 2.0
    dirichlet_alpha: float = 1.0
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.25
    temperature_drop_move: int = 8
    eval_interval: int = 40
    eval_games: int = 16
    seed: int = 0
    device: str = "cpu"
    checkpoint_score_heuristic_weight: float = 2.0
    use_horizontal_symmetry_augmentation: bool = True
    value_loss_coef: float = 1.0
    max_grad_norm: float = 5.0
    anneal_learning_rate: bool = True
    root_noise_each_move: bool = True
    tactical_eval_examples: int = 96

    def validate(self):
        if self.learning_rate < 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        # Prefer explicit episodes_per_batch; fall back to num_workers if not set.
        if getattr(self, "episodes_per_batch", 1) < 1:
            raise ValueError(
                f"episodes_per_batch must be >= 1, got {getattr(self, 'episodes_per_batch', None)}"
            )
        if self.n_filters < 32:
            raise ValueError(f"n_filters must be >= 32, got {self.n_filters}")
        if self.n_res_blocks < 1:
            raise ValueError(f"n_res_blocks must be >= 1, got {self.n_res_blocks}")
        if self.mcts_simulations < 1:
            raise ValueError(f"mcts_simulations must be >= 1, got {self.mcts_simulations}")
        if self.mcts_start_search_iter is not None and self.mcts_start_search_iter < 1:
            raise ValueError(f"mcts_start_search_iter must be >= 1, got {self.mcts_start_search_iter}")
        if self.mcts_max_search_iter is not None and self.mcts_max_search_iter < 1:
            raise ValueError(f"mcts_max_search_iter must be >= 1, got {self.mcts_max_search_iter}")
        if self.mcts_search_increment < 0:
            raise ValueError(f"mcts_search_increment must be >= 0, got {self.mcts_search_increment}")
        if (
            self.mcts_start_search_iter is not None
            and self.mcts_max_search_iter is not None
            and self.mcts_start_search_iter > self.mcts_max_search_iter
        ):
            raise ValueError(
                "mcts_start_search_iter must be <= mcts_max_search_iter, "
                f"got start={self.mcts_start_search_iter}, max={self.mcts_max_search_iter}"
            )
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be > 0, got {self.c_puct}")
        if not 0 < self.dirichlet_alpha <= 1:
            raise ValueError(f"dirichlet_alpha must be in (0, 1], got {self.dirichlet_alpha}")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError(f"dirichlet_epsilon must be in [0, 1], got {self.dirichlet_epsilon}")


@dataclass
class MCTSConfig:
    simulations: int = 100
    exploration_weight: float = 1.0
    use_heuristic_rollout: bool = True

    def validate(self):
        if self.simulations < 1:
            raise ValueError(f"simulations must be >= 1, got {self.simulations}")
        if self.exploration_weight <= 0:
            raise ValueError(f"exploration_weight must be > 0, got {self.exploration_weight}")


@dataclass
class EvaluationConfig:
    # Tournament
    games_per_matchup: int = 20
    calculate_elo: bool = True

    # Logging
    log_dir: str = "./runs"
    save_checkpoints: bool = True
    checkpoint_freq: int = 50
    save_best_model: bool = True

    # Game recording
    save_game_traces: bool = False
    record_game_videos: bool = False

    def validate(self):
        if self.games_per_matchup < 1:
            raise ValueError(f"games_per_matchup must be >= 1, got {self.games_per_matchup}")
        if self.checkpoint_freq < 1:
            raise ValueError(f"checkpoint_freq must be >= 1, got {self.checkpoint_freq}")


@dataclass
class BaselineConfig:
    """Configuration for a single baseline agent"""
    type: str
    simulations: Optional[int] = None

    def validate(self):
        if self.type not in ["random", "heuristic", "mcts"]:
            raise ValueError(f"type must be 'random', 'heuristic', or 'mcts', got {self.type}")
        if self.type == "mcts" and (self.simulations is None or self.simulations < 1):
            raise ValueError(f"MCTS baseline requires simulations >= 1, got {self.simulations}")


@dataclass
class NotebookSettings:
    """Settings specific to notebook execution"""
    seed: int = 42  # Defaults to global seed
    quick_test_episodes: int = 180
    quick_test_eval_interval: int = 30
    quick_test_eval_games: int = 12
    quick_test_mcts_simulations: int = 30
    quick_test_eval_mcts_simulations: int = 50
    
    baseline_games_per_pair: int = 20
    
    mcts_games_per_pair: int = 4
    mcts_default_simulations: int = 80
    mcts_simulation_sweep: List[int] = field(default_factory=lambda: [20, 40, 80, 120])
    mcts_sweep_games_per_pair: int = 20
    
    model_comparison_games_per_pair: int = 20
    model_comparison_mcts_simulations: int = 50
    model_comparison_validation_games: int = 100
    model_comparison_validation_threshold: float = 0.95
    
    ablation_seeds: List[int] = field(default_factory=lambda: [7, 17, 27])


@dataclass
class Config:
    """Master configuration class"""
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    alphazero: AlphaZeroConfig = field(default_factory=AlphaZeroConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    baselines: Dict[str, BaselineConfig] = field(default_factory=dict)
    notebook_settings: NotebookSettings = field(default_factory=NotebookSettings)

    def validate(self):
        """Validate all sub-configurations"""
        self.global_.validate()
        self.environment.validate()
        self.dqn.validate()
        self.ppo.validate()
        self.alphazero.validate()
        self.mcts.validate()
        self.evaluation.validate()
        for name, baseline in self.baselines.items():
            baseline.validate()
        print("✓ Configuration validated successfully")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def resolve_device(self) -> str:
        """Resolve device: 'auto' -> 'cuda' or 'cpu'"""
        if self.global_.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.global_.device


def load_config(config_path: str) -> Config:
    config_path = Path(config_path)

    candidate_paths = [config_path]
    if not config_path.is_absolute():
        candidate_paths.append(Path(__file__).resolve().parents[1] / config_path)

    resolved_path = next((path for path in candidate_paths if path.exists()), None)
    if resolved_path is None:
        tried_paths = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"Configuration file not found. Tried: {tried_paths}")

    with open(resolved_path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Extract top-level sections
    global_dict = data.get("global", {})
    env_dict = data.get("environment", {})
    dqn_dict = data.get("dqn", {})
    ppo_dict = data.get("ppo", {})
    alphazero_dict = dict(data.get("alphazero", {}))
    mcts_dict = data.get("mcts", {})
    eval_dict = data.get("evaluation", {})
    baselines_dict = data.get("baselines", {})
    
    # Create global config first to get seed for notebook settings
    global_config = GlobalConfig(**global_dict)
    
    # Extract and parse notebook settings (pass global_seed as default)
    notebook_dict = data.get("notebook_settings", {})
    notebook_settings_obj = _parse_notebook_settings(notebook_dict, global_config.seed)

    # Parse baselines
    parsed_baselines = {}
    for name, baseline_config in baselines_dict.items():
        parsed_baselines[name] = BaselineConfig(**baseline_config)

    # Backward compatibility with the old simplified AlphaZero config.
    alphazero_dict.pop("hidden_dim", None)

    # Create config object
    config = Config(
        global_=global_config,
        environment=EnvironmentConfig(**env_dict),
        dqn=DQNConfig(**dqn_dict),
        ppo=PPOConfig(**ppo_dict),
        alphazero=AlphaZeroConfig(**alphazero_dict),
        mcts=MCTSConfig(**mcts_dict),
        evaluation=EvaluationConfig(**eval_dict),
        baselines=parsed_baselines,
        notebook_settings=notebook_settings_obj,
    )

    config.validate()
    return config


def _parse_notebook_settings(notebook_dict: Dict[str, Any], global_seed: int) -> NotebookSettings:
    """Parse notebook settings from YAML configuration"""
    # Seed defaults to global seed if not specified in notebook_settings
    seed = notebook_dict.get("seed", global_seed)
    
    # Extract quick_test settings
    quick_test = notebook_dict.get("quick_test", {})
    quick_test_episodes = quick_test.get("episodes", 180)
    quick_test_eval_interval = quick_test.get("eval_interval", 30)
    quick_test_eval_games = quick_test.get("eval_games", 12)
    quick_test_mcts_simulations = quick_test.get("mcts_simulations", 30)
    quick_test_eval_mcts_simulations = quick_test.get("eval_mcts_simulations", 50)
    
    # Extract testing settings
    testing = notebook_dict.get("testing", {})
    
    baseline = testing.get("baseline", {})
    baseline_games_per_pair = baseline.get("games_per_pair", 20)
    
    mcts = testing.get("mcts", {})
    mcts_games_per_pair = mcts.get("games_per_pair", 4)
    mcts_default_simulations = mcts.get("default_simulations", 80)
    mcts_simulation_sweep = mcts.get("simulation_sweep", [20, 40, 80, 120])
    mcts_sweep_games_per_pair = mcts.get("sweep_games_per_pair", 20)
    
    model_comparison = testing.get("model_comparison", {})
    model_comparison_games_per_pair = model_comparison.get("games_per_pair", 20)
    model_comparison_mcts_simulations = model_comparison.get("mcts_simulations", 50)
    model_comparison_validation_games = model_comparison.get("validation_games_vs_random", 100)
    model_comparison_validation_threshold = model_comparison.get("validation_threshold", 0.95)
    
    # Extract ablation seeds
    ablation_seeds = notebook_dict.get("ablation_seeds", [7, 17, 27])
    
    return NotebookSettings(
        seed=seed,
        quick_test_episodes=quick_test_episodes,
        quick_test_eval_interval=quick_test_eval_interval,
        quick_test_eval_games=quick_test_eval_games,
        quick_test_mcts_simulations=quick_test_mcts_simulations,
        quick_test_eval_mcts_simulations=quick_test_eval_mcts_simulations,
        baseline_games_per_pair=baseline_games_per_pair,
        mcts_games_per_pair=mcts_games_per_pair,
        mcts_default_simulations=mcts_default_simulations,
        mcts_simulation_sweep=mcts_simulation_sweep,
        mcts_sweep_games_per_pair=mcts_sweep_games_per_pair,
        model_comparison_games_per_pair=model_comparison_games_per_pair,
        model_comparison_mcts_simulations=model_comparison_mcts_simulations,
        model_comparison_validation_games=model_comparison_validation_games,
        model_comparison_validation_threshold=model_comparison_validation_threshold,
        ablation_seeds=ablation_seeds,
    )


def get_default_config() -> Config:
    """Get a default configuration"""
    return Config()
