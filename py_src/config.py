from dataclasses import dataclass


@dataclass
class AlphaClaudeConfig:
    # Neural network
    num_res_blocks: int = 20
    num_filters: int = 256
    input_planes: int = 119
    policy_size: int = 4672

    # MCTS
    num_simulations: int = 800
    c_base: float = 19652.0
    c_init: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    mcts_batch_size: int = 64

    # Temperature
    temp_threshold: int = 30  # moves before switching to tau->0
    temperature: float = 1.0
    temperature_final: float = 0.01  # near-deterministic

    # Self-play
    num_self_play_games: int = 100
    num_parallel_games: int = 256
    max_game_length: int = 512

    # Training
    batch_size: int = 256
    learning_rate: float = 0.02
    lr_milestones: tuple = (100, 200, 300)
    lr_gamma: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9
    num_epochs: int = 1
    replay_buffer_size: int = 500_000
    min_replay_size: int = 2048  # min samples before training
    training_steps_per_iteration: int = 1000

    # Evaluation (arena)
    arena_games: int = 40
    arena_threshold: float = 0.55  # win rate to accept new model

    # System
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    num_iterations: int = 100
