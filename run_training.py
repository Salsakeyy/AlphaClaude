#!/usr/bin/env python3
"""Entry point for RunPod training."""
import argparse
import torch
from py_src.config import AlphaClaudeConfig
from py_src.train import Trainer


def main():
    parser = argparse.ArgumentParser(description="AlphaClaude Training")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--self-play-games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--res-blocks", type=int, default=10)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--parallel-games", type=int, default=64)
    parser.add_argument("--mcts-batch-size", type=int, default=16)
    parser.add_argument("--arena-games", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--min-replay-size", type=int, default=2048)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--no-perf-log", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = AlphaClaudeConfig(
        num_iterations=args.iterations,
        num_self_play_games=args.self_play_games,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_res_blocks=args.res_blocks,
        num_filters=args.filters,
        num_parallel_games=args.parallel_games,
        mcts_batch_size=args.mcts_batch_size,
        arena_games=args.arena_games,
        num_epochs=args.epochs,
        training_steps_per_iteration=args.train_steps,
        min_replay_size=args.min_replay_size,
        use_amp=not args.no_amp,
        use_tf32=not args.no_tf32,
        log_perf=not args.no_perf_log,
        checkpoint_dir=args.checkpoint_dir,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        if config.num_parallel_games < 128 or config.mcts_batch_size < 32:
            print("Warning: self-play batching is conservative for a 4090 "
                  f"(parallel_games={config.num_parallel_games}, mcts_batch_size={config.mcts_batch_size}). "
                  "Expect low average GPU utilization.")
        if config.num_res_blocks <= 10 and config.num_filters <= 128:
            print("Warning: model size is small (10x128 or smaller). "
                  "A 4090 can finish inference/training batches very quickly and appear underutilized.")

    try:
        import alphaclaude_cpp as ac
        has_omp = getattr(ac, "HAS_OPENMP", None)
        if has_omp is not None:
            print(f"OpenMP-enabled MCTS: {bool(has_omp)}")
            if not has_omp:
                print("Warning: C++ MCTS was built without OpenMP, so self-play search is CPU single-threaded.")
    except Exception:
        pass

    trainer = Trainer(config)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.run()


if __name__ == "__main__":
    main()
