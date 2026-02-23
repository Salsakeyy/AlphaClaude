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
    parser.add_argument("--arena-games", type=int, default=40)
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
        arena_games=args.arena_games,
        checkpoint_dir=args.checkpoint_dir,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    trainer = Trainer(config)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.run()


if __name__ == "__main__":
    main()
