import os
import time
import torch

from .model import AlphaZeroNet
from .config import AlphaClaudeConfig


def save_model(model: AlphaZeroNet, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str, config: AlphaClaudeConfig = None, device: torch.device = None):
    if config is None:
        config = AlphaClaudeConfig()
    model = AlphaZeroNet(config)
    model.load_state_dict(torch.load(path, map_location=device or "cpu", weights_only=True))
    if device:
        model = model.to(device)
    return model


class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        if self.name:
            print(f"{self.name}: {elapsed:.2f}s")


class ThroughputTracker:
    def __init__(self):
        self.start_time = time.time()
        self.games = 0
        self.moves = 0
        self.training_steps = 0

    def add_games(self, n: int, moves: int):
        self.games += n
        self.moves += moves

    def add_training_steps(self, n: int):
        self.training_steps += n

    def report(self):
        elapsed = time.time() - self.start_time
        print(f"  Throughput: {self.games} games, {self.moves} moves, "
              f"{self.training_steps} train steps in {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Rates: {self.games/elapsed:.1f} games/s, "
                  f"{self.moves/elapsed:.0f} moves/s, "
                  f"{self.training_steps/elapsed:.0f} train_steps/s")
