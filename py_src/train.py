import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from .config import AlphaClaudeConfig
from .model import AlphaZeroNet, get_device
from .self_play import run_self_play
from .arena import pit


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, examples: list):
        self.buffer.extend(examples)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        inputs = np.stack([b[0] for b in batch])
        policies = np.stack([b[1] for b in batch])
        values = np.array([b[2] for b in batch], dtype=np.float32)

        return (
            torch.from_numpy(inputs),
            torch.from_numpy(policies),
            torch.from_numpy(values),
        )

    def __len__(self):
        return len(self.buffer)


class Trainer:
    def __init__(self, config: AlphaClaudeConfig = None):
        self.config = config or AlphaClaudeConfig()
        self.device = get_device()
        print(f"Using device: {self.device}")

        self.use_amp = bool(self.config.use_amp and self.device.type == "cuda")
        self.autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        if self.device.type == "cuda":
            if self.config.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            torch.backends.cudnn.benchmark = True
        print(f"AMP training: {self.use_amp}")
        if self.device.type == "cuda":
            print(f"TF32: {bool(self.config.use_tf32)}")

        self.model = AlphaZeroNet(self.config).to(self.device)
        self.scaler = None
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except AttributeError:
                self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=list(self.config.lr_milestones),
            gamma=self.config.lr_gamma,
        )
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        self.iteration = 0

    def train_step(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.config.batch_size

        if len(self.replay_buffer) < batch_size:
            return None

        self.model.train()
        inputs, target_policies, target_values = self.replay_buffer.sample(batch_size)
        inputs = inputs.to(self.device, non_blocking=True)
        target_policies = target_policies.to(self.device, non_blocking=True)
        target_values = target_values.to(self.device, non_blocking=True)

        with torch.amp.autocast(self.autocast_device, enabled=self.use_amp):
            # Forward
            policy_logits, pred_values = self.model(inputs)
            pred_values = pred_values.squeeze(-1)

            # Value loss: MSE
            value_loss = F.mse_loss(pred_values, target_values)

            # Policy loss: cross-entropy with target distribution
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()

            # Total loss
            loss = value_loss + policy_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def train_epoch(self):
        total_loss = 0
        total_vloss = 0
        total_ploss = 0
        steps = 0

        num_steps = min(
            self.config.training_steps_per_iteration,
            max(1, len(self.replay_buffer) // self.config.batch_size)
        )

        for _ in range(num_steps):
            result = self.train_step()
            if result is None:
                break
            total_loss += result["loss"]
            total_vloss += result["value_loss"]
            total_ploss += result["policy_loss"]
            steps += 1

        self.scheduler.step()

        if steps == 0:
            return None

        return {
            "loss": total_loss / steps,
            "value_loss": total_vloss / steps,
            "policy_loss": total_ploss / steps,
            "steps": steps,
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "iteration": self.iteration,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.iteration = checkpoint["iteration"]

    def run(self):
        """Main training loop: self-play -> train -> evaluate -> repeat."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for iteration in range(self.iteration, self.config.num_iterations):
            self.iteration = iteration
            iter_start = time.perf_counter()
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.config.num_iterations}")
            print(f"{'='*60}")

            # Self-play phase
            print("\n--- Self-Play ---")
            self_play_start = time.perf_counter()
            examples = run_self_play(
                self.model, self.config, self.device,
                num_games=self.config.num_self_play_games,
            )
            self_play_time = time.perf_counter() - self_play_start
            self.replay_buffer.add(examples)
            print(f"Generated {len(examples)} examples, buffer size: {len(self.replay_buffer)}")

            # Training phase
            train_time = 0.0
            if len(self.replay_buffer) >= self.config.min_replay_size:
                print("\n--- Training ---")
                # Save model before training for comparison
                prev_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                train_start = time.perf_counter()
                for epoch in range(self.config.num_epochs):
                    result = self.train_epoch()
                    if result:
                        print(f"  Epoch {epoch+1}: loss={result['loss']:.4f} "
                              f"(v={result['value_loss']:.4f}, p={result['policy_loss']:.4f}), "
                              f"steps={result['steps']}")
                train_time = time.perf_counter() - train_start

                # Arena evaluation
                arena_time = 0.0
                if self.config.arena_games > 0 and iteration > 0:
                    print("\n--- Arena ---")
                    old_model = AlphaZeroNet(self.config).to(self.device)
                    old_model.load_state_dict(prev_state)

                    arena_start = time.perf_counter()
                    win_rate = pit(
                        self.model, old_model, self.config, self.device,
                        num_games=self.config.arena_games,
                    )
                    arena_time = time.perf_counter() - arena_start
                    print(f"  New model win rate: {win_rate:.2%}")

                    if win_rate < self.config.arena_threshold:
                        print("  Rejecting new model, reverting...")
                        self.model.load_state_dict(prev_state)
                    else:
                        print("  Accepting new model!")
            else:
                arena_time = 0.0

            # Save checkpoint
            ckpt_path = os.path.join(self.config.checkpoint_dir, f"model_{iteration:04d}.pt")
            self.save_checkpoint(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            if self.config.log_perf:
                iter_total = time.perf_counter() - iter_start
                sp_pct = 100.0 * self_play_time / iter_total if iter_total > 0 else 0.0
                tr_pct = 100.0 * train_time / iter_total if iter_total > 0 else 0.0
                ar_pct = 100.0 * arena_time / iter_total if iter_total > 0 else 0.0
                print("Iteration timing: "
                      f"self-play={self_play_time:.1f}s ({sp_pct:.1f}%), "
                      f"train={train_time:.1f}s ({tr_pct:.1f}%), "
                      f"arena={arena_time:.1f}s ({ar_pct:.1f}%), "
                      f"total={iter_total:.1f}s")

        print("\nTraining complete!")


def main():
    trainer = Trainer()
    trainer.run()


if __name__ == "__main__":
    main()
