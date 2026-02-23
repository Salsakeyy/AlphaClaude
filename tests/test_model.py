"""Test neural network model shapes and behavior."""
import torch
from py_src.model import AlphaZeroNet, get_device
from py_src.config import AlphaClaudeConfig


def test_model_output_shapes():
    config = AlphaClaudeConfig()
    model = AlphaZeroNet(config)
    x = torch.randn(4, 119, 8, 8)
    p, v = model(x)
    assert p.shape == (4, 4672), f"Policy shape {p.shape} != (4, 4672)"
    assert v.shape == (4, 1), f"Value shape {v.shape} != (4, 1)"


def test_value_in_range():
    config = AlphaClaudeConfig()
    model = AlphaZeroNet(config)
    x = torch.randn(16, 119, 8, 8)
    _, v = model(x)
    assert v.min() >= -1.0, f"Value min {v.min()} < -1"
    assert v.max() <= 1.0, f"Value max {v.max()} > 1"


def test_single_batch():
    config = AlphaClaudeConfig()
    model = AlphaZeroNet(config)
    x = torch.randn(1, 119, 8, 8)
    p, v = model(x)
    assert p.shape == (1, 4672)
    assert v.shape == (1, 1)


def test_overfit_single_example():
    """Train on a single example to verify loss decreases."""
    config = AlphaClaudeConfig()
    model = AlphaZeroNet(config)

    x = torch.randn(1, 119, 8, 8)
    target_p = torch.zeros(1, 4672)
    target_p[0, 42] = 1.0
    target_v = torch.tensor([0.5])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial_loss = None

    for i in range(100):
        p, v = model(x)
        v = v.squeeze(-1)
        vloss = torch.nn.functional.mse_loss(v, target_v)
        ploss = -torch.sum(target_p * torch.log_softmax(p, dim=1), dim=1).mean()
        loss = vloss + ploss
        if i == 0:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, \
        f"Loss didn't decrease enough: {initial_loss:.4f} -> {final_loss:.4f}"


if __name__ == "__main__":
    import sys
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    sys.exit(1 if failed else 0)
