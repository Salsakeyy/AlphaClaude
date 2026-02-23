import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AlphaClaudeConfig


class ResBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(self, config: AlphaClaudeConfig = None):
        super().__init__()
        if config is None:
            config = AlphaClaudeConfig()

        nf = config.num_filters

        # Input conv
        self.input_conv = nn.Conv2d(config.input_planes, nf, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(nf)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(nf) for _ in range(config.num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(nf, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, config.policy_size)

        # Value head
        self.value_conv = nn.Conv2d(nf, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch, 119, 8, 8)
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.res_blocks(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # raw logits (batch, 4672)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # (batch, 1)

        return p, v


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
