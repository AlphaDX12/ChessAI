# Chess Engine Parameter Check Script
import torch
import torch.nn as nn
from chess_engine import ChessActorCritic, OBS_SHAPE, MOVE_ACTION_DIM

def test_model(hidden_dim, pol_channels, res_blocks=10):
    class CustomChess(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = nn.Conv2d(OBS_SHAPE[0], hidden_dim, kernel_size=3, padding=1)
            self.bn_in = nn.BatchNorm2d(hidden_dim)
            
            # Simple ResBlock replica for counting
            self.res_tower = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 16, bias=False),
                        nn.Linear(hidden_dim // 16, hidden_dim, bias=False)
                    )
                ) for _ in range(res_blocks)
            ])
            
            self.policy_conv = nn.Conv2d(hidden_dim, pol_channels, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(pol_channels)
            self.policy_fc = nn.Linear(pol_channels * 8 * 8, MOVE_ACTION_DIM)
            
            self.val_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
            self.val_bn = nn.BatchNorm2d(1)
            self.val_fc1 = nn.Linear(1 * 8 * 8, 64)
            self.val_fc2 = nn.Linear(64, 1)

    m = CustomChess()
    params = sum(p.numel() for p in m.parameters())
    return params

print("Current 256, pol32:", test_model(256, 32))
print("Target: ~3.2M")
print("128 hd, pol2:", test_model(128, 2))
print("128 hd, pol1:", test_model(128, 1))
print("128 hd, pol4:", test_model(128, 4))
