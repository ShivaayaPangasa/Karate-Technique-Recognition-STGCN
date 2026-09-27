import torch
import torch.nn as nn


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, temporal_kernel=9):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.residual = (nn.Conv2d(in_channels, out_channels, kernel_size=1)
                          if in_channels != out_channels else nn.Identity())
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (N, C, T, V)
        res = self.residual(x)
        x = torch.einsum("nctv,vw->nctw", x, self.A)  # graph conv: aggregate over neighbors
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, A, in_channels=3, num_classes=5):
        super().__init__()
        A = torch.from_numpy(A).float()
        self.block1 = STGCNBlock(in_channels, 32, A)
        self.block2 = STGCNBlock(32, 64, A)
        self.block3 = STGCNBlock(64, 64, A)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (N, T, V, C) -> (N, C, T, V)
        x = x.permute(0, 3, 1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)