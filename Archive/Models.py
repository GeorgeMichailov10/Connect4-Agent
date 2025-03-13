import torch
import torch.nn as nn
import torch.nn.functional as F
from MCTS import Config

class CNNModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.base = ConvBase(config)

        self.policy_head = nn.Sequential(
            nn.Conv2d(self.config.n_filters, self.config.n_filters // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.config.n_filters // 4),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear((self.config.n_filters // 4) * self.config.n_rows * self.config.n_cols, self.config.n_cols * 4),
            nn.Linear(self.config.n_cols * 4, self.config.n_cols)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(self.config.n_filters, self.config.n_filters//32, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.config.n_filters//32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.config.n_filters//32 * self.config.n_rows * self.config.n_cols, 8),
            nn.Linear(8, 1),
            nn.Tanh()
        )

        self.apply(self.__init_weights__)

    def __init_weights__(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)
        v = self.value_head(x)
        p = self.policy_head(x)
        return v, p

class ConvBase(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.base = nn.Sequential(
            nn.Conv2d(3, self.config.n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.config.n_filters),
            nn.GELU()
        )

        self.res_blocks = nn.ModuleList(
            [ResNetBlock(self.config.n_filters) for _ in range(self.config.n_res_blocks)]
        )

    def forward(self, x):
        x = self.base(x)
        for block in self.res_blocks:
            x = block(x)
        return x
    

class ResNetBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()

        self.c1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.c2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.gelu = nn.GELU()

    def forward(self, x):
        xp = self.gelu(self.bn1(self.c1(x)))
        xp = self.gelu(x + self.bn2(self.c2(xp)))
        return xp
        