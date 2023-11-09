import torch as th
import torch.nn as nn

from typing import Tuple


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x.squeeze(self.dim)


class GaussianNoise(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        noise = th.randn_like(x)
        return noise + x
        

class FrontEnd(nn.Module):
    def __init__(self, ch, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.ELU(),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, (1, 9), padding=(0, 4)),
            nn.ELU(),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ELU(),
            nn.MaxPool2d((1, 4), (1, 4)),
            nn.Dropout(dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(dropout)
        )

    def forward(self, x: th.Tensor):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.squeeze(-1).transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size, dilation, dropout=0.):
        super().__init__()
        self.res_conv = nn.Conv1d(ch, ch, 1)
    
        padding = (kernel_size // 2) * dilation
        self.conv_a = nn.Conv1d(ch, ch, kernel_size, padding=padding, dilation=dilation)
        self.conv_b = nn.Conv1d(ch, ch, kernel_size, padding=2*padding, dilation=2*dilation)
        self.conv_c = nn.Conv1d(2*ch, ch, 1)
        
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: th.Tensor):
        res_x = self.res_conv(x)
        x_a = self.conv_a(x)
        x_b = self.conv_b(x)

        x = self.elu(th.cat([x_a, x_b], dim=1))
        x = self.dropout(x)
        x = self.conv_c(x)

        return res_x + x, x


class TCN(nn.Module):
    def __init__(
        self, channels=20, kernel_size=5,
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        dropout=0.15
    ):
        super().__init__()

        res_blocks = []
        for dilation in dilations:
            res_blocks.append(
                ResidualBlock(channels, kernel_size, dilation, dropout)
            )
        self.res_blocks = nn.ModuleList(res_blocks)
        self.activation = nn.ELU()
    
    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        skipped = 0

        for block in self.res_blocks:
            x, skip = block(x)
            skipped += skip
        
        return self.activation(x), skipped


class BeatBockNet(nn.Module):
    def __init__(
        self, channels=20, kernel_size=5, mode="beats",
        dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        dropout=0.15,
    ):
        super().__init__()

        assert mode in ["tempo", "beats", "beats+tempo"]
        self.mode = mode

        self.frontend = FrontEnd(channels, dropout)
        self.tcn = TCN(channels, kernel_size, dilations, dropout)
        
        if "tempo" in self.mode:
            self.to_tempo = nn.Sequential(
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool1d(1),
                Squeeze(-1),
                GaussianNoise(),
                nn.Linear(channels, 300),
                # nn.Softmax(-1)
            )
    
        if "beats" in self.mode:
            self.to_beat = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(channels, 1),
                # nn.Sigmoid()
            )
        

    def forward(self, x: th.Tensor):
        # x.shape: B, T, C

        x = self.frontend(x)

        x = x.transpose(1, 2)
        x, skip = self.tcn(x)
        x = x.transpose(1, 2)

        tempo = self.to_tempo(skip).squeeze(-1) if "tempo" in self.mode else None
        beats = self.to_beat(x).squeeze(-1) if "beats" in self.mode else None

        return tempo, beats
