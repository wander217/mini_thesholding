from torch import Tensor
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self,
                 in_channel: int,
                 hidden_channel: int,
                 out_channel: int):
        super().__init__()
        self._conv: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=hidden_channel,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channel,
                      out_channels=out_channel,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(out_channel))

    def forward(self, x: Tensor):
        y: Tensor = self._conv(x)
        y = y + x
        return y


class Heed(nn.Module):
    def __init__(self, in_channel: int, layer_num: int):
        super().__init__()
        self._weight_init: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self._residuals: nn.ModuleList = nn.ModuleList([
            Residual(32, 32, 32) for _ in range(layer_num)
        ])
        self._conv: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

    def forward(self, x: Tensor):
        y: Tensor = self._weight_init(x)
        for residual in self._residuals:
            y = residual(y)
        y = self._conv(y)
        return y
