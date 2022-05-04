import torch.nn as nn
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self._eps: float = eps

    def __call__(self, pred: Tensor, probMap: Tensor):
        intersection: Tensor = (pred.float() * probMap).sum()
        uninon: Tensor = probMap.sum() + pred.sum() + self._eps
        loss: Tensor = 1. - 2. * intersection / uninon
        assert loss <= 1., loss
        return loss
