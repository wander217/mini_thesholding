import torch.nn as nn
from typing import Dict
from measure.loss.bce_loss import BceLoss
from collections import OrderedDict
from torch import Tensor


class HeedLoss(nn.Module):
    def __init__(self, heed: Dict):
        super().__init__()
        self._binaryLoss = BceLoss(**heed)

    def __call__(self, pred: Tensor, batch: OrderedDict) -> Tensor:
        binaryDist: Tensor = self._binaryLoss(pred, batch['binaryMap'])
        return binaryDist
