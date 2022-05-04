import torch.nn as nn
from typing import Dict
from measure.loss.dice_loss import DiceLoss
from collections import OrderedDict
from torch import Tensor


class HeedLoss(nn.Module):
    def __init__(self, heed: Dict):
        super().__init__()
        self._binaryLoss = DiceLoss(**heed)

    def __call__(self, pred: OrderedDict, batch: OrderedDict) -> Tensor:
        binaryDist: Tensor = self._binaryLoss(pred['binaryMap'], batch['binaryMap'])
        return binaryDist
