import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from structure.heed import Heed
import yaml
import math


def weight_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        init_range = 1 / math.sqrt(module.out_features)
        nn.init.uniform_(module.weight, -init_range, init_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ThresholdModel(nn.Module):
    def __init__(self, heed: Dict):
        super().__init__()
        self._heed: nn.Module = Heed(**heed)
        self.apply(weight_init)

    def forward(self, x: Tensor):
        y = self._heed(x)
        return y


if __name__ == "__main__":
    with open(r"D:\workspace\project\mini_thesholding\asset\total_text\eb0.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model = ThresholdModel(**config['lossModel']['model'])
    image = cv2.imread(r"D:\total_text\train\image\image0.jpg")
    h, w, c = image.shape
    new_image = np.zeros((640, 640, 3), dtype=np.uint8)
    new_image[:h, :w] = image
    in_image = torch.from_numpy(new_image).unsqueeze(0).permute(0, 3, 1, 2).float()
    start = time.time()
    out = model(in_image)
    out = out.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    cv2.imshow("abc", np.uint8(out * 255))
    cv2.waitKey(0)
    print(time.time() - start)
    # binaryMap = out['binaryMap'].squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.uint8)
    # cv2.imshow("abc", binaryMap * 255)
    # cv2.waitKey(0)
    # print(out['borderMap'].size(), out['binaryMap'].size())
    pram = sum([param.numel() for param in model.parameters()])
    print(pram)
