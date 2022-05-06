from collections import OrderedDict
import numpy as np
from typing import Dict, List
import cv2 as cv


class DetForm:
    def __init__(self, shrinkRatio: float):
        self._shrinkRatio = shrinkRatio

    def __call__(self, data: Dict, isVisual: bool = False) -> OrderedDict:
        output: OrderedDict = self._build(data)
        if isVisual:
            self._visual(output)
        return output

    def _visual(self, data: Dict):
        print(data.keys())

    def _build(self, data: Dict) -> OrderedDict:
        '''
        :param data: a dict contain : anno, img, train, target
        :return: a ordered idict contain: img, polygon, ignore, train
        '''
        polygon: List = []
        ignore: List = []
        anno: np.ndarray = data['target']
        img: np.ndarray = data['img']
        mask: np.ndarray = data['mask']
        train: bool = data['train']
        target: np.ndarray = np.zeros(img.shape[:2], dtype=np.uint8)

        for tar in anno:
            if not tar['ignore']:
                cv.fillPoly(target, [tar['polygon'].astype(np.int32)], 1)

        return OrderedDict(
            img=img,
            target=target,
            ignore=np.array(ignore, dtype=np.uint8),
            mask=mask,
            train=train)
