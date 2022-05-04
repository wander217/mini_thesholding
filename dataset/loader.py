import random
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
from collections import OrderedDict
import dataset.prep as prep_module
import os
import json
import numpy as np
import cv2 as cv


class DetDataset(Dataset):
    def __init__(self, imgDir: str, maskDir: str, imgType: int, tarFile: str, prep: Dict):
        # image dir
        self._imgDir: str = imgDir
        # mask dir
        self._maskDir: str = maskDir
        # target file
        self._tarFile: str = tarFile
        # is training
        self._train: bool = 'train' in imgDir
        # image type: 0:jpg, 1:numpy
        self._imgType: int = imgType
        # preprocessing
        self._prep: List = []
        # loading prep module
        if prep is not None:
            for key, item in prep.items():
                cls = getattr(prep_module, key)
                self._prep.append(cls(**item))

        self._imgPath: List = []
        self._maskPath: List = []
        self._target: List = []
        self._loadData()

    def _loadData(self):
        # loading annotation
        with open(self._tarFile, 'r', encoding='utf-8') as file:
            annos = json.loads(file.readline().strip('\n').strip('\r\t').strip())
        # loading image path
        for anno in annos:
            self._imgPath.append(os.path.join(self._imgDir, anno['file_name']))
            self._maskPath.append(os.path.join(self._maskDir, anno['file_name']))
            polygons: List = [tar for tar in anno['target']]
            self._target.append(polygons)

    def _loadImage(self, imgPath: str):
        if self._imgType == 1:
            return np.load(imgPath)
        return cv.imread(imgPath, cv.IMREAD_COLOR)

    def __getitem__(self, index: int, isVisual: bool = False) -> OrderedDict:
        data: OrderedDict = OrderedDict()
        imgPath: str = self._imgPath[index]
        maskPath: str = self._maskPath[index]
        image: np.ndarray = self._loadImage(imgPath)
        mask: np.ndarray = self._loadImage(maskPath)
        data['train'] = self._train
        data['target'] = self._target[index]
        data['img'] = image
        data['mask'] = mask
        for proc in self._prep:
            data = proc(data, isVisual)
        return data
        # try:
        #     for proc in self._prep:
        #         data = proc(data, isVisual)
        #     return data
        # except Exception as e:
        #     print(e)
        #     return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self._imgPath)


class DetCollate:
    def __init__(self):
        pass

    def __call__(self, batch: Tuple) -> OrderedDict:
        imgs: List = []
        binaryMaps: List = []
        output: OrderedDict = OrderedDict()
        for element in batch:
            imgs.append(element['img'])
            binaryMaps.append(element['mask'])
        output.update(
            img=torch.from_numpy(np.asarray(imgs, dtype=np.float64)).float(),
            binaryMap=torch.from_numpy(np.asarray(binaryMaps, dtype=np.float64)).float())
        return output


class DetLoader:
    def __init__(self,
                 dataset: Dict,
                 numWorkers: int,
                 batchSize: int,
                 dropLast: bool,
                 shuffle: bool,
                 pinMemory: bool):
        self._dataHolder: DetDataset = DetDataset(**dataset)
        self._numWorkers: int = numWorkers
        self._batchSize: int = batchSize
        self._dropLast: bool = dropLast
        self._shuffle: bool = shuffle
        self._pinMemory: bool = pinMemory
        self._collate = DetCollate()

    def build(self):
        return DataLoader(
            dataset=self._dataHolder,
            batch_size=self._batchSize,
            num_workers=self._numWorkers,
            drop_last=self._dropLast,
            shuffle=self._shuffle,
            pin_memory=self._pinMemory,
            collate_fn=self._collate
        )
