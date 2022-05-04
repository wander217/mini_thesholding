import os
import time
from loss_model import LossModel
import torch
import yaml
from measure.metric import DetScore
from typing import Dict, List, Tuple, OrderedDict
import numpy as np
import cv2 as cv
import math


class DBPredictor:
    def __init__(self, config: str, pretrained):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        with open(config) as f:
            config: Dict = yaml.safe_load(f)
        self._model = LossModel(**config['lossModel'], device=self.device)
        # print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        state_dict = torch.load(pretrained, map_location=self.device)
        self._model.load_state_dict(state_dict['model'])
        # multi scale problem => training
        self._score: DetScore = DetScore(**config['score'])
        self._limit: int = 960

    def _resize(self, image: np.ndarray) -> Tuple:
        org_h, org_w, _ = image.shape
        scale = min([640 / org_w, 640 / org_h])
        new_h = math.ceil(scale * org_h)
        new_w = math.ceil(scale * org_w)
        new_image = np.zeros((math.ceil(new_h / 32) * 32,
                              math.ceil(new_w / 32) * 32, 3), dtype=np.uint8)
        image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        new_image[:new_h, :new_w, :] = image
        print(new_w, new_h)
        return new_image, new_h, new_w

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        # mean = [122.67891434, 116.66876762, 104.00698793]
        image = image.astype(np.float64)
        image = (image) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image.unsqueeze(0)

    def __call__(self, image: np.ndarray) -> Tuple:
        self._model.eval()
        bboxes: List = []
        scores: List = []
        with torch.no_grad():
            h, w, _ = image.shape
            reImage, newH, newW = self._resize(image)
            inputImage = self._normalize(reImage)
            pred: OrderedDict = self._model(dict(img=inputImage, shape=[newH, newW]), training=False)
            binary = pred['binaryMap'].squeeze().cpu().detach().numpy()
            print(binary.shape)
            cv.imshow("abc", np.uint8(binary * 255))
            cv.waitKey(0)
            bs, ss = self._score(pred, dict(img=inputImage))
            for i in range(len(bs[0])):
                if ss[0][i] > 0:
                    bboxes.append(bs[0][i])
                    scores.append(ss[0][i])
            return bboxes, scores


if __name__ == "__main__":
    configPath: str = r'asset/total_text/eb0.yaml'
    pretrainedPath: str = r'asset/total_text/checkpoint_210.pth'
    # configPath: str = r'config/dbpp_eb0.yaml'
    # pretrainedPath: str = r'pretrained/eb0/checkpoint_941.pth'
    # imgPath: str = r'C:\Users\thinhtq\Downloads\vietnamese_original\vietnamese\unseen_test_images\im1999.jpg'
    imgPath: str = r'C:\Users\Trinh_Thinh\Downloads\vietnamese_original\vietnamese\test_image\im1269.jpg'
    predictor = DBPredictor(configPath, pretrainedPath)
    img = cv.imread(imgPath)
    start = time.time()
    boxes, scores = predictor(img)
    print(len(boxes))
    # for box in boxes:
    #     img = cv.polylines(img, [box], True, (0, 0, 255), 2)
    # cv.imwrite("abc.jpg", img)
    # cv.imshow("abc", img)
    # cv.waitKey(0)
    # cv.imshow("result", img)
    # cv.waitKey(0)
    # cv.imwrite("result/test.jpg", img)
    # end = time.time() - start
    # print("Process time:", end)
    # root: str = r'C:\Users\thinhtq\Downloads\pdftoimage (1)'
    # count = 0
    # for subRoot, dirs, files in os.walk(root):
    #     for file in files:
    #         if file.endswith(".jpg"):
    #             print(file)
    #             img = cv.imread(os.path.join(subRoot, file))
    #             if img is None:
    #                 count+=1
    #                 continue
    #             # kernel = np.ones((3, 3), dtype=np.uint8)
    #             # img = cv.erode(img, kernel, iterations=1)
    #             boxes, scores = predictor(img)
    #             for box in boxes:
    #                 img = cv.polylines(img, [box], True, (0, 0, 255), 2)
    #             cv.imwrite("result5/test{}.jpg".format(count), img)
    #             count += 1
