import os.path
import yaml
import torch
from dataset import DetLoader
from tool import DetLogger, DetAverager, DetCheckpoint
from typing import Dict, Tuple
import torch.optim as optim
import argparse
import warnings
from loss_model import LossModel


class DetTrainer:
    def __init__(self,
                 lossModel: Dict,
                 train: Dict,
                 valid: Dict,
                 optimizer: Dict,
                 checkpoint: Dict,
                 logger: Dict,
                 totalEpoch: int,
                 startEpoch: int,
                 lr: float,
                 factor: float,
                 **kwargs):
        self._device = torch.device('cpu')
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        self._model: LossModel = LossModel(**lossModel, device=self._device)
        self._train = DetLoader(**train).build()
        self._valid = DetLoader(**valid).build()
        self._checkpoint: DetCheckpoint = DetCheckpoint(**checkpoint)
        self._logger: DetLogger = DetLogger(**logger)
        optimCls = getattr(optim, optimizer['name'])
        self._lr: float = lr
        self._factor: float = factor
        self._optim: optim.Optimizer = optimCls(**optimizer['args'],
                                                lr=self._lr,
                                                params=self._model.parameters())
        self._totalEpoch: int = totalEpoch + 1
        self._startEpoch: int = startEpoch
        self._curLR: float = lr

    def _load(self):
        stateDict: Tuple = self._checkpoint.load(self._device)
        if stateDict is not None:
            self._model.load_state_dict(stateDict[0])
            self._optim.load_state_dict(stateDict[1])
            self._startEpoch = stateDict[2] + 1

    def _updateLR(self, epoch: int):
        rate: float = (1. - epoch / self._totalEpoch) ** self._factor
        self._curLR: float = rate * self._lr
        for groups in self._optim.param_groups:
            groups['lr'] = self._curLR

    def train(self):
        self._load()
        self._logger.reportDelimitter()
        self._logger.reportTime("Start")
        self._logger.reportDelimitter()
        self._logger.reportNewLine()
        for i in range(self._startEpoch, self._totalEpoch):
            self._logger.reportDelimitter()
            self._logger.reportTime("Epoch {}".format(i))
            self._updateLR(i)
            trainRS: Dict = self._trainStep()
            validRS: Dict = self._validStep()
            self._save(trainRS, validRS, i)
        self._logger.reportDelimitter()
        self._logger.reportTime("Finish")
        self._logger.reportDelimitter()

    def _trainStep(self) -> Dict:
        self._model.train()
        heedLoss: DetAverager = DetAverager()
        for i, batch in enumerate(self._train):
            self._optim.zero_grad()
            batchSize: int = batch['img'].size(0)
            pred, loss = self._model(batch)
            loss = loss.mean()
            loss.backward()
            self._optim.step()
            heedLoss.update(loss.item() * batchSize, batchSize)
        return {
            'heedLoss': heedLoss.calc(),
            'lr': self._curLR
        }

    def _validStep(self) -> Dict:
        self._model.eval()
        heedLoss: DetAverager = DetAverager()
        for batch in self._valid:
            with torch.no_grad():
                batchSize: int = batch['img'].size(0)
                pred, loss = self._model(batch)
                heedLoss.update(loss.mean().item() * batchSize, batchSize)
            return {
                'heedLoss': heedLoss.calc()
            }

    def _save(self, trainRS: Dict, validRS: Dict, epoch: int):
        self._logger.reportMetric("training", trainRS)
        self._logger.reportMetric("validation", validRS)
        self._logger.writeFile({
            'training': trainRS,
            'validation': validRS
        })
        self._checkpoint.saveCheckpoint(epoch, self._model, self._optim)
        self._checkpoint.saveModel(self._model, epoch)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument("-c", '--config', type=str, help="path of config file")
    parser.add_argument("-d", '--root', default='', type=str, help="path of root")
    parser.add_argument("-i", '--imgType', default=0, type=int, help="type of image")
    parser.add_argument("-r", '--resume', default='', type=str, help="resume path")
    args = parser.parse_args()
    with open(args.config) as f:
        config: Dict = yaml.safe_load(f)
    if args.root.strip():
        root = args.root.strip()
        for item in ["train", 'valid']:
            config[item]['dataset']['imgDir'] = os.path.join(root, "{}/image/".format(item))
            config[item]['dataset']['maskDir'] = os.path.join(root, "{}/mask/".format(item))
            config[item]['dataset']['tarFile'] = os.path.join(root, "{}/target.json".format(item))
            config[item]['dataset']['imgType'] = args.imgType
    if args.resume.strip():
        config['checkpoint']['resume'] = args.resume.strip()
    trainer = DetTrainer(**config)
    trainer.train()
