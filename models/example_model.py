import pytorch_lightning as pl
from torch import nn, Tensor
from torchmetrics import Accuracy
import torch
from typing import Dict, Tuple, Any
from types import SimpleNamespace


class Net(pl.LightningModule):

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            nn.Linear(2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        )
        self._loss_fn = nn.BCEWithLogitsLoss()
        self._train_acc = Accuracy('binary', num_classes=2)
        self._val_acc = Accuracy('binary', num_classes=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x

    def _step(self, batch: Tensor, log_name: str, metric: Accuracy) -> Dict[str, Tensor]:
        x, y = batch
        y_pred = self(x)
        loss = self._loss_fn(y_pred, y)
        self._make_log_entry(loss, f'{log_name}_loss', on_step=False)
        metric(y_pred, y.int())
        self._make_log_entry(metric, f'{log_name}_acc', on_step=False)
        return {'loss': loss, 'outputs': y_pred, 'labels': y.to(torch.int)}

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        return self._step(batch, 'train', self._train_acc)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        return self._step(batch, 'val', self._val_acc)

    def configure_optimizers(self) -> Tuple:
        return [self.cfg.optimizer], [self.cfg.scheduler]

    def _make_log_entry(
        self, loss: Tensor, name: str = 'train_loss', on_step: bool = True,
        on_epoch: bool = True, prog_bar: bool = True, logger: bool = True
    ) -> None:
        self.log(
            name,
            loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            logger=logger
        )
