import pytorch_lightning as pl
from torch import nn, Tensor
from torchmetrics import Accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io
import torchvision
from PIL import Image
import torch
from typing import Dict, Tuple, Any
import matplotlib
import matplotlib.pyplot as plt
from types import SimpleNamespace
matplotlib.use('Agg')


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

    def validation_epoch_end(self, outputs):
        preds = torch.cat([tmp['outputs'] for tmp in outputs])
        preds = preds.argmax(dim=1).cpu().detach().numpy()
        labels = torch.cat([tmp['labels'] for tmp in outputs])
        labels = labels.argmax(dim=1).cpu().detach().numpy()
        label_idxs = [i for i in range(self.cfg.num_classes)]
        cf_matrix = confusion_matrix(labels, preds, labels=label_idxs)
        disp = ConfusionMatrixDisplay(cf_matrix, display_labels=label_idxs)
        _, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, colorbar=False)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        cf_img = torchvision.transforms.ToTensor()(Image.open(buf))
        self.logger.experiment.add_image(
            'val_confusion_matrix',
            cf_img,
            global_step=self.current_epoch
        )
        plt.close()

    def configure_optimizers(self) -> Tuple:
        optimizer = self.cfg.optimizer
        lr_scheduler = {
            'scheduler': self.cfg.scheduler,
            'interval': self.cfg.scheduler_interval,
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx: int, metric: Any | None) -> None:
        # NOTE: required for timm schedulers to work
        scheduler.step(epoch=self.current_epoch)

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
