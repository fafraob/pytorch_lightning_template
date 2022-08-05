import pytorch_lightning as pl
from torch import nn
from torchmetrics import Accuracy


class Net(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self._model = nn.Sequential(
            nn.Linear(2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        )
        self._loss_fn = nn.BCEWithLogitsLoss()
        self._train_acc = Accuracy(num_classes=2)
        self._val_acc = Accuracy(num_classes=2)

    def forward(self, x, y=None):
        x = self._model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self._loss_fn(y_pred, y)
        self._make_log_entry(loss, 'train_loss', on_step=False)
        self._train_acc(y_pred, y.int())
        self._make_log_entry(self._train_acc, 'train_acc', on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self._loss_fn(y_pred, y)
        self._make_log_entry(loss, 'val_loss', on_step=False)
        self._val_acc(y_pred, y.int())
        self._make_log_entry(self._val_acc, 'val_acc', on_step=False)
        return loss

    def configure_optimizers(self):
        out = {'optimizer': self._cfg.optimizer}
        scheduler = self._cfg.scheduler
        if scheduler is not None:
            out['lr_scheduler'] = scheduler
        return out

    def _make_log_entry(
        self, loss, name='train_loss', on_step=True,
        on_epoch=True, prog_bar=True, logger=True
    ):
        self.log(
            name,
            loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            logger=logger
        )
