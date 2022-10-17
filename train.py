from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path
from typing import Dict
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler as LRSched
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


class TrainConfigurator():

    def __init__(self):
        self._parser = ArgumentParser()
        self._parser.add_argument(
            '-c', '--config', help='path to config file', required=True)
        args = self._parser.parse_known_args()[0]
        self.cfg = import_module('.' + args.config, 'configs').cfg
        self.Net = import_module('.' + self.cfg.model, 'models').Net
        self.Dataset = import_module(
            '.' + self.cfg.dataset, 'datasets').CustomDataset

    def config(self):
        cfg = self.cfg
        cfg.net = self.Net(cfg)
        cfg.net_class = self.Net
        cfg.train_dataset = self._config_dataset(train=True)
        cfg.val_dataset = self._config_dataset(train=False)
        cfg.train_loader = self._config_dataloader(cfg.train_dataset, True)
        cfg.val_loader = self._config_dataloader(cfg.val_dataset, False)
        cfg.optimizer = self._config_optimizer()
        cfg.scheduler = self._config_scheduler()
        cfg.seed = self._config_seed()
        return cfg

    def _config_optimizer(self):
        if self.cfg.optimizer == 'adamw':
            optimizer = optim.AdamW(
                self.cfg.net.parameters(), self.cfg.lr, capturable=self.cfg.optim_capturable)
        else:
            raise ValueError('Optimizer must be set to an allowed value.')
        return optimizer

    def _config_scheduler(self) -> LRSched:
        if self.cfg.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.cfg.optimizer, self.cfg.epochs, self.cfg.lr_min)
        else:
            scheduler = None
        return scheduler

    def _config_dataset(self, train: bool = True) -> Dataset:
        dataset = self.Dataset(
            self.cfg.train_df if train else self.cfg.val_df,
            self.cfg.data_folder if train else self.cfg.val_data_folder,
            train,
            self.cfg,
            self.cfg.train_aug if train else self.cfg.val_aug
        )
        return dataset

    def _config_dataloader(self, dataset: Dataset, train: bool = True) -> DataLoader:
        dataloader = DataLoader(
            dataset,
            shuffle=train,
            batch_size=self.cfg.batch_size if train else self.cfg.batch_size_val,
            drop_last=False,
            num_workers=self.cfg.dataloader_workers,
            collate_fn=self.cfg.dataloader_collate_fn
        )
        return dataloader

    def _config_seed(self):
        if self.cfg.seed < 0:
            seed = np.random.randint(1_000_000)
        else:
            seed = self.cfg.seed
        return seed


def batch_to_device(batch: Dict, device: str):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def main():
    cfg = TrainConfigurator().config()
    pl.seed_everything(cfg.seed, workers=True)
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.to_monitor, mode=cfg.to_monitor_mode,
        save_top_k=cfg.save_top_k, save_last=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        default_root_dir=Path(cfg.output_dir).joinpath(cfg.name),
        max_epochs=cfg.epochs,
        devices=1,
        accelerator=cfg.accelerator_type,
        precision=16 if cfg.mixed_precision else 32,
        deterministic=True,
        log_every_n_steps=cfg.log_every_n_steps
    )
    if cfg.initial_weights:
        checkpoint = torch.load(cfg.initial_weights)
        cfg.net.load_state_dict(checkpoint['state_dict'])
    trainer.fit(
        model=cfg.net,
        train_dataloaders=cfg.train_loader,
        val_dataloaders=cfg.val_loader,
        ckpt_path=cfg.resume_from_ckpt
    )


if __name__ == '__main__':
    main()
