from argparse import ArgumentParser
from importlib import import_module
import os
from pathlib import Path, PosixPath
from typing import Optional, Union
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.optim import Optimizer, Adam, AdamW, SGD
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from timm.scheduler import CosineLRScheduler


class TrainConfigurator():

    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument(
            '-c', '--config', help='path to config file', required=True)
        args = self.parser.parse_known_args()[0]
        self.cfg = import_module('.' + args.config, 'configs').cfg
        self.path_to_cfg = Path(__file__).parent.joinpath(
            'configs', f'{args.config}.py')
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

    def _config_optimizer(self) -> Optimizer:
        optimizer = Optimizer(self.cfg.net.parameters(), {})
        cfg = self.cfg
        name = cfg.optimizer
        if name == 'adamw':
            optimizer = AdamW(
                self.cfg.net.parameters(), cfg.lr_optim, betas=cfg.optim_betas,
                eps=cfg.optim_eps, capturable=cfg.optim_capturable)
        elif name == 'adam':
            optimizer = Adam(
                cfg.net.parameters(), cfg.lr_optim, betas=cfg.optim_betas,
                eps=cfg.optim_eps, capturable=cfg.optim_capturable)
        elif name == 'sgd':
            optimizer = SGD(cfg.net.parameters(), lr=cfg.lr_optim,
                            momentum=cfg.optim_momentum, weight_decay=cfg.optim_weight_decay)
        else:
            raise ValueError('Optimizer must be set to an allowed value.')
        return optimizer

    def _config_scheduler(self) -> Optional[_LRScheduler]:
        scheduler = None
        cfg = self.cfg
        name = cfg.scheduler
        if name == 'step':
            scheduler = StepLR(cfg.optimizer, step_size=cfg.scheduler_step_size,
                               gamma=cfg.scheduler_gamma)
        return scheduler

    def _config_dataset(self, train: bool = True) -> Dataset:
        cfg = self.cfg
        dataset = self.Dataset(
            cfg.train_df if train else cfg.val_df,
            cfg.data_folder if train else cfg.val_data_folder,
            train,
            cfg,
            cfg.train_aug if train else cfg.val_aug
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

    def save_config(self, trainer_log_dir: Union[PosixPath, str]):
        # NOTE: makes assumptions about how lightning logs are saved
        Path(trainer_log_dir).mkdir(parents=True, exist_ok=True)
        log_cfg_file = os.path.join(trainer_log_dir, f'config.py')
        with open(self.path_to_cfg, 'r') as f:
            cfg_contents = f.read()
        with open(log_cfg_file, 'w') as f:
            f.write(cfg_contents)


def main():
    tc = TrainConfigurator()
    cfg = tc.config()
    torch.set_float32_matmul_precision(
        'medium' if cfg.mixed_precision else 'high')

    pl.seed_everything(cfg.seed, workers=True)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.to_monitor, mode=cfg.to_monitor_mode,
        save_top_k=cfg.save_top_k, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=str(Path(cfg.output_dir).joinpath(cfg.name)),
        max_epochs=cfg.epochs,
        devices=cfg.devices,
        accelerator=cfg.accelerator_type,
        precision='16-mixed' if cfg.mixed_precision else 32,
        deterministic=True,
        log_every_n_steps=cfg.log_every_n_steps
    )

    tc.save_config(trainer.logger.log_dir)

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
