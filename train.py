from argparse import ArgumentParser
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path, PosixPath
import sys
from typing import Optional, Union
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim import Optimizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch


class TrainConfigurator():

    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument(
            '-c', '--config', help='path to config file', required=True)
        args = self.parser.parse_known_args()[0]
        self.cfg = import_module('.' + args.config, 'configs').cfg
        self.raw_cfg = deepcopy(self.cfg.__dict__)
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
        name = self.cfg.optimizer
        if name == 'adamw':
            optimizer = optim.AdamW(
                self.cfg.net.parameters(), self.cfg.lr, betas=self.cfg.optim_betas,
                eps=self.cfg.optim_eps, capturable=self.cfg.optim_capturable)
        elif name == 'adam':
            optimizer = torch.optim.Adam(
                self.cfg.net.parameters(), self.cfg.lr, betas=self.cfg.optim_betas,
                eps=self.cfg.optim_eps, capturable=self.cfg.optim_capturable)
        else:
            raise ValueError('Optimizer must be set to an allowed value.')
        return optimizer

    def _config_scheduler(self) -> Optional[_LRScheduler]:
        scheduler = None
        name = self.cfg.scheduler
        if name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.cfg.optimizer, self.cfg.epochs, self.cfg.lr_min)
        elif name == 'noamopt':
            # from http://nlp.seas.harvard.edu/annotated-transformer/
            def rate(step, model_size, factor, warmup):
                if step == 0:
                    step = 1
                return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
            scheduler = LambdaLR(
                optimizer=self.cfg.optimizer,
                lr_lambda=lambda step: rate(
                    step, model_size=self.cfg.d_model, factor=1.0,
                    warmup=self.cfg.optim_warmup
                ),
            )
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

    def save_config_in_logs(self, trainer_root_dir: Union[PosixPath, str]):
        # makes assumptions about how lightning logs are saved
        max_version = -1
        path = Path(trainer_root_dir).joinpath('lightning_logs')
        path.mkdir(parents=True, exist_ok=True)
        for log_dir in path.glob('version_*'):
            try:
                version = int(log_dir.name.split('_')[-1])
                max_version = max(max_version, version)
            except ValueError:
                continue
            except Exception as e:
                print(e)
                sys.exit(1)
        next_log_version = max_version + 1
        log_cfg_file = path.joinpath(f'version_{next_log_version}_cfg.json')
        with open(log_cfg_file, 'w') as f:
            f.write(json.dumps(self.raw_cfg, indent=4))


def main():
    tc = TrainConfigurator()
    cfg = tc.config()

    pl.seed_everything(cfg.seed, workers=True)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.to_monitor, mode=cfg.to_monitor_mode,
        save_top_k=cfg.save_top_k, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer_root_dir = str(Path(cfg.output_dir).joinpath(cfg.name))
    tc.save_config_in_logs(trainer_root_dir)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=trainer_root_dir,
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
