import os
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# paths
cfg.name = 'logs'
cfg.data_dir = os.path.join('data', 'example_data')
cfg.data_folder = cfg.data_dir
cfg.val_data_folder = cfg.data_dir
cfg.output_dir = '.'
cfg.train_df = os.path.join(cfg.data_dir, 'spiral_train.csv')
cfg.val_df = os.path.join(cfg.data_dir, 'spiral_val.csv')
cfg.resume_from_ckpt = None  # otherwise path to .ckpt
cfg.initial_weights = None  # otherwise path to .ckpt

# dataset
cfg.dataset = 'example_dataset'
cfg.num_classes = 2
cfg.dataloader_collate_fn = None
cfg.normalize = None
cfg.batch_size = 64
cfg.batch_size_val = 64
cfg.dataloader_workers = 12

# model
cfg.model = 'example_model'

# training
cfg.epochs = 30
cfg.epochs_scheduler = cfg.epochs
cfg.scheduler = 'cosine'  # cosine
cfg.lr_optim = 1e-3
cfg.warmup_lr_init = 0
cfg.lr_min = 1e-9
cfg.warmup_t = 2
cfg.cycle_decay = 1.0
cfg.cycle_limit = 1
cfg.scheduler_interval = 'step'  # epoch

cfg.optimizer = 'adamw'  # adamw, adam
cfg.optim_betas = (0.9, 0.999)
cfg.optim_eps = 1e-8
cfg.optim_capturable = True

cfg.devices = 1
cfg.seed = -1
cfg.mixed_precision = True
cfg.device = 'cuda:0'
cfg.accelerator_type = 'gpu'
cfg.log_every_n_steps = 1
cfg.to_monitor = 'val_acc'  # save best model based on this score
cfg.to_monitor_mode = 'max'  # higher is better when saving models
cfg.save_top_k = -1  # save this number of best models (-1 for all models)

cfg.train_aug = None
cfg.val_aug = None

basic_config = cfg
