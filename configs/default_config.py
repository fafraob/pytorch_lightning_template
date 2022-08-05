from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# paths
cfg.name = 'example_usage'
cfg.data_dir = 'data/example_data/'
cfg.data_folder = cfg.data_dir
cfg.val_data_folder = cfg.data_dir
cfg.output_dir = 'outputs/'
cfg.train_df = cfg.data_dir + 'spiral_train.csv'
cfg.val_df = cfg.data_dir + 'spiral_val.csv'
cfg.resume_from_ckpt = None  # otherwise path to .ckpt

# dataset
cfg.dataset = 'example_dataset'
cfg.normalize = None
cfg.batch_size = 64
cfg.batch_size_val = 64
cfg.dataloader_workers = 12

# model
cfg.model = 'example_model'

# training
cfg.epochs = 20
cfg.lr = 1e-4
cfg.lr_min = 1e-7
cfg.scheduler = 'cosine'  # cosine
cfg.optimizer = 'adamw'  # adamw
cfg.optim_capturable = True
cfg.seed = -1
cfg.mixed_precision = True
cfg.device = 'cuda:0'
cfg.accelerator_type = 'gpu'
cfg.log_every_n_steps = 1
cfg.image_size = 32

cfg.train_aug = None
cfg.val_aug = None

basic_config = cfg
