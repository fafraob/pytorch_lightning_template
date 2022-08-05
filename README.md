# Pytorch Lightning Training Template

Simple and flexible training framework template based on [Pytorch Lightning](https://www.pytorchlightning.ai/).

## Installation
Install Python3 packages in a virtual environment. The file scripts/setup_env.sh may be adapted according to one's needs.
```shell
source scripts/setup_env.sh
```

## Usage

After everything has been configured, training can be initialized with:
```shell
# -c or --config flag is used to specify a config file
# replace NAME_OF_CONFIG with an appropiate config file name such as default_config
python train.py -c NAME_OF_CONFIG
```

## Demo

As a demo, you can run 
```shell
python train.py -c default_config
```
which will train a binary classification on the spiral dataset.
![example_data](data/example_data/example_data.png)