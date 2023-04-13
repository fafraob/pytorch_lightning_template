# Pytorch Lightning Training Template

Simple and flexible training framework template based on [Pytorch Lightning](https://www.pytorchlightning.ai/).

## Installation
There are different ways to use this template.

### Docker
- Use Docker with vscode and the Docker vscode extension (```Strg + Shift + P``` -> ```Remote-Containers: Rebuild Container```).
- Use Docker the regular way with the provided [Dockerfile](.devcontainer/Dockerfile).
- 
In both cases, you need to activate the conda environment in the container (```source /opt/conda/bin/activate base```).

### Virtual Environment
Install Python3 packages in a virtual environment, e.g.:
```shell
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Usage

After everything has been configured, training can be initialized with:
```shell
# -c or --config flag is used to specify a config file
# replace NAME_OF_CONFIG with an appropiate config file name such as default_cfg
python train.py -c NAME_OF_CONFIG
```

## Demo

As a demo, you can run 
```shell
python train.py -c default_cfg
```
which will train a binary classification on the spiral dataset.
![example_data](data/example_data/example_data.png)