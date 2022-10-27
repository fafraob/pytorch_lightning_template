
python3 -m venv venv
source venv/bin/activate

python3 -c 'import torch' 2> /dev/null || pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install pytorch-lightning timm
pip3 install numpy pandas
pip3 install opencv-python albumentations
pip3 install notebook autopep8 tensorboard tqdm
pip3 install matplotlib