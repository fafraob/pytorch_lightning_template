import sys
import cv2
import numpy as np
import torch
from torch import tensor


def load_img(path: str) -> np.ndarray:
    try:
        img = cv2.imread(path)[:, :, ::-1]
    except Exception as e:
        print('Error reading image', path, ':', e)
        sys.exit(1)
    return img


def normalize_img(img: np.ndarray, norm_type: str) -> np.ndarray:
    if norm_type == 'channel':
        # print(img.shape)
        pixel_mean = img.mean((0, 1))
        pixel_std = img.std((0, 1)) + 1e-4
        img = (img - pixel_mean[None, None, :]) / pixel_std[None, None, :]
        img = img.clip(-20, 20)

    elif norm_type == 'image':
        img = (img - img.mean()) / (img.std() + 1e-4)
        img = img.clip(-20, 20)

    elif norm_type == 'simple':
        img = img/255

    elif norm_type == 'inception':
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img = img.astype(np.float32)
        img = img/255.
        img -= mean
        img *= np.reciprocal(std, dtype=np.float32)

    elif norm_type == 'imagenet':
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= np.reciprocal(std, dtype=np.float32)

    elif norm_type == 'min_max':
        img = img - np.min(img)
        img = img / np.max(img)
        return img

    else:
        pass

    return img


def to_torch_tensor(img: np.ndarray) -> tensor:
    return torch.from_numpy(img.transpose((2, 0, 1)))


def augment(img: np.ndarray, aug) -> np.ndarray:
    img_aug = aug(image=img)['image']
    return img_aug
