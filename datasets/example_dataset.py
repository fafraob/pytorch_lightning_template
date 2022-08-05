from types import SimpleNamespace
from typing import Any, Tuple
import torch
from torch.utils.data import Dataset
import pandas as pd


class CustomDataset(Dataset):

    def __init__(self, df_path: str, data_folder: str, cfg: SimpleNamespace, aug) -> None:
        self._df = pd.read_csv(df_path)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index) -> Tuple[Any]:
        row = self._df.iloc[index]
        x, y, label = row.x, row.y, row.label
        input = torch.tensor([x, y], dtype=torch.float32)
        target = torch.zeros(2, dtype=torch.float32)
        target[int(label)] = 1
        return input, target
