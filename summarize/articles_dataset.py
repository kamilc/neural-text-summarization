import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ArticlesDataset(Dataset):
    def __init__(self, dataframe, mode, transforms=[]):
        if mode not in ['train', 'test', 'val']:
            raise ValueError(f"{mode} not in the set of modes of the dataset (['train', 'test', 'val'])")

        self.data = dataframe[dataframe.set == mode]
        self.transforms = transforms
        self.mode = mode

        if len(self) == 0:
            raise ValueError(f"{mode} appears to have 0 elements")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _idx = []

        if torch.is_tensor(idx):
            _idx = idx.tolist()

        if isinstance(idx, list):
            _idx = idx
        else:
            _idx = [ idx ]

        data = self.data.iloc[_idx, :]

        data = pd.DataFrame(
            {
                'text': data.apply(lambda row: row['text'].strip().lower(), axis=1),
                'headline': data.apply(lambda row: row['headline'].strip().lower(), axis=1),
                'title': data['normalized_title'],
                'idx': np.array([ i for i in _idx ]),
            }
        )

        for transform in self.transforms:
            data = transform(data)

        return data

