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

    def __len__(self):
        return 2*len(self.data)

    def __getitem__(self, idx):
        _idx = []

        if torch.is_tensor(idx):
            _idx = idx.tolist()

        if isinstance(idx, list):
            _idx = idx
        else:
            _idx = [ idx ]

        _ids = [ (i - (i % 2))/2 for i in _idx]

        data = self.data.iloc[_ids, :]
        data['asked_id'] = _idx

        data = pd.DataFrame(
            {
                'set': [self.mode for _ in range(0, len(_ids))],
                'mode': np.array([ (0.0 if i % 2 == 0 else 1.0) for i in _idx ]),
                'orig_text': data['text'],
                'orig_headline': data['headline'],
                'text': data.apply(lambda row: row['text'].strip().lower() if row['set'] == 'test' or row['asked_id'] % 2 == 0 else row['headline'].strip().lower(), axis=1),
                'title': data['normalized_title'],
                'idx': np.array([ i for i in _idx ]),
            }
        )

        for transform in self.transforms:
            data = transform(data)

        return data

