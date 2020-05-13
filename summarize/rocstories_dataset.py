import torch
import re
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset

class RocstoriesDataset(Dataset):
    def __init__(self, dataframe, mode, transforms=[]):
        if mode not in ['train', 'test', 'val']:
            raise ValueError(f"{mode} not in the set of modes of the dataset (['train', 'test', 'val'])")

        self.data = dataframe[dataframe.set == mode]
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return 6*len(self.data)

    def get_orig_text(self, row, shuffle=False):
        indices = list(range(1, 6))

        if shuffle:
            random.shuffle(indices)

        return ' '.join(
            [
                row[f"sentence{i}"]
                for i in indices
            ]
        )

    def get_text(self, row):
        sentence_columns = [ f"sentence{i}" for i in range(1,6) ]
        in_story_id = row['asked_id'] % 6

        should_shuffle = random.random() < 0.5

        if in_story_id == 0:
            return self.get_orig_text(row, shuffle=should_shuffle)
        else:
            return row[f"sentence{in_story_id}"]

    def __getitem__(self, idx):
        _idx = []

        if torch.is_tensor(idx):
            _idx = idx.tolist()

        if isinstance(idx, list):
            _idx = idx
        else:
            _idx = [ idx ]

        _ids = [ (i - (i % 6))/6 for i in _idx]
        data = self.data.iloc[_ids, :]

        data['asked_id'] = _idx

        data = pd.DataFrame(
            {
                'set': [self.mode for _ in range(0, len(_ids))],
                'mode': np.array([ (0.0 if i % 6 == 0 else 1.0) for i in _idx ]),
                'orig_text': data.apply(self.get_orig_text, axis=1),
                'orig_headline': "<none>",
                'text': data.apply(self.get_text, axis=1),
                'title': data['storytitle'],
                'idx': np.array([ i for i in _idx ]),
            }
        )

        for transform in self.transforms:
            data = transform(data)

        return data

