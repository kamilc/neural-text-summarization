import torch
import torch.nn.functional as F
import numpy as np

class MergeBatch(object):
    def __init__(self, device):
        self.device = device

    def stack_vectors(self, vectors):
        max_seq = max([vector.shape[0] for vector in vectors])

        return np.stack(
            [
                np.pad(vector, [(0, max_seq - vector.shape[0]), (0, 0)])
                for vector in vectors
            ]
        )

    def pad_word_ids(self, word_ids):
        max_len = max([ ids.shape[0] for ids in word_ids ])

        return [
            F.pad(ids, (0, 0, max_len - ids.shape[0], 0), mode='constant', value=0)
            for ids in word_ids
        ]

    def __call__(self, sample):
        del sample['doc']

        sample = sample.to_dict(orient="list")

        sample['word_embeddings'] = torch.from_numpy(
            self.stack_vectors(
                sample['word_embeddings']
            ).astype(np.float32, copy=False)
        ).to(self.device)

        sample['word_ids'] = torch.stack(
            self.pad_word_ids(
                sample['word_ids']
            )
        )

        if 'noisy_word_embeddings' in sample:
            sample['noisy_word_embeddings'] = torch.from_numpy(
                self.stack_vectors(
                    sample['noisy_word_embeddings']
                ).astype(np.float32, copy=False)
            ).to(self.device)

        sample['mode'] = torch.from_numpy(
            np.stack(
                sample['mode']
            ).astype(np.float32, copy=False)
        ).to(self.device)

        return sample
