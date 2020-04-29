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

    def __call__(self, sample):
        del sample['text_doc']
        del sample['headline_doc']

        sample = sample.to_dict(orient="list")

        sample['text_embeddings'] = torch.tensor(
            self.stack_vectors(
                sample['text_embeddings']
            ).astype(np.float32, copy=False)
        ).to(self.device)

        sample['headline_embeddings'] = torch.tensor(
            self.stack_vectors(
                sample['headline_embeddings']
            ).astype(np.float32, copy=False)
        ).to(self.device)

        sample['headline_embeddings'] = F.pad(
            sample['headline_embeddings'],
            (
                0,
                0,
                0,
                sample['text_embeddings'].shape[1] - sample['headline_embeddings'].shape[1]
            )
        )

        return sample
