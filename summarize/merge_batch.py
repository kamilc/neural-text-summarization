import torch
import torch.nn.functional as F
import numpy as np

class MergeBatch(object):
    def __init__(self, device):
        self.device = device

    def stack_vectors(self, vectors):
        max_seq = max([vector.shape[0] for vector in vectors])

        return torch.stack(
            [
                F.pad(vector, (0, 0, 0, max_seq - vector.shape[0]))
                for vector in vectors
            ]
        )

    def __call__(self, sample):
        sample = sample.to_dict(orient="list")

        sample['word_embeddings'] = self.stack_vectors(
            sample['word_embeddings']
        ).to(self.device)

        sample['clean_word_embeddings'] = self.stack_vectors(
            sample['clean_word_embeddings']
        ).to(self.device)

        sample['mode'] = torch.tensor(
            sample['mode']
        ).to(self.device)

        sample['lengths'] = torch.tensor(
            sample['lengths']
        ).to(self.device)

        return sample
