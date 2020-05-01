import math
import random

class DataLoader(object):
    def __init__(self, dataset, batch_size=8):
        self.dataset = dataset
        self.batch_size = batch_size

    @property
    def epoch_size(self):
        return math.ceil(len(self.dataset) / self.batch_size) * self.batch_size

    def __iter__(self):
        ids = random.choices(range(0, len(self.dataset)), k=self.epoch_size)

        for start_ix in range(0, self.epoch_size, self.batch_size):
            end_ix = start_ix + self.batch_size
            if end_ix > len(self.dataset) - 1:
                end_ix = len(dataset) - 1
            yield self.dataset[ids[start_ix:end_ix]]
