import statistics
import numpy as np

class Metrics(object):
    def __init__(self, mode, loss=None):
        self.mode = mode
        self.losses = [loss.cpu().item()] if loss is not None else []

    @classmethod
    def empty(cls, mode):
        return cls(mode)

    @property
    def loss(self):
        if len(self.losses) == 0:
            return 0
        else:
            return statistics.mean(self.losses)

    @property
    def last_loss(self):
        return self.losses[len(self.losses) - 1]

    def running_mean_loss(self, n=100):
        cumsum = np.cumsum(np.insert(np.array(self.losses), 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    def __add__(self, other):
        self.losses += other.losses

        return self
