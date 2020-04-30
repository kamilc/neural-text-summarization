import statistics
import numpy as np
import torch

class Metrics(object):
    def __init__(self, mode, loss=None):
        if torch.is_tensor(loss):
            loss = loss.cpu().item()

        self.mode = mode
        self.losses = [loss] if loss is not None else []

    @classmethod
    def empty(cls, mode):
        return cls(mode)

    def __len__(self):
        return len(self.losses)

    @property
    def loss(self):
        if len(self.losses) == 0:
            return 0
        else:
            return statistics.mean(self.losses)

    @property
    def last_loss(self):
        return self.losses[len(self.losses) - 1]

    def running_mean_loss(self, num=1000):
        return statistics.mean(self.losses[len(self.losses)-num:])

    def __add__(self, other):
        self.losses += other.losses

        return self
