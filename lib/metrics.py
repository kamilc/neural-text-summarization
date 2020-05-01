import statistics
import numpy as np
import torch

class Metrics(object):
    def __init__(self, mode, loss=None, model_loss=None, fooling_loss=None):
        if torch.is_tensor(loss):
            loss = loss.cpu().item()

        if torch.is_tensor(model_loss):
            model_loss = model_loss.cpu().item()

        if torch.is_tensor(fooling_loss):
            fooling_loss = fooling_loss.cpu().item()

        self.mode = mode
        self.losses = [loss] if loss is not None else []
        self.model_losses = [model_loss] if model_loss is not None else []
        self.fooling_losses = [fooling_loss] if fooling_loss is not None else []

    @classmethod
    def empty(cls, mode):
        return cls(mode)

    def __len__(self):
        return len(self.losses)

    @property
    def model_loss(self):
        if len(self.model_losses) == 0:
            return 0
        else:
            return statistics.mean(self.model_losses)

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

    def running_mean_model_loss(self, num=1000):
        return statistics.mean(self.model_losses[len(self.model_losses)-num:])

    def running_mean_fooling_loss(self, num=1000):
        return statistics.mean(self.fooling_losses[len(self.fooling_losses)-num:])

    def __add__(self, other):
        self.losses += other.losses
        self.model_losses += other.model_losses
        self.fooling_losses += other.fooling_losses

        return self
