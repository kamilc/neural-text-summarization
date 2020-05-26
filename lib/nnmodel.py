from pathlib import Path
import copy
import torch
import torch.nn as nn

class NNModel(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()

        self._args = _args
        self._kwargs = _kwargs

    def save(self, path):
        torch.save(
            {
                'state': self.state_dict(),
                'args': self._args,
                'kwargs': self._kwargs
            },
            path
        )

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    @classmethod
    def load(cls, path):
        if Path(path).exists():
            data = torch.load(path)

            model = cls(*data['args'], **data['kwargs'])
            model.load_state_dict(data['state'])

            return model
        else:
            raise FileNotFoundError

