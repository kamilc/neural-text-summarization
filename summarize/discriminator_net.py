import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.nnmodel import NNModel

class DiscriminatorNet(NNModel):
    def __init__(self, device, input_size, hidden_size):
        super(DiscriminatorNet, self).__init__(
            device=device,
            input_size=input_size,
            hidden_size=hidden_size
        )

        self.device = torch.device(device)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.discriminate_in = nn.Linear(input_size, hidden_size)
        self.discriminate_out = nn.Linear(hidden_size, 1)

    def forward(self, encoded):
        result = torch.tanh(
            self.discriminate_in(encoded)
        )

        return torch.sigmoid(
            self.discriminate_out(result)
        )
