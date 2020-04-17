import torch.nn as nn
import torch.nn.functional as F
from lib.nnmodel import NNModel

class DiscriminatorNet(NNModel):
    def __init__(self, input_size):
        super(DiscriminatorNet, self).__init__()

        self.input_size = input_size

        self.linear = nn.Linear(input_size, 1)

    def forward(self, state):
        """
        The forward pass for the network

        hidden_state : tensor (batch_num, hidden_size)

        returns         : tensor (batch_num, 1)
        """

        state = state.transpose(0, 1).reshape(-1, self.input_size)
        state = self.linear(state)
        state = F.sigmoid(state)

        return state

