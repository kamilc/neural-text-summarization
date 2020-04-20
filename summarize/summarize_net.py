import torch.nn as nn
import torch.nn.functional as F
from lib.nnmodel import NNModel
from summarize.discriminator_net import DiscriminatorNet

class SummarizeNet(NNModel):
    def __init__(self, hidden_size, input_size, num_layers):
        super(SummarizeNet, self).__init__(
            hidden_size=hidden_size,
            input_size=input_size,
            num_layers=num_layers,
        )

        self.hidden_size = hidden_size

        self.encode_gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.decode_gru = nn.GRU(
            hidden_size,
            input_size,
            num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.decode_linear = nn.Linear(input_size*2, 300)

        self.discriminate = DiscriminatorNet(num_layers * 2 * input_size)

    def take_last_pass(self, predicted):
        return predicted.reshape(
            predicted.shape[0],
            predicted.shape[1],
            2,
            int(predicted.shape[2] / 2)
        )[:, :, 1, :]

    def forward(self, word_embeddings, modes):
        predicted, _ = self.encode_gru(word_embeddings)
        predicted = self.take_last_pass(predicted)

        predicted_embeddings, state = self.decode_gru(predicted)
        predicted_embeddings = self.decode_linear(predicted_embeddings)

        predicted_modes = self.discriminate(state)

        return predicted_embeddings, predicted_modes
