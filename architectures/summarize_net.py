import torch.nn as nn
import torch.nn.functional as F
from lib.nnmodel import NNModel
from architectures.discriminator_net import DiscriminatorNet

class SummarizeNet(NNModel):
    def __init__(self, hidden_size, input_size, num_layers, vocabulary_size, cutoffs):
        super(SummarizeNet, self).__init__(
            hidden_size=hidden_size,
            input_size=input_size,
            num_layers=num_layers,
            vocabulary_size=vocabulary_size
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

        self.discriminate = DiscriminatorNet(num_layers * 2 * input_size)

        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=input_size,
            n_classes=vocabulary_size,
            cutoffs=cutoffs
        )

    def take_last_pass(self, predicted):
        return predicted.reshape(
            predicted.shape[0],
            predicted.shape[1],
            2,
            int(predicted.shape[2] / 2)
        )[:, :, 1, :]

    def forward(self, word_embeddings, modes, target_probs):
        predicted, _ = self.encode_gru(word_embeddings)
        predicted = self.take_last_pass(predicted)

        predicted, state = self.decode_gru(predicted)
        predicted = self.take_last_pass(predicted)

        # pdb.set_trace()
        predicted_probs, loss = self.adaptive_softmax(predicted, target_probs)

        predicted_modes = self.discriminate(state)

        return predicted_probs, predicted_modes, loss
