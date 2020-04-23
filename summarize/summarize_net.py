import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.nnmodel import NNModel

class SummarizeNet(NNModel):
    def __init__(self, hidden_size, input_size, num_heads, num_layers, dropout_rate):
        super(SummarizeNet, self).__init__(
            hidden_size=hidden_size,
            input_size=input_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)

        self.encode_modes = nn.Linear(1, input_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encode_linear = nn.Linear(input_size + 1, hidden_size)

        self.decode = nn.Linear(hidden_size, input_size)

        self.discriminate = nn.Linear(hidden_size, 1)

    def forward(self, word_embeddings, modes):
        noisy_embeddings = self.dropout(word_embeddings)

        batch_size, seq_len, _ = word_embeddings.shape

        expanded_modes = modes.unsqueeze(1).unsqueeze(1).expand(batch_size, seq_len, 1)

        encoded = self.transformer_encoder(noisy_embeddings)
        encoded = torch.cat([encoded, expanded_modes], dim=2)

        encoded = F.tanh(
            self.encode_linear(encoded)
        )

        encoded = self.batch_norm(encoded.transpose(2, 1)).transpose(2, 1)

        predicted_modes = F.sigmoid(
            self.discriminate(encoded).mean(dim=1)
        )

        decoded = self.decode(encoded)
        decoded = F.normalize(decoded, dim=2)

        return decoded, predicted_modes
