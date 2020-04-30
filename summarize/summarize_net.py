import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.nnmodel import NNModel
from lib.nlp.positional_encoding import PositionalEncoding

class SummarizeNet(NNModel):
    def __init__(self, device, hidden_size, input_size,
                 num_heads, dim_feedforward_transformer,
                 num_layers, dropout_rate,
                 vocabulary_size):
        super(SummarizeNet, self).__init__(
            device=device,
            hidden_size=hidden_size,
            input_size=input_size,
            num_heads=num_heads,
            dim_feedforward_transformer=dim_feedforward_transformer,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            vocabulary_size=vocabulary_size
        )

        self.device = torch.device(device)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.vocabulary_size = vocabulary_size

        self.pos_encoder = PositionalEncoding(input_size).to(self.device)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)

        self.encode_modes = nn.Linear(input_size+1, input_size)
        self.to_input_batch_norm = nn.BatchNorm1d(
            num_features=input_size
        )

        self.encoders = self._get_clones(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=input_size,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward_transformer
                ),
                num_layers=1
            ),
            num_layers
        )

        self.decoders = self._get_clones(
            nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=input_size,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward_transformer
                ),
                num_layers=1
            ),
            num_layers
        )

        self.encode_batch_norms = self._get_clones(
            nn.BatchNorm1d(
                num_features=input_size
            ),
            num_layers
        )

        self.decode_batch_norms = self._get_clones(
            nn.BatchNorm1d(
                num_features=input_size
            ),
            num_layers
        )

        self.to_hidden_batch_norm = nn.BatchNorm1d(
            num_features=hidden_size
        )

        self.from_hidden_batch_norm = nn.BatchNorm1d(
            num_features=input_size
        )

        self.discriminate = nn.Linear(hidden_size, 1)
        self.linear_logits = nn.Linear(input_size, self.vocabulary_size)
        self.to_hidden = nn.Linear(input_size, hidden_size)
        self.from_hidden = nn.Linear(hidden_size, input_size)

        self.to(self.device)

    def forward(self, word_embeddings, modes):
        """
        Shapes:
          word_embeddings: [B, T, D]
          modes:           [B]
          returns:         [B, T, V]

        Where:
          B = batch size
          T = max sequence length
          D = embeddings dimentionality
          V = vocabulary size
        """
        batch_size, seq_len, _ = word_embeddings.shape
        expanded_modes = modes.unsqueeze(1).unsqueeze(1).expand(batch_size, seq_len, 1)

        noisy_embeddings = self.dropout(word_embeddings)
        noisy_embeddings = noisy_embeddings.transpose(1,0) * math.sqrt(self.input_size)
        noisy_embeddings = self.pos_encoder(noisy_embeddings)

        # noisy_embeddings = torch.cat(
        #     [
        #         expanded_modes.transpose(1,0),
        #         noisy_embeddings
        #     ],
        #     dim=2
        # )

        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.requires_grad_(False).to(self.device)

        encoded = noisy_embeddings

        for ix, encoder in enumerate(self.encoders):
            encoded = encoder(
                encoded,
                mask=mask
            )
            encoded = self.encode_batch_norms[ix](encoded.transpose(2,1)).transpose(2,1)

        last_encoded = encoded

        encoded = torch.cat(
            [
                expanded_modes.transpose(1,0),
                encoded
            ],
            dim=2
        )
        encoded = self.encode_modes(encoded)
        encoded = self.to_input_batch_norm(
            encoded.transpose(2,1)
        ).transpose(2,1)

        encoded = torch.tanh(
            self.to_hidden(encoded)
        )

        self.to_hidden_batch_norm(encoded.transpose(2,1)).transpose(2,1)

        predicted_modes = torch.sigmoid(
            self.discriminate(encoded.transpose(1,0)).mean(dim=1)
        )

        # decoded = self.pre_decode(encoded)
        decoded = torch.tanh(
            self.from_hidden(encoded)
        )
        decoded = self.from_hidden_batch_norm(decoded.transpose(2,1)).transpose(2,1)

        for ix, decoder in enumerate(self.decoders):
            decoded = decoder(
                decoded,
                last_encoded,
                tgt_mask=mask,
                memory_mask=mask
            )
            decoded = self.decode_batch_norms[ix](decoded.transpose(2,1)).transpose(2,1)

        decoded = self.linear_logits(decoded)

        return decoded.transpose(1,0), predicted_modes
