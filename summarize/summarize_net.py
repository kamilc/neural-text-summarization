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

        self.linear_logits = nn.Linear(input_size, self.vocabulary_size)

        self.to_hidden = nn.Linear(input_size, hidden_size)
        self.from_hidden = nn.Linear(hidden_size, input_size)

        self.to(self.device)

    def mask_for(self, embeddings):
        _, seq_len, _ = embeddings.shape

        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask.requires_grad_(False).to(self.device)

    def encode(self, embeddings):
        encoded = embeddings.transpose(1,0)

        for ix, encoder in enumerate(self.encoders):
            encoded = encoder(encoded)
            encoded = self.encode_batch_norms[ix](encoded.transpose(2,1)).transpose(2,1)

        last_encoded = encoded

        encoded = torch.tanh(
            self.to_hidden(encoded)
        )

        return self.to_hidden_batch_norm(
            encoded.transpose(2,1)
        ).transpose(2,1)[0, :, :]

    def decode(self, encoded, mask, modes):
        encoded = encoded.unsqueeze(axis=1).expand(
            modes.shape[0], mask.shape[0], encoded.shape[1]
        )

        decoded = torch.tanh(
            self.from_hidden(encoded)
        )

        decoded = self.from_hidden_batch_norm(
            decoded.transpose(2,1)
        ).transpose(2,1)

        decoded = decoded.transpose(1,0)

        for ix, decoder in enumerate(self.decoders):
            decoded = decoder(
                decoded,
                torch.zeros_like(decoded),
                tgt_mask=mask
            )
            decoded = self.decode_batch_norms[ix](decoded.transpose(2,1)).transpose(2,1)

        return self.linear_logits(decoded.transpose(1,0))

    def encode_positions(self, embeddings):
        embeddings = embeddings.transpose(1,0) * math.sqrt(self.input_size)
        return self.pos_encoder(embeddings).transpose(1,0)

    def forward(self, embeddings, modes):
        #noisy_embeddings = self.dropout(word_embeddings)

        embeddings = self.encode_positions(embeddings)

        mask = self.mask_for(embeddings)

        encoded = self.encode(embeddings)

        decoded = self.decode(encoded, mask, modes)

        return (
            decoded,
            encoded
        )
