import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.nnmodel import NNModel
from lib.nlp.positional_encoding import PositionalEncoding

class SummarizeNet(NNModel):
    def __init__(self, device, hidden_size, input_size, num_heads, dim_feedforward_transformer, num_layers, dropout_rate):
        super(SummarizeNet, self).__init__(
            device=device,
            hidden_size=hidden_size,
            input_size=input_size,
            num_heads=num_heads,
            dim_feedforward_transformer=dim_feedforward_transformer,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )

        self.device = torch.device(device)
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.pos_encoder = PositionalEncoding(input_size).to(self.device)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=False)

        self.encode_modes = nn.Linear(1, input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward_transformer
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward_transformer
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.encode_linear = nn.Linear(input_size + 1, hidden_size)
        self.pre_decode = nn.Linear(hidden_size, input_size)

        self.discriminate = nn.Linear(hidden_size, 1)

        self.to(self.device)

    def forward(self, word_embeddings, word_embeddings_len, modes):
        batch_size, seq_len, _ = word_embeddings.shape
        expanded_modes = modes.unsqueeze(1).unsqueeze(1).expand(batch_size, seq_len, 1)

        noisy_embeddings = word_embeddings.transpose(1,0) * math.sqrt(self.input_size)
        noisy_embeddings = self.pos_encoder(noisy_embeddings)

        #noisy_embeddings = self.dropout(noisy_embeddings)

        # noisy_embeddings = torch.cat(
        #     [
        #         noisy_embeddings,
        #         expanded_modes.transpose(1,0)
        #     ],
        #     dim=2
        # )

        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.requires_grad_(False).to(self.device)

        encoded = self.transformer_encoder(
            noisy_embeddings,
            mask=mask
        )

        # predicted_modes = torch.sigmoid(
        #     self.discriminate(encoded.transpose(1,0)).mean(dim=1)
        # )

        # decoded = self.pre_decode(encoded)
        decoded = self.transformer_decoder(
            word_embeddings.transpose(1, 0),
            encoded,
            tgt_mask=mask,
            memory_mask=mask
        )
        decoded = F.normalize(decoded, dim=2)

        return decoded.transpose(1,0), modes.unsqueeze(1) # predicted_modes
