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

        self.decoders_full = self._get_clones(
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

        self.decode_full_batch_norms = self._get_clones(
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
        self.linear_full_logits = nn.Linear(input_size, self.vocabulary_size)

        self.to_hidden = nn.Linear(input_size, hidden_size)
        self.from_hidden = nn.Linear(hidden_size, input_size)

        self.to(self.device)

    def mask_for(self, embeddings):
        _, seq_len, _ = embeddings.shape

        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask.requires_grad_(False).to(self.device)

    def encode(self, embeddings, mask):
        encoded = embeddings.transpose(1,0)

        for ix, encoder in enumerate(self.encoders):
            encoded = encoder(
                encoded,
                mask=mask
            )
            encoded = self.encode_batch_norms[ix](encoded.transpose(2,1)).transpose(2,1)

        last_encoded = encoded

        encoded = torch.tanh(
            self.to_hidden(encoded)
        )

        return self.to_hidden_batch_norm(encoded.transpose(2,1)).transpose(2,1), last_encoded

    def decode(self, encoded, mask, full=False):
        decoders = self.decoders if full is False else self.decoders_full
        decode_batch_norms = self.decode_batch_norms if full is False else self.decode_full_batch_norms
        linear_logits = self.linear_logits if full is False else self.linear_full_logits

        decoded = torch.tanh(
            self.from_hidden(encoded)
        )
        decoded = self.from_hidden_batch_norm(decoded.transpose(2,1)).transpose(2,1)

        for ix, decoder in enumerate(decoders):
            decoded = decoder(
                decoded,
                #last_state,
                torch.zeros_like(decoded),
                tgt_mask=mask,
                memory_mask=mask
            )
            decoded = decode_batch_norms[ix](decoded.transpose(2,1)).transpose(2,1)

        return linear_logits(decoded)

    def encode_positions(self, embeddings):
        embeddings = embeddings.transpose(1,0) * math.sqrt(self.input_size)
        return self.pos_encoder(embeddings).transpose(1,0)

    def forward(self, text_embeddings, headline_embeddings):
        #noisy_embeddings = self.dropout(word_embeddings)

        text_embeddings = self.encode_positions(text_embeddings)
        headline_embeddings = self.encode_positions(headline_embeddings)

        text_mask = self.mask_for(text_embeddings)
        headline_mask = self.mask_for(headline_embeddings)

        encoded_text, _ = self.encode(text_embeddings, text_mask)
        encoded_headline, _ = self.encode(headline_embeddings, headline_mask)

        decoded_text = self.decode(encoded_text, text_mask, full=True)
        decoded_headline = self.decode(encoded_headline, headline_mask, full=False)

        return (
            decoded_text.transpose(1,0),
            encoded_text,
            decoded_headline.transpose(1,0),
            encoded_headline,
        )
