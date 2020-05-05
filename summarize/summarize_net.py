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

    def encode(self, embeddings, lengths):
        batch_size, seq_len, _ = embeddings.shape

        embeddings = self.encode_positions(embeddings)

        paddings_mask = torch.arange(end=seq_len).unsqueeze(dim=0).expand((batch_size, seq_len)).to(self.device)
        paddings_mask = (paddings_mask + 1) > lengths.unsqueeze(dim=1).expand((batch_size, seq_len))

        encoded = embeddings.transpose(1,0)

        for ix, encoder in enumerate(self.encoders):
            encoded = encoder(encoded, src_key_padding_mask=paddings_mask)
            encoded = self.encode_batch_norms[ix](encoded.transpose(2,1)).transpose(2,1)

        last_encoded = encoded

        encoded = encoded[0, :, :]

        encoded = self.to_hidden(encoded)

        return encoded # self.to_hidden_batch_norm(encoded)

    def decode(self, embeddings, encoded, lengths, modes):
        batch_size, seq_len, _ = embeddings.shape

        embeddings = self.add_start_and_end_tokens(embeddings, modes)
        embeddings = self.encode_positions(embeddings)

        mask = self.mask_for(embeddings)

        encoded = self.from_hidden(encoded)

        decoded = embeddings.transpose(1,0)

        paddings_mask = torch.arange(end=seq_len+1).unsqueeze(dim=0).expand((batch_size, seq_len+1)).to(self.device)
        paddings_mask = paddings_mask > lengths.unsqueeze(dim=1).expand((batch_size, seq_len+1))

        for ix, decoder in enumerate(self.decoders):
            decoded = decoder(
                decoded,
                encoded,
                tgt_mask=mask,
                tgt_key_padding_mask=paddings_mask
            )
            decoded = self.decode_batch_norms[ix](decoded.transpose(2,1)).transpose(2,1)

        decoded = decoded.transpose(1,0)[:, 0:(decoded.shape[0] - 1), :]

        return self.linear_logits(decoded)

    def decode_prediction(self, vocabulary, encoded1xH, lengths1x):
        tokens = ['<start-short>']
        last_token = None
        seq_len = 1

        encoded1xH = self.from_hidden(encoded1xH)

        while last_token != '<end>' and seq_len < 500:
            embeddings1xSxD = vocabulary.embed(tokens).unsqueeze(dim=0).to(self.device)
            embeddings1xSxD = self.encode_positions(embeddings1xSxD)

            maskSxS = self.mask_for(embeddings1xSxD)

            decodedSx1xD = embeddings1xSxD.transpose(1,0)

            for ix, decoder in enumerate(self.decoders):
                decodedSx1xD = decoder(
                    decodedSx1xD,
                    encoded1xH,
                    tgt_mask=maskSxS,
                )
                decodedSx1xD = self.decode_batch_norms[ix](decodedSx1xD.transpose(2,1))
                decodedSx1xD = decodedSx1xD.transpose(2,1)

            decoded1x1xD = decodedSx1xD.transpose(1,0)[:, (seq_len-1):seq_len, :]
            decoded1x1xV = self.linear_logits(decoded1x1xD)

            word_id = F.softmax(decoded1x1xV[0, 0, :]).argmax().cpu().item()
            last_token = vocabulary.words[word_id]
            tokens.append(last_token)
            seq_len += 1

        return ' '.join(tokens[1:]).replace('\n', ' ').strip('❟ ❟ ❟')

    def encode_positions(self, embeddings):
        embeddings = embeddings.transpose(1,0) * math.sqrt(self.input_size)
        return self.pos_encoder(embeddings).transpose(1,0)

    def add_start_and_end_tokens(self, embeddings, modes):
        batch_size, _, dim = embeddings.shape

        modes = modes.cos().unsqueeze(dim=1).unsqueeze(dim=1)
        modes = modes.expand(batch_size, 1, dim)

        return torch.cat([modes, embeddings], dim=1)

    def forward(self, embeddings, lengths, modes):
        #noisy_embeddings = self.dropout(word_embeddings)

        encoded = self.encode(embeddings, lengths)
        decoded = self.decode(embeddings, encoded, lengths, modes)

        return (
            decoded,
            encoded
        )

    def predict(self, vocabulary, embeddings, lengths):
        previous_mode = self.training

        self.eval()

        batch_size, _, _ = embeddings.shape

        results = []

        for row in range(0, batch_size):
            encoded = self.encode(
                embeddings[row, :, :].unsqueeze(dim=0),
                lengths[row].unsqueeze(dim=0)
            )

            results.append(
                self.decode_prediction(
                    vocabulary,
                    encoded,
                    lengths[row].unsqueeze(dim=0)
                )
            )

        self.training = previous_mode

        return results

