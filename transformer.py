import torch
import torch.nn as nn
import math


class Transformer(nn.Module):
    def __init__(self, opt, num_classes, d_model=256, nhead=4, num_encoder_layers=3, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(num_classes, d_model)
        self.positional_encoding = self._generate_positional_encoding(max_len, d_model)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout), num_encoder_layers)
        self.decoder = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src + self.positional_encoding[:src.size(0), :]
        output = self.encoder(src)
        output = self.decoder(output[-1])  # Assuming prediction based on the last state
        return output
