import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, opt, num_classes, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dropout=0.1):
        super(Transformer, self).__init__()

        # Parameters similar to GRU and LSTM models
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

        # Embedding layer
        self.embedding = nn.Embedding(num_classes, d_model)

        # Transformer specific layers
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=dropout)

        # Decoder to map Transformer output to class labels
        self.decoder = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, src, tgt):
        # Assuming src and tgt are provided with proper padding and batch dimensions
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # Forward pass through the Transformer
        output = self.transformer(src, tgt)

        # Pass through the decoder
        output = self.decoder(output)

        return output
