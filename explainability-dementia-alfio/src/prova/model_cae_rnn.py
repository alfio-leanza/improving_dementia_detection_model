import torch
import torch.nn as nn
from model_cae import CWTEncoder


class CAE_RNN(nn.Module):
    """
    • Encoder CNN (pre-addestrato) su ogni crop.
    • GRU bidirezionale sulla sequenza di embedding.
    • Head FC → 3 classi.
    """
    def __init__(self,
                 emb_dim: int = 128,
                 rnn_hidden: int = 256,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 freeze_encoder: bool = True):
        super().__init__()
        self.encoder = CWTEncoder(emb_dim)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.gru = nn.GRU(
            input_size   = emb_dim,
            hidden_size  = rnn_hidden,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= bidirectional,
            dropout = 0.3
        )

        factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
                        nn.Dropout(0.4),                # <── dropout sul vettore hidden finale
                        nn.Linear(rnn_hidden * factor, 3)
)

    def forward(self, x_seq):   # x_seq (B, T, 19, 40, 500)
        B, T = x_seq.size(0), x_seq.size(1)
        x_seq = x_seq.view(B * T, 19, 40, 500)
        with torch.set_grad_enabled(self.encoder.training):
            z = self.encoder(x_seq)              # (B*T, emb)
        z = z.view(B, T, -1)                     # (B, T, emb)

        gru_out, _ = self.gru(z)                 # (B, T, h*2)
        h_last = gru_out[:, -1, :]               # usa ultimo timestamp
        logits = self.head(h_last)
        return logits
