import torch
import torch.nn as nn
import torch.nn.functional as F


class CWTEncoder(nn.Module):
    """19 × 40 × 500 → 128-d embedding."""
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(19, 32, kernel_size=3, padding=1),  # (32,40,500)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((1, 2)),                         # (32,40,250)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64,40,250)
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                         # (64,20,125)

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (128,20,125)
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                  # (128,1,1)
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):                 # x (B,19,40,500)
        z = self.conv(x).flatten(1)       # (B,128)
        z = F.normalize(self.fc(z))
        return z                          # (B,emb)


class CWTDecoder(nn.Module):
    """Ricostruisce 40 × 500 ×19 (canali in uscita = 19)."""
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 128)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5),      # (64,5,5)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(6,12), stride=(2,2)),
            nn.ReLU(),                                       # (32,16,28)
            nn.ConvTranspose2d(32, 19, kernel_size=(25,18), stride=(2,2)),
            # output ➜ (19, 40, 500)
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 1, 1)
        x = self.deconv(x)
        return x


class CAE(nn.Module):
    """Encoder+Decoder per pre-training."""
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.encoder = CWTEncoder(emb_dim)
        self.decoder = CWTDecoder(emb_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
