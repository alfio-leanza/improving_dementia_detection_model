"""
artifact_utils.py
────────────────────────────────────────────────────────────
• detect_artifact(cwt_tensor)  → True/False  (regole rapide)
• CWTArtCNN                    → piccola rete 0=clean / 1=artifact
   * Input CWT:  (19, 40, 500)  — torch.FloatTensor
────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 1) Rilevatore euristico ultra-light
# ---------------------------------------------------------------------
def detect_artifact(cwt: torch.Tensor,
                    blink_thr: float = 5.0,
                    emg_thr: float   = 3.0,
                    line_thr: float  = 4.0,
                    spike_thr: float = 6.0) -> bool:
    """
    Parametri
    ---------
    cwt : Tensor (19, 40, 500)
        CWT canale × frequenza × tempo (float32).
    blink_thr, emg_thr, line_thr, spike_thr : float
        Moltiplicatori di soglia (alte = filtro più severo).

    Ritorna
    -------
    bool
        True se il crop è quasi certamente un artefatto.
    """
    if cwt.shape != (19, 40, 500):
        raise ValueError("CWT shape deve essere (19, 40, 500)")

    pow_map = cwt.pow(2)  # potenza

    # -------- Blink / EOG lento (Fp1/Fp2, 0–3 Hz) ----------------------
    blink = pow_map[:2, :4, :].mean(dim=(0, 1))          # (T,)
    if blink.max() > blink_thr * blink.median():
        return True

    # -------- EMG (20–200 Hz diffuso) ----------------------------------
    emg_global = pow_map[:, 25:40, :].mean()             # righe 25–39
    if emg_global > emg_thr * pow_map.median():
        return True

    # -------- Line-noise 50 Hz -----------------------------------------
    p50   = pow_map[:, 34, :].mean()                     # riga ≈50 Hz
    neigh = pow_map[:, 30:38, :].mean()
    if p50 / neigh > line_thr:
        return True

    # -------- Spike / movimento elettrodo ------------------------------
    col_var = pow_map.var(dim=(0, 1))                    # var tempo
    if col_var.max() > spike_thr * col_var.median():
        return True

    return False


# ---------------------------------------------------------------------
# 2) CNN supervisionata per classificare artefatti
# ---------------------------------------------------------------------
class CWTArtCNN(nn.Module):
    """
    Architettura minimale:
      (19,40,500) → Conv32 → Conv64 → Conv128 → GAP → FC(2)
    """
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(19, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((1, 2)),                            # (32,40,250)

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                            # (64,20,125)

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                     # (128,1,1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 2)                                # logits: 0=clean, 1=artifact
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parametri
        ---------
        x : Tensor (B, 19, 40, 500)

        Ritorna
        -------
        Tensor (B, 2)
            Logits per classi [clean, artifact].
        """
        z = self.features(x)
        return self.classifier(z)
