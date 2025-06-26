"""
PyG dataset per segmenti EEG (CWT 40 × 500 × 19).

- Normalizza con mean/std globali (file .npz) oppure StandardScaler.
- Opzionalmente aggiunge rumore gaussian (solo label == 1).
- Ritorna un `torch_geometric.data.Data` con:
    • x           Tensor (19,40,500)  (o (19,800) se `flatten=True`)
    • y           int64 label
    • edge_index  grafo fisso a 60 archi
    • Attributi extra: pid, crop_file, start_sec, end_sec, gt_label
"""
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import StandardScaler

__all__ = ["CWTGraphDataset"]


class CWTGraphDataset(Dataset):
    def __init__(
        self,
        annot_df,
        dataset_crop_path: str,
        norm_stats_path: str | None = None,
        augment: bool = False,
        flatten: bool = False,
    ):
        super().__init__()
        self.annot_df = annot_df
        self.crop_dir = Path(dataset_crop_path)
        self.augment = augment
        self.flatten = flatten

        # ---- normalizzazione --------------------------------------------------
        if norm_stats_path is not None:
            stats = np.load(norm_stats_path)
            self.norm_mean = stats["list_mean"].astype(np.float32)
            self.norm_std = stats["list_std"].astype(np.float32)
            self._use_scaler = False
        else:
            self.scaler = StandardScaler()
            self._use_scaler = True

        # ---- grafo (19 nodi, 60 archi) ---------------------------------------
        self.edge_index = torch.tensor(
            [
                [0, 0, 1, 1, 10, 10, 2, 2, 2, 2, 16, 16, 16, 16, 16, 3, 3, 3, 3,
                 11, 11, 12, 12, 12, 4, 4, 4, 4, 17, 17, 17, 17, 5, 5, 5, 5, 13,
                 13, 13, 14, 14, 6, 6, 6, 6, 18, 18, 18, 18, 18, 7, 7, 7, 7, 15,
                 15, 8, 8, 9, 9],
                [2, 16, 16, 3, 2, 12, 0, 16, 4, 10, 0, 1, 3, 17, 2, 1, 11, 5, 16,
                 3, 13, 10, 4, 14, 2, 17, 6, 12, 16, 5, 18, 4, 3, 13, 7, 17, 11,
                 5, 15, 12, 6, 4, 18, 8, 14, 17, 7, 9, 8, 6, 5, 15, 9, 18, 13, 7,
                 6, 18, 18, 7],
            ],
            dtype=torch.long,
        )

    # PyG vuole len() invece di __len__()
    def len(self):
        return len(self.annot_df)

    # PyG vuole get() invece di __getitem__()
    def get(self, idx):
        rec = self.annot_df.iloc[idx]
        cwt = np.load(self.crop_dir / rec["crop_file"])  # (40,500,19)

        # ---- normalizzazione --------------------------------------------------
        if self._use_scaler:
            flat = cwt.reshape(-1, 1)
            cwt = self.scaler.fit_transform(flat).reshape(cwt.shape).astype(np.float32)
        else:
            cwt = ((cwt.astype(np.float32) - self.norm_mean) / self.norm_std)

        # ---- augment (solo label 1) ------------------------------------------
        if self.augment and rec["label"] == 1:
            cwt += np.random.normal(0.0, 0.01, cwt.shape).astype(cwt.dtype)

        # ---- tensor ----------------------------------------------------------
        x = torch.from_numpy(cwt).permute(2, 0, 1).float()  # (19,40,500)
        if self.flatten:
            x = x.reshape(19, -1)                           # (19,800)

        y = torch.tensor(int(rec["label"]), dtype=torch.long)

        data = Data(x=x, edge_index=self.edge_index.clone(), y=y)

        # ---- metadati --------------------------------------------------------
        data.pid        = rec["patient_id"]
        data.crop_file  = rec["crop_file"]
        data.start_sec  = rec["start_sec"]
        data.end_sec    = rec["end_sec"]
        data.gt_label   = rec["label"]

        return data
