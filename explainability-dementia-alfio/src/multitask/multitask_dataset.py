"""
Wrapper sul dataset originale per aggiungere la label Good/Bad.
Richiede i CSV prodotti da monitor_cnn.py (…_predictions_detailed.csv).
"""
import pandas as pd
import torch
from torch_geometric.data import Data
from datasets import CWTGraphDataset    # ← file originale, invariato


class MultiTaskCWTGraphDataset(CWTGraphDataset):
    def __init__(
        self,
        annot_df,
        dataset_crop_path: str,
        monitor_csv_path: str,
        thr: float = 0.5,             # soglia sul “goodness”
        norm_stats_path: str | None = None
    ):
        super().__init__(annot_df, dataset_crop_path, norm_stats_path)
        monitor_df = pd.read_csv(monitor_csv_path)
        # mappa crop_file ➜ good(1)/bad(0)
        self.good_map = {
            row.crop_file: int(row.goodness >= thr)
            for _, row in monitor_df.iterrows()
        }

    # ------------------------------------------------------------------
    def get(self, idx):
        data: Data = super().get(idx)
        crop_file = self.annot_df.iloc[idx].crop_file
        good_lbl = self.good_map.get(crop_file, 0)  # default “Bad”
        data.good_label = torch.tensor([good_lbl], dtype=torch.long)
        return data
