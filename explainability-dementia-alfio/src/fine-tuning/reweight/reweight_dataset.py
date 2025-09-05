"""
Dataset wrapper che restituisce:
  • x, edge_index, y  (come prima)
  • goodness          (float 0-1) per il re-weighting del loss
"""
import pandas as pd
import torch
from torch_geometric.data import Data
from datasets import CWTGraphDataset


class ReweightCWTGraphDataset(CWTGraphDataset):
    def __init__(self,
                 annot_df,
                 dataset_crop_path: str,
                 monitor_csv_path: str,
                 norm_stats_path: str | None = None):
        super().__init__(annot_df, dataset_crop_path, norm_stats_path)

        # Leggi il CSV prodotto dal monitor-CNN: deve contenere col. 'crop_file' e 'goodness'
        monitor_df = pd.read_csv(monitor_csv_path)
        self.goodness_map = {
            row.crop_file: float(row.goodness) for _, row in monitor_df.iterrows()
        }

    # ------------------------------------------------------------------
    def get(self, idx):
        data: Data = super().get(idx)

        crop_file = self.annot_df.iloc[idx].crop_file
        g         = self.goodness_map.get(crop_file, 0.0)      # default 0

        data.goodness = torch.tensor([g], dtype=torch.float32)
        return data
