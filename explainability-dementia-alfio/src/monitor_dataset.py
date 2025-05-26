"""
MonitorGraphDataset che usa true_pred.csv:
    • legge per ogni crop_file se la predizione GNN è corretta (Good=1, Bad=0)
    • usa la colonna 'dataset' per sapere se il campione è train/val/test
"""

import torch, pandas as pd
from datasets import CWTGraphDataset   # tua implementazione originale

class MonitorGraphDatasetCSV(CWTGraphDataset):
    def __init__(self, annot_df, crop_dir, true_pred_csv):
        super().__init__(annot_df, crop_dir, norm_stats_path=None)
        tp = pd.read_csv(true_pred_csv)
        # etichetta Good=1 se pred==true
        tp['good_label'] = (tp['pred_label'] == tp['true_label']).astype(int)
        tp['label'] = tp['true_label']
        self.good_map    = tp.set_index('crop_file')['good_label'].to_dict()

    def get(self, idx):
        data = super().get(idx)
        crop_file = self.annot_df.iloc[idx].crop_file
        data.y = torch.tensor([self.good_map[crop_file]], dtype=torch.long)  # 0=Bad, 1=Good
        return data
