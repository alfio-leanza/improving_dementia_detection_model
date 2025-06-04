import os, numpy as np, torch
from torch.utils.data import Dataset

class CWTRawDataset(Dataset):
    """
    Restituisce CWT normalizzato come tensore (19, 40, 500) float32.
    Non crea grafi: serve per il pre-training del CAE.
    """
    def __init__(self, annot_df, crops_dir, use_zscore: bool = True):
        self.df = annot_df
        self.crops_dir = crops_dir
        self.use_zscore = use_zscore

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cwt = np.load(os.path.join(self.crops_dir, row['crop_file']))  # (40,500,19)

        if self.use_zscore:
            mean = cwt.mean()
            std  = cwt.std() + 1e-8
            cwt  = (cwt - mean) / std

        # (40,500,19) -> (19,40,500)
        cwt = torch.tensor(cwt, dtype=torch.float32).permute(2,0,1)
        return cwt
