import os, numpy as np, torch
from torch.utils.data import Dataset


class CWTSequenceDataset(Dataset):
    """
    Crea finestre di lunghezza `seq_len` sovrapposte (stride 1) per soggetto.
    `annot_df` deve contenere: crop_file, label, original_rec, start_sample.
    """
    def __init__(self, annot_df, crops_dir, seq_len: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.crops_dir = crops_dir

        # raggruppa per soggetto e ordina per start_sample
        windows = []
        for rec_id, group in annot_df.groupby('original_rec'):
            g = group.sort_values('start_sample')
            files  = g['crop_file'].tolist()
            labels = g['label'].tolist()
            for i in range(0, len(files) - seq_len + 1):
                win_files  = files[i:i+seq_len]
                win_labels = labels[i:i+seq_len]
                windows.append((win_files, win_labels))
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        files, labels = self.windows[idx]
        crops = [np.load(os.path.join(self.crops_dir, f)) for f in files]
        # to tensor (F,T,19) -> (19,F,T)
        crops = [torch.tensor(c).permute(2,0,1) for c in crops]
        x_seq = torch.stack(crops)               # (T,19,40,500)
        y_seq = torch.tensor(labels[self.seq_len // 2])  # label centrale
        return x_seq.float(), y_seq.long()
