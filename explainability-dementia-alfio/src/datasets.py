import os
import torch
import ipdb
import scipy
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import StandardScaler


__all__ = ["CWTGraphDataset"]


def noise_augmentation(noise_type, features, noise_mean=0.0, noise_std=1e-3):
    # 'noise_type' should be 'add' for additive gaussian noise or 'mult' for multiplicative gaussian noise.
    # 'features' should be a float32 torch tensor.
    noise = torch.normal(mean=torch.ones_like(features) * noise_mean, std=torch.ones_like(features) * noise_std)
    if noise_type == 'mult':
        noisy_features = features + (features * noise)
    elif noise_type == 'add':
        noisy_features = features + noise
    else:
        raise Exception('Invalid noise type.')
    return noisy_features


def normalize_eeg(eeg, eeg_mean, eeg_std):
    # All inputs are expected to be float32 torch tensors.
    return (eeg - eeg_mean) / eeg_std


class CWTGraphDataset(Dataset):
    def __init__(self, annot_df, dataset_crop_path, norm_stats_path, augment = False):
        super().__init__()
        self.annot_df = annot_df
        self.dataset_crop_path = dataset_crop_path
        self.augment = augment
        if norm_stats_path is not None:
            norm_stats = np.load(norm_stats_path)
            self.norm_mean = norm_stats['list_mean']
            self.norm_std = norm_stats['list_std']
        else:
            self.norm_mean = None
            self.norm_std = None
            self.scaler = StandardScaler()

    def len(self):
        # torch_geometric.data.Dataset objects are peculiar.
        # Instead of implementing __len__(), a len() function should be defined.
        # The superclass implemets a __len__() that internally calls this code.
        # More details in the docs:
        # https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
        # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html
        return len(self.annot_df)


    def get(self, idx):
        # Here, a get() is expected instead of __getitem__() for analogous reasons to the __len__() case.

        record = self.annot_df.iloc[idx]
        file_path = os.path.join(self.dataset_crop_path, record['crop_file'])
        # Special case for miltiadous_deriv_uV_d1.0s_o0.5s (wrong file extensions in annot file):
        #file_path = os.path.join(self.dataset_crop_path, record['crop_file'][:-4] + '.mat')
        #cwt = scipy.io.loadmat(file_path)['cwts']
        cwt = np.load(file_path)

        # Normalization
        if self.norm_mean is not None:
            norm_cwt = normalize_eeg(cwt.astype(np.float32), self.norm_mean.astype(np.float32), self.norm_std.astype(np.float32))
        else:
            orig_shape = cwt.shape
            norm_cwt = self.scaler.fit_transform(np.reshape(cwt, (-1, 1)))
            norm_cwt = np.reshape(norm_cwt, orig_shape).astype(np.float32)


        # Miltiadous electrodes:
        # ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        #     0      1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18
        edge_index_miltiadous = torch.tensor([[0, 0, 1, 1, 10, 10, 2, 2, 2, 2, 16, 16,
                                              16, 16, 16, 3, 3, 3, 3, 11, 11, 12, 12, 12,
                                              4, 4, 4, 4, 17, 17, 17, 17, 5, 5, 5, 5,
                                              13, 13, 13, 14, 14, 6, 6, 6, 6, 18, 18, 18,
                                              18, 18, 7, 7, 7, 7, 15, 15, 8, 8, 9, 9],
                                              [2, 16, 16, 3, 2, 12, 0, 16, 4, 10, 0, 1,
                                              3, 17, 2, 1, 11, 5, 16, 3, 13, 10, 4, 14,
                                              2, 17, 6, 12, 16, 5, 18, 4, 3, 13, 7, 17,
                                              11, 5, 15, 12, 6, 4, 18, 8, 14, 17, 7, 9,
                                              8, 6, 5, 15, 9, 18, 13, 7, 6, 18, 18, 7]])
        
            # ---------------- augmentation only for FTD ---------------- #
        if self.augment and record['label'] == 1:          # 1 = FTD
            cwt = cwt + np.random.normal(0, 0.01, cwt.shape)

        x = np.moveaxis(norm_cwt, 2, 0)
        x = np.reshape(x, (19, -1))
        x = torch.tensor(x)
        y = torch.tensor([record['label']])

        graph = Data(edge_index=edge_index_miltiadous, x=x, y=y)

        # Only Data objects should be returned. The label is embedded in it.
        return graph

