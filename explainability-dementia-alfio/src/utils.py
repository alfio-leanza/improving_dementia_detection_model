import ipdb
import os
import random
import torch
import json
import mne
import asrpy
import scipy
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mne_bids import BIDSPath, read_raw_bids


def get_caueeg_annotations(caueeg_path: str, annot_file: str = 'dementia') -> dict:
    """
    Receive the path of the original CAUEEG dataset and the name of an annotation
    file ('dementia', 'dementia-no-overlap', ...) and return the annotation dict.
    """
    annot_path = os.path.join(caueeg_path, annot_file + '.json')
    with open(annot_path) as f:
        annot_dict = json.load(f)
    annot_dict['val_split'] = annot_dict.pop('validation_split')  # rename split for consistency
    return annot_dict


def miltiadous_group_to_label(group: str) -> int:
    """
    Convert groups to numerical labels.
    """
    if group == 'C':
        return 0
    elif group == 'F':
        return 1
    elif group == 'A':
        return 2
    else:
        raise Exception('Invalid group name.')


def generate_miltiadous_annotations_stratified(root_path: str) -> dict:
    """
    Receive the path of the Miltiadous dataset and return the dictionary of
    annotations splitting the dataset in train/val/test.
    Hardcoded stratified splitting is applied.
    """
    annot_path = os.path.join(root_path, 'participants.tsv')
    annot_df = pd.read_csv(annot_path, sep='\t')
    output_dict = {'class_label_to_name': ['HC', 'FTD', 'AD']}

    # hardcoded stratified splitting
    hc_val = [41, 51, 63, 65]
    hc_test = [43, 46, 47, 54]
    hc_train = list(set([*range(37, 66)]) - set(hc_val) - set(hc_test))
    ftd_val = [85, 75, 77]
    ftd_test = [68, 70, 88]
    ftd_train = list(set([*range(66, 89)]) - set(ftd_val) - set(ftd_test))
    ad_val = [19, 9, 10, 15, 25]
    ad_test = [6, 4, 22, 26, 33]
    ad_train = list(set([*range(1, 37)]) - set(ad_val) - set(ad_test))
    rec_split = {'train': hc_train + ftd_train + ad_train,
                 'val': hc_val + ftd_val + ad_val,
                 'test': hc_test + ftd_test + ad_test}

    for split in ['train', 'val', 'test']:
        current_split_list = []
        for s in rec_split[split]:
            subject_id = '{:03d}'.format(s)
            record = annot_df.loc[annot_df['participant_id'] == 'sub-'+subject_id].iloc[0]
            current_split_list.append({'subject_id': subject_id,
                                       'class_label': miltiadous_group_to_label(record['Group']),
                                       'gender': record['Gender'],
                                       'age': record['Age'].item(),
                                       'mmse': record['MMSE'].item()})
        output_dict[f'{split}_split'] = current_split_list

    return output_dict


def rename_caueeg_ch(ch_name: str) -> str:
    """
    Remove '-AVG' at the end of the ch names, e.g. Fp1-AVG -> Fp1.
    No checks, blindly remove the last 4 chars.
    """
    return ch_name[:len(ch_name)-4]


def len_trim_trailing_zeros(a: npt.NDArray) -> int:
    """
    Receive a 2D np matrix and return the length (# of columns) of the trimmed
    version of the matrix, i.e. with no columns full of zeros on the right.
    """
    assert type(a) == np.ndarray
    trim = a.shape[-1]
    for i in range(a.shape[-1]):
        if np.any(a[..., -1 - i] != 0):
            trim = i
            break
    if trim != 0:
        a = a[..., :-trim]
    return a.shape[-1]


def get_caueeg_edf_raw(file_path: str) -> mne.io.Raw:
    """
    Receive the path of a EDF file and return the raw MNE object with fixed
    channels (only EEG with correct names).
    """
    raw = mne.io.read_raw_edf(file_path, verbose=False)
    raw.drop_channels(['EKG', 'Photic'])  # only keep the 19 EEG electrodes
    raw.rename_channels(rename_caueeg_ch, verbose=False)
    return raw


def get_caueeg_edf_crops(
    file_path: str,
    duration_sec: int,
    overlap_sec: int,
    apply_asr: bool = False,
    asr_cutoff: int = 17
) -> tuple[npt.NDArray, list, list]:
    """
    Receive the path of an untrimmed EDF recording, duration and overlap of the
    crops to generate expressed in seconds, whether or not apply ASR with cutoff.
    ASR is done here because it needs the whole recording, before crops.
    Return a tuple with:
      - generated crops: ndarray with dims (ncrops, nchannels, nsamples)
      - crop starts: list (values are inclusive)
      - crop ends: list (values are inclusive)
    Data is in Volts (MNE default behaviour).
    """
    raw = get_caueeg_edf_raw(file_path)
    sfreq = int(raw.info['sfreq'])
    # lenght of the trimmed recording expressed in samples
    file_len_trimmed_sample = len_trim_trailing_zeros(raw[:][0])
    # lenght of the trimmed recording expressed in seconds
    file_len_trimmed_sec = file_len_trimmed_sample / sfreq
    # duration of the crops expressed in samples
    duration_samples = duration_sec * sfreq

    # trim trailing zeros, asr, generate crops with overlap
    raw.crop(tmin=0.0, tmax=file_len_trimmed_sec, include_tmax=False, verbose=False)
    if apply_asr:
        asr = asrpy.ASR(sfreq=raw.info['sfreq'], cutoff=asr_cutoff)
        raw.load_data(verbose=False)
        asr.fit(raw)
        raw = asr.transform(raw)
    epochs = mne.make_fixed_length_epochs(raw, duration=duration_sec, overlap=overlap_sec, verbose=False)

    crop_data = epochs.get_data(verbose=False)  # ndarray (ncrops, nchannels, nsamples)
    crop_starts = [i[0] for i in epochs.events]  # inclusive
    crop_ends = [i+duration_samples-1 for i in crop_starts]  # inclusive
    return (crop_data, crop_starts, crop_ends)


def get_miltiadous_bids_crops(
    dataset_root_path: str,
    subject: str,
    duration_sec: int,
    overlap_sec: int
) -> tuple[npt.NDArray, list, list]:
    """
    Receive the root path of the BIDS-formatted dataset, the subject id,
    duration and overlap of the crops to generate expressed in seconds.
    Return a tuple with:
      - generated crops: ndarray with dims (ncrops, nchannels, nsamples)
      - crop starts: list (values are inclusive)
      - crop ends: list (values are inclusive)
    Data is in Volts (MNE default behaviour).
    """
    rec_bids_path = BIDSPath(subject=subject, root=dataset_root_path, datatype='eeg', task='eyesclosed')
    raw = read_raw_bids(rec_bids_path, verbose=40)
    sfreq = int(raw.info['sfreq'])
    # duration of the crops expressed in samples
    duration_samples = duration_sec * sfreq

    # generate crops with overlap
    epochs = mne.make_fixed_length_epochs(raw, duration=duration_sec, overlap=overlap_sec, verbose=False)

    crop_data = epochs.get_data(verbose=False)  # ndarray (ncrops, nchannels, nsamples)
    crop_starts = [i[0] for i in epochs.events]  # inclusive
    crop_ends = [i+duration_samples-1 for i in crop_starts]  # inclusive
    return (crop_data, crop_starts, crop_ends)


def plot_cwt_figure_19ch(
    file_name: str,
    base_path: str = '~/fast2/gnn-datasets/miltiadous_cwt_d10s_o0s/data/'
) -> None:
    """
    Plot in a single figure the 19 channels of a CWT, given the path and
    filename of the .mat file. Files should have the 19 channels
    in the last dimension.
    """
    complete_path = os.path.expanduser(os.path.join(base_path, file_name))
    matr = scipy.io.loadmat(complete_path)['cwts']
    plt.figure(figsize=(16, 8))
    for i in range(19):
        plt.subplot(4, 5, i+1)
        plt.imshow(matr[:, :, i], cmap='plasma', aspect='auto')
    plt.subplots_adjust(0.04, 0.04, 0.96, 0.94, 0.2, 0.2)
    plt.suptitle(file_name)
    plt.show(block=False)


def full_conn_edge_index(N: int) -> torch.LongTensor:
    """
    Generate the edge_index tensor for a complete graph with N (>=1)
    nodes (full connectivity without self-loops) as expected by PyG.
    """
    values = torch.arange(N)
    first_row = values.repeat_interleave(N-1)
    values_square = values.repeat(N, 1)
    mask = ~torch.eye(N, dtype=bool)
    second_row = torch.masked_select(values_square, mask)
    return torch.vstack((first_row, second_row))


def batch_full_conn_edge_index(N: int, batch_size: int) -> torch.LongTensor:
    """
    Generate the edge indices for a batch of identical fully connected graphs.
    Each graph of the batch is a fully connected component.
      - N: number of nodes for each graph.
      - batch_size: number of graphs in the batch.
    See `full_conn_edge_index` for reference.
    """
    single_edge_index = full_conn_edge_index(N)
    batch_edge_index = torch.hstack([single_edge_index+i*N for i in range(batch_size)])
    return batch_edge_index


def write_tboard_dict(data_dict, writer) -> None:
    """
    Receive a Python dictionary and save it to tensorboard as a markdown table.
    """
    header = '| ' + ' | '.join(data_dict.keys()) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(data_dict)) + ' |'
    content = '| ' + ' | '.join(str(value) for value in data_dict.values()) + ' |'

    markdown_table = '\n'.join([header, separator, content])
    writer.add_text('Exp_config', markdown_table)


def seed_everything(seed: int) -> None:
    """
    Set all seeds for reproducibility.
    TODO: check PyTorch docs.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
