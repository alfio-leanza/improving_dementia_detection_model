import os
import sys
import json
import ipdb
import mne
import csv
import torch
import argparse
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from utils import get_miltiadous_bids_crops, generate_miltiadous_annotations_stratified


def argparser():
    parser = argparse.ArgumentParser(description='Preprocessing for Miltiadous dataset. Generates crops and annotations.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--name', required=True, help='name (prefix) of the preprocessd dataset')
    parser.add_argument('-d', '--crop_duration', default=10, type=int, help='crop duration in seconds')
    parser.add_argument('-o', '--crop_overlap', default=5, type=int, help='crop overlap in seconds')
    parser.add_argument('-p', '--ds_root_path', default='/home/tom/Dataset/miltiadous-eeg-ad-ftd/', help='path to the root of the original dataset')
    parser.add_argument('-f', '--output_folder', default='local/datasets/', help='destination path of the preprocessed crops')
    parser.add_argument('--derivatives', action=argparse.BooleanOptionalAction, help='whether use derivatives or not')
    parser.set_defaults(derivatives=True)
    args = parser.parse_args()
    return args


def main():
    args = argparser()
    bids_path = os.path.join(args.ds_root_path, 'derivatives') if args.derivatives else args.ds_root_path
    splits_annotations = generate_miltiadous_annotations_stratified(args.ds_root_path)

    # setup output dataset name & folder, avoid overwriting
    dataset_full_name = f'{args.name}_d{args.crop_duration}s_o{args.crop_overlap}s'
    output_path = os.path.join(args.output_folder, dataset_full_name)
    if os.path.exists(output_path):
        print("Warning: dataset name already in use (a folder with the desired name already exists). Exiting...")
        sys.exit()
    output_path_crops = os.path.join(output_path, 'data')
    os.makedirs(output_path_crops)

    crop_counter = 0

    for split in ['train', 'val', 'test']:
        print(f'\nStarting {split} split.')
        with open(os.path.join(output_path, f'annot_{split}_hc-ftd-ad.csv'), 'w', newline='') as f:
            # The 'complete' annot file including all classes (hc, ftd, ad) is written first; all the other
            # 'reduced' annot files (e.g. hc, ad) are computed at the end by processing the 'complete' one.
            # Always keep this class ordering, even in reduced annot files.
            # Numerical class labels should always be consecutive, e.g. (hc=0, ftd=1, ad=2) -> (hc=0, ad=1).
            writer = csv.writer(f)
            writer.writerow(["crop_file", "label", "original_rec", "crop_start_sample", "crop_end_sample"])

        # number of recording in the split
        n_rec = len(splits_annotations[f'{split}_split'])

        for i in tqdm(range(n_rec), ncols=100):  # for each file in the split
            rec_info = splits_annotations[f'{split}_split'][i]
            subject_id = rec_info['subject_id']
            crop_data, crop_starts, crop_ends = get_miltiadous_bids_crops(bids_path, subject_id, args.crop_duration, args.crop_overlap)
            for j in range(len(crop_data)):  # for each crop in the file
                processed_crop = crop_data[j] * 1e6
                processed_crop = processed_crop.astype(np.float32)

                # save crop
                crop_file_name = '{:08d}'.format(crop_counter) + '.npy'
                np.save(os.path.join(output_path_crops, crop_file_name), processed_crop)
                with open(os.path.join(output_path, f'annot_{split}_hc-ftd-ad.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([crop_file_name, rec_info['class_label'], 'sub-'+subject_id, crop_starts[j], crop_ends[j]])

                crop_counter += 1

        # Reduced annotation files for split
        annot_all_classes = pd.read_csv(os.path.join(output_path, f'annot_{split}_hc-ftd-ad.csv'))
        # HC + FTD
        annot_hc_ftd = annot_all_classes[annot_all_classes['label'] != 2]
        annot_hc_ftd.to_csv(os.path.join(output_path, f'annot_{split}_hc-ftd.csv'), index=False)
        # HC + AD
        annot_hc_ad = annot_all_classes[annot_all_classes['label'] != 1]
        annot_hc_ad.loc[annot_hc_ad['label'] == 2, 'label'] = 1
        annot_hc_ad.to_csv(os.path.join(output_path, f'annot_{split}_hc-ad.csv'), index=False)
        # FTD + AD
        annot_ftd_ad = annot_all_classes[annot_all_classes['label'] != 0]
        annot_ftd_ad.loc[annot_ftd_ad['label'] == 1, 'label'] = 0
        annot_ftd_ad.loc[annot_ftd_ad['label'] == 2, 'label'] = 1
        annot_ftd_ad.to_csv(os.path.join(output_path, f'annot_{split}_ftd-ad.csv'), index=False)

    print('\nDone.')


if __name__ == "__main__":
    main()
