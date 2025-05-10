import os
import sys
import json
import ipdb
import csv
import torch
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from datasets import *
from models import *


def argparser():
    parser = argparse.ArgumentParser(description='Generate a CSV with detailed inference output of a specific checkpoint.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--checkpoints_base_path', default='local/checkpoints/', help='parent directory of the available checkpoints')
    parser.add_argument('-t', '--timestamp', required=True, help='timestamp (run) of the intended checkpoint')
    parser.add_argument('-f', '--checkpoint_file', default='best_test_acc.pt', help='checkpoint file belonging to the selected run')
    parser.add_argument('-p', '--ds_parent_dir', default='local/datasets/', help='parent directory of the preprocessed dataset')
    parser.add_argument('-n', '--ds_name', required=True, help='name of the preprocessed dataset')
    parser.add_argument('-c', '--classes', required=True, help='classes to use expressed as in annot file names (e.g. \'hc-ad\')')
    parser.add_argument('-d', '--device', default='cuda:0', help='device for computations (cuda:0, cpu, etc.)')
    parser.add_argument('-o', '--output_dir', default='local/detailed_inference/', help='parent directory of the produced outputs')
    args = parser.parse_args()
    return args


class CropPredCounter:
    """
    Predictions counter for 'crop' eval mode.
    """
    def __init__(self):
        self.gt_array = np.empty((0,), dtype=np.uint8)
        self.pred_array = np.empty((0,), dtype=np.uint8)

    def add_pred(self, gt, act):
        self.gt_array = np.append(self.gt_array, gt.astype(np.uint8))
        self.pred_array = np.append(self.pred_array, np.argmax(act).astype(np.uint8))

    def get_arrays(self):
        return self.gt_array, self.pred_array


class ConsensusPredCounter:
    """
    Predictions counter for 'consensus' eval mode.
    """
    def __init__(self):
        self.recs_gt = {}
        self.recs_preds = {}

    def add_pred(self, gt, act, rec_id):
        self.recs_gt[rec_id] = gt.astype(np.uint8)
        self.recs_preds[rec_id] = np.append(self.recs_preds.get(rec_id, np.empty((0,), dtype=np.uint8)), np.argmax(act).astype(np.uint8))

    def get_arrays(self):
        gt_array = np.empty((0,), dtype=np.uint8)
        pred_array = np.empty((0,), dtype=np.uint8)
        for i in self.recs_gt:
            gt_array = np.append(gt_array, self.recs_gt[i])
            pred_array = np.append(pred_array, np.argmax(np.bincount(self.recs_preds[i])).astype(np.uint8))
        return gt_array, pred_array


class AvgPredCounter:
    """
    Predictions counter for 'avg' eval mode.
    """
    def __init__(self):
        self.recs_gt = {}
        self.recs_preds = {}

    def add_pred(self, gt, act, rec_id):
        self.recs_gt[rec_id] = gt.astype(np.uint8)
        self.recs_preds[rec_id] = self.recs_preds.get(rec_id, np.zeros((act.shape[0],))) + softmax(act)

    def get_arrays(self):
        gt_array = np.empty((0,), dtype=np.uint8)
        pred_array = np.empty((0,), dtype=np.uint8)
        for i in self.recs_gt:
            gt_array = np.append(gt_array, self.recs_gt[i])
            # division not necessary, argmax of sum = argmax of avg
            pred_array = np.append(pred_array, np.argmax(self.recs_preds[i]).astype(np.uint8))
        return gt_array, pred_array


def main():
    args = argparser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    annot_file_path = os.path.join(args.ds_parent_dir, args.ds_name, f"annot_all_{args.classes}.csv")
    crop_data_path = os.path.join(args.ds_parent_dir, args.ds_name, "cwt")  # new standard (used by miltiadous_deriv_uV_d1.0s_o0.0s)
    annotations = pd.read_csv(annot_file_path)
    subjects_list = annotations['original_rec'].unique().tolist()
    labels_list = [annotations[annotations['original_rec'] == s].iloc[0]['label'] for s in subjects_list]

    # Hardcoded Miltiadous splitting based on ADformer 'subject-independent' strategy
    # TODO substitute with JSON split info in dataset folder
    if args.classes == "hc-ad":
        train_subjects = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        val_subjects = [54, 55, 56, 57, 58, 59, 22, 23, 24, 25, 26, 27, 28]
        test_subjects = [60, 61, 62, 63, 64, 65, 29, 30, 31, 32, 33, 34, 35, 36]
    elif args.classes == "hc-ftd-ad":
        train_subjects = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        val_subjects = [54, 55, 56, 57, 58, 59, 79, 80, 81, 82, 83, 22, 23, 24, 25, 26, 27, 28]
        test_subjects = [60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 29, 30, 31, 32, 33, 34, 35, 36]
    else:
        raise Exception('Handcrafted splitting not available for the selected classes.')

    train_subjects = ['sub-{:03d}'.format(s) for s in train_subjects]
    val_subjects = ['sub-{:03d}'.format(s) for s in val_subjects]
    test_subjects = ['sub-{:03d}'.format(s) for s in test_subjects]

    train_df = annotations[annotations['original_rec'].isin(train_subjects)]  # crops in train set
    val_df = annotations[annotations['original_rec'].isin(val_subjects)]  # crops in val set
    test_df = annotations[annotations['original_rec'].isin(test_subjects)]  # crops in test set

    train_dataset = CWTGraphDataset(train_df, crop_data_path, None)
    val_dataset = CWTGraphDataset(val_df, crop_data_path, None)
    test_dataset = CWTGraphDataset(test_df, crop_data_path, None)

    num_classes = args.classes.count('-') + 1
    # equivalent to Mk15 (RTSI topology)
    model = GNNCWT2D_Mk11_1sec(19, (40, 500), num_classes)  # if modified remember to edit info.json file
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoints_base_path, args.timestamp, args.checkpoint_file), map_location=device)['model_state_dict'])
    model.eval()

    # Output folder and info file
    output_path = os.path.join(args.output_dir, 'inference_' + session_timestamp)
    os.makedirs(output_path, exist_ok=True)
    inference_info = {
        'checkpoint_timestamp': os.path.join(args.timestamp, args.checkpoint_file),
        'dataset_name': args.ds_name,
        'classes': args.classes,
        'model': 'GNNCWT2D_Mk11_1sec'
    }
    with open(os.path.join(output_path, 'info.json'), 'w') as f:  # inference run json to disk
        json.dump(inference_info, f, indent=4)

    # Inference on train set
    print('Computing inference on train set...')
    train_inference_crop_info = pd.DataFrame(columns=['crop_file', 'subject', 'label'])
    train_inference_activations = {}
    for c in tqdm(range(len(train_df)), ncols=100):
        # csv, crop info
        df_record = train_df.iloc[c]
        train_inference_crop_info.loc[len(train_inference_crop_info)] = [df_record['crop_file'], df_record['original_rec'], df_record['label']]
        # npy, activations
        data = train_dataset[c].to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, torch.zeros(19, dtype=torch.int64).to(device))  # one crop at a time :/
        train_inference_activations[df_record['crop_file']] = out.cpu().numpy()[0]
    np.save(os.path.join(output_path, 'train_activations.npy'), train_inference_activations)  # activations to disk
    train_inference_crop_info.to_csv(os.path.join(output_path, 'train_crop_annot.csv'), index=False)  # crop annot csv to disk
    print('Done.\n')

    # Inference on validation set
    print('Computing inference on validation set...')
    val_inference_crop_info = pd.DataFrame(columns=['crop_file', 'subject', 'label'])
    val_inference_activations = {}
    for c in tqdm(range(len(val_df)), ncols=100):
        # csv, crop info
        df_record = val_df.iloc[c]
        val_inference_crop_info.loc[len(val_inference_crop_info)] = [df_record['crop_file'], df_record['original_rec'], df_record['label']]
        # npy, activations
        data = val_dataset[c].to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, torch.zeros(19, dtype=torch.int64).to(device))  # one crop at a time :/
        val_inference_activations[df_record['crop_file']] = out.cpu().numpy()[0]
    np.save(os.path.join(output_path, 'val_activations.npy'), val_inference_activations)  # activations to disk
    val_inference_crop_info.to_csv(os.path.join(output_path, 'val_crop_annot.csv'), index=False)  # crop annot csv to disk
    print('Done.\n')

    # Inference on test set
    print('Computing inference on test set...')
    test_inference_crop_info = pd.DataFrame(columns=['crop_file', 'subject', 'label'])
    test_inference_activations = {}
    for c in tqdm(range(len(test_df)), ncols=100):
        # csv, crop info
        df_record = test_df.iloc[c]
        test_inference_crop_info.loc[len(test_inference_crop_info)] = [df_record['crop_file'], df_record['original_rec'], df_record['label']]
        # npy, activations
        data = test_dataset[c].to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, torch.zeros(19, dtype=torch.int64).to(device))  # one crop at a time :/
        test_inference_activations[df_record['crop_file']] = out.cpu().numpy()[0]
    np.save(os.path.join(output_path, 'test_activations.npy'), test_inference_activations)  # activations to disk
    test_inference_crop_info.to_csv(os.path.join(output_path, 'test_crop_annot.csv'), index=False)  # crop annot csv to disk
    print('Done.\n')


if __name__ == "__main__":
    main()
