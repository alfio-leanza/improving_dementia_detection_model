import os
import ipdb
import csv
import torch
import argparse
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from datetime import datetime
from scipy.special import softmax
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

from utils import seed_everything, write_tboard_dict
from datasets import *
from model_threehead import *              # ⬅️  NUOVO import
from single_fold_arcface import evaluate_and_save

"""
This is a copy of kfold_crossval.py made to work with a single custom fold (Miltiadous).
Subject idxs are hacked into the code instead of using StratifiedKFold or LeaveOneOut.
"""


def argparser():
    parser = argparse.ArgumentParser(description='Fake K-Fold cross validation (single fold) for preprocessed data. Custom handmade splitting is performed', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--ds_name', required=True, help='name of the preprocessed dataset')
    parser.add_argument('-c', '--classes', required=True, help='classes to use expressed as in annot file names (e.g. \'hc-ad\')')
    parser.add_argument('-p', '--ds_parent_dir', default='local/datasets/', help='parent directory of the preprocessed dataset')
    parser.add_argument('-d', '--device', default='cuda:0', help='device for computations (cuda:0, cpu, etc.)')
    parser.add_argument('-s', '--seed', default=1234, type=int, help='general reproducibility seed')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('-w', '--num_workers', default=4, type=int, help='number of workers for dataloaders')
    parser.add_argument('-e', '--num_epochs', default=100, type=int, help='training epochs')
    parser.add_argument('-r', '--lr', default=0.00001, type=float, help='training learning rate')
    parser.add_argument('-y', '--weight_decay', default=1e-8, type=float, help='training weight decay')
    parser.add_argument('-g', '--scheduler_gamma', default=0.98, type=float, help='exponential decay gamma')
    parser.add_argument('--debug', action='store_true', help='do not produce artifacts (no tensorboard logs, no saved checkpoints, etc)')
    parser.set_defaults(debug=False)
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


def compute_print_metrics(gt_array, pred_array):
    acc = accuracy_score(gt_array, pred_array)
    class_report = classification_report(gt_array, pred_array)
    cm = confusion_matrix(gt_array, pred_array)
    print(f'\nAccuracy: {acc:.6f}\n')
    print(class_report)
    print(cm)

# ------------------------------------------------------------------- #
def train_one_epoch(model, epoch, tb_writer, loader, device, optimizer,
                    ce_hard, kl_soft, alpha, T):
    """
    Training con loss collaborativa:
      L = α * CE_hard + (1-α) * KL_soft
    """
    model.train()
    running_loss = 0.
    correct = 0

    for data in tqdm(loader, ncols=100, desc='  Train'):
        data = data.to(device)
        optimizer.zero_grad()

        # ----- forward: otteniamo logit medio + lista delle K teste -----
        avg_logit, logits_list = model(data.x, data.edge_index, data.batch,
                                       return_all=True)

        # ---------------- Hard CE (su ogni testa) ---------------------- #
        hard_loss = torch.stack([ce_hard(l, data.y) for l in logits_list]).mean()

        # ---------------- Soft KL fra teste --------------------------- #
        with torch.no_grad():
            soft_targets = torch.softmax(torch.stack(logits_list, 0).mean(0) / T, dim=1)
        soft_loss = 0.
        for l in logits_list:
            student_log = torch.log_softmax(l / T, dim=1)
            soft_loss += kl_soft(student_log, soft_targets)
        soft_loss = (soft_loss / len(logits_list)) * (T * T)

        loss = alpha * hard_loss + (1 - alpha) * soft_loss
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        # running accuracy da logit medio
        pred = avg_logit.argmax(dim=1)
        correct += int((pred == data.y).sum())

    # ---------- TensorBoard logging ----------
    avg_loss = running_loss / len(loader)
    tb_writer.add_scalar('Loss/train', avg_loss, epoch)
    epoch_acc = correct / len(loader.dataset)
    tb_writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    return (avg_loss, epoch_acc)


def val_one_epoch(model, epoch, tb_writer, loader, device, loss_fn, testing=False):
    """
    Validation e Test: si usa solo il logit medio (no collaborativa).
    """
    model.eval()
    running_loss = 0.
    correct = 0
    tqdm_desc = '  Test' if testing else '  Val'

    for data in tqdm(loader, ncols=100, desc=tqdm_desc):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)   # logit medio
        loss = loss_fn(out, data.y)
        running_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    avg_loss = running_loss / len(loader)
    epoch_acc = correct / len(loader.dataset)

    # TensorBoard
    if testing:
        tb_writer.add_scalar('Loss/test', avg_loss, epoch)
        tb_writer.add_scalar('Accuracy/test', epoch_acc, epoch)
    else:
        tb_writer.add_scalar('Loss/val', avg_loss, epoch)
        tb_writer.add_scalar('Accuracy/val', epoch_acc, epoch)

    return (avg_loss, epoch_acc)


# ------------------------------------------------------------------- #
def main():
    args = argparser()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/multi_head/runs/train_{}'.format(session_timestamp))
    checkpoint_save_dir = f'/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/multi_head/checkpoints/train_{session_timestamp}/'
    results_save_dir = '/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/multi_head/results'
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    annot_file_path = os.path.join(args.ds_parent_dir, args.ds_name, f"annot_all_{args.classes}.csv")
    #crop_data_path = os.path.join(args.ds_parent_dir, args.ds_name, "data")
    crop_data_path = os.path.join(args.ds_parent_dir, args.ds_name, "cwt")  # new standard (used by miltiadous_deriv_uV_d1.0s_o0.0s)
    annotations = pd.read_csv(annot_file_path)
    subjects_list = annotations['original_rec'].unique().tolist()
    labels_list = [annotations[annotations['original_rec'] == s].iloc[0]['label'] for s in subjects_list]

    #splitter = LeaveOneOut() if args.loo else StratifiedKFold(n_splits=args.k, random_state=args.splitter_seed, shuffle=True)
    #for fold, (train_idxs, val_idxs) in enumerate(splitter.split(np.zeros(len(labels_list)), labels_list)):
    seed_everything(args.seed)

    # Hardcoded Miltiadous splitting based on ADformer 'subject-independent' strategy
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

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # ---------------------- MODELLO MCL ----------------------------- #
    num_classes = args.classes.count('-') + 1
    backbone = GNNCWT2D_Mk11_1sec(feat_dim=64)
    model    = MultiHeadCollaborativeGNN(backbone, feat_dim=64,
                                         num_heads=3, temperature=2.0).to(device)

    # --- losses (niente pesi di classe, niente Focal) ---------------
    ce_hard = torch.nn.CrossEntropyLoss()
    kl_soft = torch.nn.KLDivLoss(reduction='batchmean')
    alpha_start, alpha_end = 0.9, 0.7
    T = 2.0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_val_accuracy = 0
    for current_epoch in range(args.num_epochs):
        print(f'\nStarting epoch {current_epoch:03d}.')
        # annealing lineare di α
        alpha = alpha_start - (alpha_start - alpha_end) * current_epoch / (args.num_epochs - 1)

        train_loss, train_acc = train_one_epoch(
            model, current_epoch, writer, train_dataloader, device,
            optimizer, ce_hard, kl_soft, alpha, T)

        with torch.no_grad():
            val_loss, val_acc = val_one_epoch(model, current_epoch, writer,
                                              val_dataloader, device, ce_hard)
            test_loss, test_acc = val_one_epoch(model, current_epoch, writer,
                                                test_dataloader, device, ce_hard,
                                                testing=True)
        writer.flush()
        print(f'Epoch {current_epoch:03d} done.')
        print(f'  Accuracy (train/val/test): {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}')
        print(f'  Loss (train/val/test): {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}')

        # ---------- checkpoint ----------
        checkpoint = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'epoch': current_epoch,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(checkpoint_save_dir, 'last.pt'))

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(checkpoint, os.path.join(checkpoint_save_dir, 'best_val_acc.pt'))
            print("New best val acc checkpoint saved.")

   

    # Eval val set
    crop_pred_counter = CropPredCounter()
    consensus_pred_counter = ConsensusPredCounter()
    avg_pred_counter = AvgPredCounter()
    print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Evaluating on val set... <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    model.load_state_dict(torch.load(os.path.join(checkpoint_save_dir, f'best_val_acc.pt'), map_location=device)['model_state_dict'])
    model.eval()
    for s in tqdm(range(len(val_df)), ncols=100):
        data = val_dataset[s]
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, torch.zeros(19, dtype=torch.int64).to(device))

        crop_name = val_df.iloc[s]['crop_file']
        crop_gt = val_df.iloc[s]['label']
        crop_act = np.squeeze(out.detach().cpu().numpy())
        orig_rec = annotations[annotations['crop_file']==crop_name].iloc[0]['original_rec']
        crop_pred_counter.add_pred(crop_gt, crop_act)
        consensus_pred_counter.add_pred(crop_gt, crop_act, orig_rec)
        avg_pred_counter.add_pred(crop_gt, crop_act, orig_rec)

    # Final metrics
    # Crop pred counter
    print('\n\n======================= CROP ========================')
    gt_array, pred_array = crop_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array)
    # Consensus pred counter
    print('\n\n===================== CONSENSUS =====================')
    gt_array, pred_array = consensus_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array)
    # Avg pred counter
    print('\n\n======================== AVG ========================')
    gt_array, pred_array = avg_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array)

    # Eval test set
    crop_pred_counter = CropPredCounter()
    consensus_pred_counter = ConsensusPredCounter()
    avg_pred_counter = AvgPredCounter()
    print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Evaluating on test set... <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    model.load_state_dict(torch.load(os.path.join(checkpoint_save_dir, f'best_val_acc.pt'), map_location=device)['model_state_dict'])
    model.eval()
    for s in tqdm(range(len(test_df)), ncols=100):
        data = test_dataset[s]
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, torch.zeros(19, dtype=torch.int64).to(device))

        crop_name = test_df.iloc[s]['crop_file']
        crop_gt = test_df.iloc[s]['label']
        crop_act = np.squeeze(out.detach().cpu().numpy())
        orig_rec = annotations[annotations['crop_file']==crop_name].iloc[0]['original_rec']
        crop_pred_counter.add_pred(crop_gt, crop_act)
        consensus_pred_counter.add_pred(crop_gt, crop_act, orig_rec)
        avg_pred_counter.add_pred(crop_gt, crop_act, orig_rec)

    # Final metrics
    # Crop pred counter
    print('\n\n======================= CROP ========================')
    gt_array, pred_array = crop_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array)
    # Consensus pred counter
    print('\n\n===================== CONSENSUS =====================')
    gt_array, pred_array = consensus_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array)
    # Avg pred counter
    print('\n\n======================== AVG ========================')
    gt_array, pred_array = avg_pred_counter.get_arrays()
    compute_print_metrics(gt_array, pred_array)

    evaluate_and_save(model, train_df, train_dataset, 'train',
                    device, results_save_dir, kl_soft)
    evaluate_and_save(model, val_df,   val_dataset,   'val',
                    device, results_save_dir, kl_soft)
    evaluate_and_save(model, test_df,  test_dataset,  'test',
                    device, results_save_dir, kl_soft)


print('Finish')

if __name__ == "__main__":
    main()
