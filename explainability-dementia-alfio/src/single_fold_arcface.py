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
import matplotlib.pyplot as plt
from utils import seed_everything, write_tboard_dict
from datasets import *
from model_arcface import *
from pytorch_metric_learning.losses import ArcFaceLoss # NEW

"""
This is a copy of kfold_crossval.py made to work with a single custom fold (Miltiadous).
Subject idxs are hacked into the code instead of using StratifiedKFold or LeaveOneOut.
"""

def compute_print_metrics(gt_array, pred_array):
    acc = accuracy_score(gt_array, pred_array)
    class_report = classification_report(gt_array, pred_array)
    cm = confusion_matrix(gt_array, pred_array)
    print(f'\nAccuracy: {acc:.6f}\n')
    print(class_report)
    print(cm)


def evaluate_and_save(model, dataset_df, dataset_obj,
                      set_name, device, results_dir, loss_fn):
    """Valuta il modello e salva report, confusion matrix e CSV inferenze."""
    model.eval()
    inference_rows, gt_array, pred_array = [], [], []

    for s in tqdm(range(len(dataset_df)), ncols=100,
                  desc=f"  {set_name.title()} Eval"):
        data = dataset_obj[s].to(device)
        with torch.no_grad():
            embeds  = model(data.x, data.edge_index, torch.zeros(19,
                             dtype=torch.int64, device=device))
            logits_t = loss_fn.get_logits(embeds)

        logits    = np.squeeze(logits_t.cpu().numpy())
        soft_vals = softmax(logits)
        pred_lbl  = int(np.argmax(logits))

        row = dataset_df.iloc[s]
        true_lbl = int(row.get('label', row.get('true_label', -1)))

        inference_rows.append({
            'crop_file':         row['crop_file'],
            'logits':            logits.tolist(),
            'dataset':           set_name,
            'softmax_values':    soft_vals.tolist(),
            'pred_label':        pred_lbl,
            'true_label':        true_lbl,
            'original_rec':      row.get('original_rec', ''),
            'crop_start_sample': row.get('crop_start_sample',
                                         row.get('start_sample', np.nan)),
            'crop_end_sample':   row.get('crop_end_sample',
                                         row.get('end_sample',  np.nan)),
        })

        gt_array.append(true_lbl)
        pred_array.append(pred_lbl)

    # ---------- salvataggi ---------- #
    os.makedirs(results_dir, exist_ok=True)

    report_txt = classification_report(gt_array, pred_array, digits=4)
    with open(os.path.join(results_dir,
              f"{set_name}_classification_report.txt"), 'w') as f:
        f.write(report_txt)

    cm = confusion_matrix(gt_array, pred_array)
    class_names = ['hc', 'ftd', 'ad'] if cm.shape[0] == 3 else [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap='Blues')
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('N° campioni', rotation=-90, va="bottom")

    # tick & label
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # testo nelle celle
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()*0.6 else 'black')

    ax.set_title(f'Confusion Matrix - {set_name.title()}')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{set_name}_confusion_matrix.png"))
    plt.close(fig)


    pd.DataFrame(inference_rows).to_csv(
        os.path.join(results_dir, f"{set_name}_inferences.csv"),
        index=False
    )

    print(f"\n== {set_name.upper()} ==\n")
    print(report_txt)
    print(cm)



def argparser():
    parser = argparse.ArgumentParser(description='Fake K-Fold cross validation (single fold) for preprocessed data. Custom handmade splitting is performed', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-k', '--k', default=10, type=int, help='number of folds')
    parser.add_argument('-n', '--ds_name', required=True, help='name of the preprocessed dataset')
    parser.add_argument('-c', '--classes', required=True, help='classes to use expressed as in annot file names (e.g. \'hc-ad\')')
    parser.add_argument('-p', '--ds_parent_dir', default='local/datasets/', help='parent directory of the preprocessed dataset')
    parser.add_argument('-d', '--device', default='cuda:0', help='device for computations (cuda:0, cpu, etc.)')
    parser.add_argument('-s', '--seed', default=1234, type=int, help='general reproducibility seed')
    #parser.add_argument('-t', '--splitter_seed', default=69, type=int, help='kfold splitter seed')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('-w', '--num_workers', default=4, type=int, help='number of workers for dataloaders')
    parser.add_argument('-e', '--num_epochs', default=100, type=int, help='training epochs')
    parser.add_argument('-r', '--lr', default=0.00001, type=float, help='training learning rate')
    parser.add_argument('-y', '--weight_decay', default=1e-8, type=float, help='training weight decay')
    parser.add_argument('-g', '--scheduler_gamma', default=0.98, type=float, help='exponential decay gamma')
    #parser.add_argument('-l', '--loo', action='store_true', help='ignore k and apply leave-one-out cross-validation (k = # samples)')
    parser.add_argument('--debug', action='store_true', help='do not produce artifacts (no tensorboard logs, no saved checkpoints, etc)')
    #parser.set_defaults(loo=False)
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


def train_one_epoch(model, epoch, tb_writer, loader, device, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    correct = 0
    all_preds, all_labels = [], []

    for data in tqdm(loader, ncols=100, desc='  Train'):
        data = data.to(device)
        optimizer.zero_grad()

        # ① il modello ora restituisce EMBEDDINGS, non logits
        embeds = model(data.x, data.edge_index, data.batch)

        # ② ArcFaceLoss calcola internamente i logits
        loss = loss_fn(embeds, data.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # ③ otteniamo i logits per la metrica
        logits = loss_fn.get_logits(embeds).detach()
        preds  = torch.argmax(logits, dim=1)
        correct += int((preds == data.y).sum())

    avg_loss = running_loss / len(loader)
    epoch_acc = correct / len(loader.dataset)

    tb_writer.add_scalar('Loss/train', avg_loss, epoch)
    tb_writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    return avg_loss, epoch_acc



def val_one_epoch(model, epoch, tb_writer, loader, device, loss_fn, testing=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    mode = 'test' if testing else 'val'
    desc = '  Test' if testing else '  Val'

    with torch.no_grad():
        for data in tqdm(loader, ncols=100, desc=desc):
            data = data.to(device)

            embeds = model(data.x, data.edge_index, data.batch)
            loss   = loss_fn(embeds, data.y)
            running_loss += loss.item()

            logits = loss_fn.get_logits(embeds)
            preds  = torch.argmax(logits, dim=1)
            correct += int((preds == data.y).sum())

    avg_loss = running_loss / len(loader)
    epoch_acc = correct / len(loader.dataset)

    tb_writer.add_scalar(f'Loss/{mode}', avg_loss, epoch)
    tb_writer.add_scalar(f'Accuracy/{mode}', epoch_acc, epoch)

    return avg_loss, epoch_acc



def main():
    args = argparser()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/arcface/runs/train_{}'.format(session_timestamp))
    checkpoint_save_dir = f'/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/arcface/checkpoints/train_{session_timestamp}/'
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    results_save_dir = checkpoint_save_dir.replace('checkpoints', 'results')
    os.makedirs(results_save_dir, exist_ok=True) 

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

    num_classes = args.classes.count('-') + 1
    embedding_size = 128                     # dimensione embedding per ArcFace
    model = GNNCWT2D_Mk11_1sec_Arc(19, (40, 500), embedding_size)
    model.to(device)
    class_weights = torch.tensor([1.0, 3.0, 1.0], device=device)
    loss_fn = ArcFaceLoss(num_classes=num_classes,
                      embedding_size=embedding_size,
                      margin=0.5,      
                      scale=40,
                      class_weights=class_weights,
                      easy_margin=True)

    #loss_fn = ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size)
    # include i pesi del modello + quelli interni a ArcFaceLoss
    parameters = list(model.parameters()) + list(loss_fn.parameters())
    optimizer   = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    print(f'Session timestamp: {session_timestamp}')
    print(f'Model: {type(model).__name__}')
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    print(f'Args in experiment: {args}')
    print()
    #write_tboard_dict(config_dict, writer)

    # Training loop
    #best_test_accuracy = 0
    best_val_accuracy = 0 # saving the best model based on validation accuracy
    for current_epoch in range(args.num_epochs):
        print(f'\nStarting epoch {current_epoch:03d}.')
        train_loss, train_acc = train_one_epoch(model, current_epoch, writer, train_dataloader, device, optimizer, loss_fn)
        # if current_epoch > 4:
        #     scheduler.step()
        with torch.no_grad():
            val_loss, val_acc = val_one_epoch(model, current_epoch, writer, val_dataloader, device, loss_fn)
            test_loss, test_acc = val_one_epoch(model, current_epoch, writer, test_dataloader, device, loss_fn, testing=True)
        writer.flush()
        print(f'Epoch {current_epoch:03d} done.')
        print(f'  Accuracy (train/val/test): {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}')
        print(f'  Loss (train/val/test): {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}')

        # Save last model
        print("Saving checkpoint... ", end='')
        checkpoint_save_dict = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'epoch': current_epoch,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'model_state_dict': model.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint_save_dict, os.path.join(checkpoint_save_dir, f'last.pt'))
        print('done.')

        # Save best test acc model (morally and technically WRONG!)
        #if test_acc > best_test_accuracy:
            #print("New best test acc, saving checkpoint... ", end='')
            #best_test_accuracy = test_acc
            #torch.save(checkpoint_save_dict, os.path.join(checkpoint_save_dir, f'best_test_acc.pt'))
            #print('done.')

        # Save best val acc model
        if val_acc > best_val_accuracy:
            print("New best val acc, saving checkpoint... ", end='')
            best_val_accuracy = val_acc
            torch.save(checkpoint_save_dict, os.path.join(checkpoint_save_dir, f'best_val_acc.pt'))


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
                    device, results_save_dir, loss_fn)
    evaluate_and_save(model, val_df,   val_dataset,   'val',
                    device, results_save_dir, loss_fn)
    evaluate_and_save(model, test_df,  test_dataset,  'test',
                    device, results_save_dir, loss_fn)


if __name__ == "__main__":
    main()
