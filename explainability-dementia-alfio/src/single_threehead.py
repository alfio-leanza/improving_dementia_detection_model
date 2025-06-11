import os, ipdb, csv, torch, argparse
import pandas as pd, numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.special import softmax
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix)
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils import seed_everything
from datasets import *
from model_threehead import GNNCWT2D_Mk11_1sec_3H                
import numpy.typing as npt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from utils import seed_everything, write_tboard_dict
from single_fold_arcface import evaluate_and_save

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

# ─────────────────────────── LOSS 3-HEAD ───────────────────────────
def compute_loss_threehead(log_main, log_bin, log_dem, y,
                           l_bin=0.3, l_dem=0.3):
    """
    Combina le tre teste:
      • head_main : loss sempre
      • head_bin  : loss sempre
      • head_dem  : loss solo sui campioni demenza
    """
    y_bin = (y != 0).long()          # 0=hc 1=demenza
    y_dem = (y - 1).clamp(min=0)     # 0=ftd 1=ad (hc dummy)

    loss = F.cross_entropy(log_main, y)
    loss += l_bin * F.cross_entropy(log_bin, y_bin)

    dem_mask = y_bin == 1
    if dem_mask.any():
        loss += l_dem * F.cross_entropy(log_dem[dem_mask], y_dem[dem_mask])
    return loss

def train_one_epoch(model, epoch, tb_writer, loader, device, optimizer):
    model.train()
    running_loss, correct = 0., 0

    for data in tqdm(loader, ncols=100, desc='  Train'):
        data = data.to(device)
        optimizer.zero_grad()

        logA, logB, logC = model(data.x, data.edge_index, data.batch)   # >>> 3-HEAD
        loss = compute_loss_threehead(logA, logB, logC, data.y,
                                      model.lambda_bin, model.lambda_dem)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += int((logA.argmax(1) == data.y).sum())               # >>> 3-HEAD

    avg_loss = running_loss / len(loader)
    acc = correct / len(loader.dataset)

    tb_writer.add_scalar('Loss/train', avg_loss, epoch)
    tb_writer.add_scalar('Accuracy/train', acc,  epoch)
    return avg_loss, acc


def val_one_epoch(model, epoch, tb_writer, loader, device, split='val'):
    model.eval()
    running_loss, correct = 0., 0

    with torch.no_grad():
        for data in tqdm(loader, ncols=100, desc=f'  {split.capitalize()}'):
            data = data.to(device)
            logA, logB, logC = model(data.x, data.edge_index, data.batch)  # >>> 3-HEAD
            loss = compute_loss_threehead(logA, logB, logC, data.y,
                                          model.lambda_bin, model.lambda_dem)
            running_loss += loss.item()
            correct += int((logA.argmax(1) == data.y).sum())               # >>> 3-HEAD

    avg_loss = running_loss / len(loader)
    acc = correct / len(loader.dataset)

    tb_writer.add_scalar(f'Loss/{split}', avg_loss, epoch)
    tb_writer.add_scalar(f'Accuracy/{split}', acc, epoch)
    return avg_loss, acc


def main():
    args = argparser()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/three_head/runs/train_{}'.format(session_timestamp))
    checkpoint_save_dir = f'/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/three_head/checkpoints/train_{session_timestamp}/'
    results_save_dir = '/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/three_head/results'
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
    val_dataset   = CWTGraphDataset(val_df,   crop_data_path, None)
    test_dataset  = CWTGraphDataset(test_df,  crop_data_path, None)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.num_workers,
                                  pin_memory=False)
    val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
    test_dataloader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
    
    num_classes = args.classes.count('-') + 1
    model = GNNCWT2D_Mk11_1sec_3H(19, (40, 500), num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
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
        train_loss, train_acc = train_one_epoch(model, current_epoch, writer, train_dataloader, device, optimizer)
        # if current_epoch > 4:
        #     scheduler.step()
        with torch.no_grad():
            val_loss, val_acc = val_one_epoch(model, current_epoch, writer, val_dataloader, device, split='val')
            test_loss, test_acc = val_one_epoch(model, current_epoch, writer, test_dataloader, device, split ='test')
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

    evaluate_and_save(model, train_df, train_dataset, 'train',
                    device, results_save_dir, loss_fn)
    evaluate_and_save(model, val_df,   val_dataset,   'val',
                    device, results_save_dir, loss_fn)
    evaluate_and_save(model, test_df,  test_dataset,  'test',
                    device, results_save_dir, loss_fn)

print('Finish')

if __name__ == "__main__":
    main()