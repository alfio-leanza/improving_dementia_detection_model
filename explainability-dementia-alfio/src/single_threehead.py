import os, torch, argparse, csv
import pandas as pd, numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.special import softmax
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix)
import matplotlib.pyplot as plt                                       # NEW
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import seed_everything
from datasets import CWTGraphDataset
from model_threehead import GNNCWT2D_Mk11_1sec_3H

# ───────────────────────── parser ─────────────────────────
def argparser():
    p = argparse.ArgumentParser(..., formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-n', '--ds_name',  required=True)
    p.add_argument('-c', '--classes',  required=True)
    p.add_argument('-p', '--ds_parent_dir', default='local/datasets/')
    p.add_argument('-d', '--device',  default='cuda:0')
    p.add_argument('-s', '--seed',    default=1234, type=int)
    p.add_argument('-b', '--batch_size', default=64,  type=int)
    p.add_argument('-w', '--num_workers', default=4, type=int)
    p.add_argument('-e', '--num_epochs',  default=100, type=int)
    p.add_argument('-r', '--lr', default=1e-5,  type=float)
    p.add_argument('-y', '--weight_decay', default=1e-8, type=float)
    p.set_defaults(debug=False); return p.parse_args()

# ───────────────────────── loss 3-head ─────────────────────
def compute_loss_threehead(logA, logB, logC, y, l_bin=0.3, l_dem=0.3):
    y_bin = (y != 0).long(); y_dem = (y - 1).clamp(min=0)
    loss  = F.cross_entropy(logA, y)
    loss += l_bin * F.cross_entropy(logB, y_bin)
    mask = y_bin == 1
    if mask.any():
        loss += l_dem * F.cross_entropy(logC[mask], y_dem[mask])
    return loss

# ───────────────────────── train / val ─────────────────────
def train_one_epoch(model, epoch, tb, loader, dev, optim):
    model.train(); run, cor = 0., 0
    for data in tqdm(loader, ncols=100, desc='  Train'):
        data = data.to(dev); optim.zero_grad()
        logA, logB, logC = model(data.x, data.edge_index, data.batch)
        loss = compute_loss_threehead(logA, logB, logC, data.y,
                                      model.lambda_bin, model.lambda_dem)
        loss.backward(); optim.step()
        run += loss.item(); cor += int((logA.argmax(1)==data.y).sum())
    tb.add_scalar('Loss/train', run/len(loader), epoch)
    tb.add_scalar('Accuracy/train', cor/len(loader.dataset), epoch)
    return run/len(loader), cor/len(loader.dataset)

@torch.no_grad()
def val_one_epoch(model, epoch, tb, loader, dev, tag='val'):
    model.eval(); run, cor = 0., 0
    for data in tqdm(loader, ncols=100, desc=f'  {tag.capitalize()}'):
        data = data.to(dev)
        logA, logB, logC = model(data.x, data.edge_index, data.batch)
        run += compute_loss_threehead(logA, logB, logC, data.y,
                                      model.lambda_bin, model.lambda_dem).item()
        cor += int((logA.argmax(1) == data.y).sum())
    tb.add_scalar(f'Loss/{tag}', run/len(loader), epoch)
    tb.add_scalar(f'Accuracy/{tag}', cor/len(loader.dataset), epoch)
    return run/len(loader), cor/len(loader.dataset)

# ───────────────────────── evaluate & save ─────────────────
def evaluate_and_save(model, df, ds_obj, tag, device, out_dir, loss_fn=None):
    model.eval(); rows, gt, pred = [], [], []
    for i in tqdm(range(len(df)), desc=f"  {tag} Eval", ncols=100):
        data = ds_obj[i].to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, torch.zeros(19,dtype=torch.int64,device=device))
            logits = out[0] if isinstance(out, tuple) else out
        logits_np = logits.squeeze().cpu().numpy()
        soft = softmax(logits_np)
        p = int(soft.argmax())
        t = int(df.iloc[i]['label'])
        rows.append({'crop_file': df.iloc[i]['crop_file'],
                     'logits': logits_np.tolist(),
                     'softmax_values': soft.tolist(),
                     'pred_label': p,
                     'true_label': t})
        gt.append(t); pred.append(p)
    os.makedirs(out_dir, exist_ok=True)
    rep = classification_report(gt, pred, digits=4)
    with open(os.path.join(out_dir,f"{tag}_classification_report.txt"),'w') as f:
        f.write(rep)
    cm = confusion_matrix(gt, pred)
    plt.figure(figsize=(6,6)); plt.imshow(cm,cmap='Blues'); plt.title(f"{tag} CM")
    plt.savefig(os.path.join(out_dir,f"{tag}_confusion_matrix.png")); plt.close()
    pd.DataFrame(rows).to_csv(os.path.join(out_dir,f"{tag}_inferences.csv"),index=False)
    print(f"\n== {tag.upper()} ==\n{rep}\n{cm}")

# ───────────────────────── main ────────────────────────────
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

    train_ds = CWTGraphDataset(train_df, crop_data_path, None)
    val_ds   = CWTGraphDataset(val_df,   crop_data_path, None)
    test_ds  = CWTGraphDataset(test_df,  crop_data_path, None)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=False)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    model = GNNCWT2D_Mk11_1sec_3H(19, (40,500)).to(device)
    optim = torch.optim.Adam(model.parameters(),
                             lr=args.lr, weight_decay=args.weight_decay)

    best = 0.0
    for ep in range(args.num_epochs):
        tl, ta = train_one_epoch(model, ep, writer, train_ld, device, optim)
        vl, va = val_one_epoch(model, ep, writer, val_ld, device, 'val')
        print(f"Ep{ep}  train {ta:.3f}  val {va:.3f}")
        if va > best + 1e-3:
            best = va; torch.save(model.state_dict(), os.path.join(checkpoint_save_dir,'best.pt'))

    model.load_state_dict(torch.load(os.path.join(checkpoint_save_dir,'best.pt'), map_location=device))
    val_one_epoch(model, ep, writer, val_ld, device, 'val')
    val_one_epoch(model, ep, writer, test_ld, device, 'test')

    evaluate_and_save(model, train_df, train_ds, 'train', device, results_save_dir)
    evaluate_and_save(model, val_df, val_ds, 'val', device, results_save_dir)
    evaluate_and_save(model, test_df, test_ds, 'test', device, results_save_dir)

if __name__ == "__main__":
    main()
