#!/usr/bin/env python3
"""
train_multitask.py
------------------
Fine-tuning multi-task della GNN:

  • head principale → 3 classi  (hc-ftd-ad)
  • head ausiliaria → 2 classi  (Good / Bad)

Richiede i moduli:
  multitask_model.py    (definisce MultiTaskGNNCWT2D_Mk11_1sec)
  multitask_dataset.py  (definisce MultiTaskCWTGraphDataset)

Tutti i sorgenti originali rimangono intatti.
"""
import os, argparse, torch
import pandas as pd, numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils              import seed_everything
from multitask_dataset  import MultiTaskCWTGraphDataset
from multitask_model    import MultiTaskGNNCWT2D_Mk11_1sec

# ============================= ARGPARSE ==============================
def argparser():
    p = argparse.ArgumentParser(
        description='Multi-task fine-tuning (GNN + Good/Bad head)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset & path
    p.add_argument('-n','--ds_name', required=True,
                   help='nome del dataset preprocessato')
    p.add_argument('-c','--classes', required=True,
                   help='classi (es. hc-ftd-ad)')
    p.add_argument('-p','--ds_parent_dir', default='local/datasets/',
                   help='cartella padre del dataset')
    p.add_argument('-m','--monitor_dir', required=True,
                   help='dir con *_predictions_detailed.csv del monitor-CNN')
    p.add_argument('-k','--pretrained_ckpt', required=True,
                   help='checkpoint .pt della vecchia GNN')
    # training
    p.add_argument('-l','--lambda_aux', default=0.3, type=float,
                   help='peso λ per la loss Good/Bad')
    p.add_argument('-d','--device', default='cuda:0')
    p.add_argument('-s','--seed', default=1234, type=int)
    p.add_argument('-b','--batch_size', default=64, type=int)
    p.add_argument('-w','--num_workers', default=4, type=int)
    p.add_argument('-e','--num_epochs', default=30, type=int)
    p.add_argument('-r','--lr', default=1e-4, type=float)
    p.add_argument('-y','--weight_decay', default=1e-8, type=float)
    # regularization
    p.add_argument('--patience',       default=5, type=int,
                   help='early-stopping patience (val acc)')
    p.add_argument('--freeze_epochs',  default=3, type=int,
                   help='epoche con backbone congelato')
    return p.parse_args()

# ====================== METRIC UTILITIES ============================
def compute_metrics(gt, pr):
    acc = accuracy_score(gt, pr)
    print(f'\nAccuracy: {acc:.6f}\n')
    print(classification_report(gt, pr))
    print(confusion_matrix(gt, pr))

# ====================== TRAIN / VAL ONE EPOCH =======================
def train_one_epoch(model, epoch, tb, loader, device, opt,
                    loss_fn, lambda_aux):
    model.train()
    run_loss, correct = 0., 0
    for data in tqdm(loader, ncols=100, desc='  Train'):
        data = data.to(device)
        opt.zero_grad()
        log_main, log_aux = model(data.x, data.edge_index, data.batch)
        loss  = loss_fn(log_main, data.y.squeeze())
        loss += lambda_aux * loss_fn(log_aux, data.good_label.squeeze())
        run_loss += loss.item()
        loss.backward(); opt.step()
        correct += (log_main.argmax(1) == data.y.squeeze()).sum().item()

    epoch_loss = run_loss / len(loader)
    epoch_acc  = correct  / len(loader.dataset)
    tb.add_scalar('Loss/train', epoch_loss, epoch)
    tb.add_scalar('Accuracy/train', epoch_acc, epoch)
    return epoch_loss, epoch_acc


def val_one_epoch(model, epoch, tb, loader, device,
                  loss_fn, lambda_aux, tag='val'):
    model.eval()
    run_loss, correct = 0., 0
    for data in tqdm(loader, ncols=100, desc=f'  {tag.capitalize()}'):
        data = data.to(device)
        with torch.no_grad():
            log_main, log_aux = model(data.x, data.edge_index, data.batch)
            loss  = loss_fn(log_main, data.y.squeeze())
            loss += lambda_aux * loss_fn(log_aux, data.good_label.squeeze())
        run_loss += loss.item()
        correct += (log_main.argmax(1) == data.y.squeeze()).sum().item()

    epoch_loss = run_loss / len(loader)
    epoch_acc  = correct  / len(loader.dataset)
    tb.add_scalar(f'Loss/{tag}', epoch_loss, epoch)
    tb.add_scalar(f'Accuracy/{tag}', epoch_acc, epoch)
    return epoch_loss, epoch_acc

# ================================ MAIN ===============================
def main():
    args = argparser()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---------- logging dirs ----------------------------------------
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    run = f'multitask_{ts}'
    tb  = SummaryWriter(f'local/runs/{run}')
    ckpt_dir = f'local/checkpoints/{run}/'; os.makedirs(ckpt_dir, exist_ok=True)

    # ---------- dataset & split hard-coded (come single_fold) -------
    annot_fp = os.path.join(args.ds_parent_dir, args.ds_name,
                            f'annot_all_{args.classes}.csv')
    crop_dir = os.path.join(args.ds_parent_dir, args.ds_name, 'cwt')
    annot = pd.read_csv(annot_fp)

    def sub(lst): return [f'sub-{s:03d}' for s in lst]
    if args.classes == "hc-ad":
        tr = sub([37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
                  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
        vl = sub([54,55,56,57,58,59,22,23,24,25,26,27,28])
        ts_ = sub([60,61,62,63,64,65,29,30,31,32,33,34,35,36])
    elif args.classes == "hc-ftd-ad":
        tr = sub([37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
                  66,67,68,69,70,71,72,73,74,75,76,77,78,
                  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
        vl = sub([54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28])
        ts_ = sub([60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36])
    else:
        raise Exception('Splitting non disponibile per le classi richieste.')

    df_train = annot[annot.original_rec.isin(tr)]
    df_val   = annot[annot.original_rec.isin(vl)]
    df_test  = annot[annot.original_rec.isin(ts_)]

    # ---- CSV monitor ------------------------------------------------
    csv_train = os.path.join(args.monitor_dir, 'train_predictions_detailed.csv')
    csv_val   = os.path.join(args.monitor_dir, 'val_predictions_detailed.csv')
    csv_test  = os.path.join(args.monitor_dir, 'test_predictions_detailed.csv')

    train_ds = MultiTaskCWTGraphDataset(df_train, crop_dir, csv_train)
    val_ds   = MultiTaskCWTGraphDataset(df_val,   crop_dir, csv_val)
    test_ds  = MultiTaskCWTGraphDataset(df_test,  crop_dir, csv_test)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False)

    # ---------- modello & optimizer ---------------------------------
    model = MultiTaskGNNCWT2D_Mk11_1sec.from_pretrained(args.pretrained_ckpt, device)

    # *Congeliamo il backbone per le prime `freeze_epochs`*
    for p in model.backbone.parameters():
        p.requires_grad = False
    print(f'Backbone congelato per {args.freeze_epochs} epoche.')

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val, best_test = 0., 0.
    epochs_no_improve   = 0

    print(f'\nRun: {run}')
    print(f'Split sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}')
    print(f'λ (aux loss) = {args.lambda_aux}\n')

    # ======================== TRAIN LOOP ============================
    for ep in range(args.num_epochs):

        # --- unfreeze backbone -------------------------------------
        if ep == args.freeze_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(),
                                   lr=args.lr * 0.5,
                                   weight_decay=args.weight_decay)
            print('Backbone sbloccato - LR dimezzato.')

        print(f'\n===== Epoch {ep:03d} =====')
        tr_loss, tr_acc = train_one_epoch(model, ep, tb, train_dl, device,
                                          opt, loss_fn, args.lambda_aux)
        vl_loss, vl_acc = val_one_epoch(model, ep, tb, val_dl, device,
                                        loss_fn, args.lambda_aux, 'val')
        ts_loss, ts_acc = val_one_epoch(model, ep, tb, test_dl, device,
                                        loss_fn, args.lambda_aux, 'test')

        print(f'Acc  (train/val/test): {tr_acc:.4f}/{vl_acc:.4f}/{ts_acc:.4f}')
        print(f'Loss (train/val/test): {tr_loss:.4f}/{vl_loss:.4f}/{ts_loss:.4f}')

        # ---- early-stopping --------------------------------------
        if vl_acc > best_val:
            best_val = vl_acc; epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f'\n>>> Early-Stopping dopo {args.patience} epoche senza miglioramento (val acc).')
                break

        # ---- checkpoint ------------------------------------------
        ckpt = {'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_acc': tr_acc, 'val_acc': vl_acc, 'test_acc': ts_acc}
        torch.save(ckpt, os.path.join(ckpt_dir, 'last.pt'))
        if ts_acc > best_test:
            best_test = ts_acc
            torch.save(ckpt, os.path.join(ckpt_dir, 'best.pt'))
            print('>>> nuovo best test acc salvato.')

    # =================== EVAL FINALE (best) =========================
    print('\n=== Valutazione finale su Test con modello BEST ===')
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best.pt'),
                                     map_location=device)['model_state_dict'])
    model.eval(); y_true, y_pred = [], []
    for data in tqdm(test_dl, ncols=100, desc='  Eval'):
        data = data.to(device)
        with torch.no_grad():
            log_main, _ = model(data.x, data.edge_index, data.batch)
        y_true.extend(data.y.squeeze().cpu().numpy())
        y_pred.extend(log_main.argmax(1).cpu().numpy())
    compute_metrics(np.array(y_true), np.array(y_pred))

# --------------------------------------------------------------------
if __name__ == '__main__':
    main()
