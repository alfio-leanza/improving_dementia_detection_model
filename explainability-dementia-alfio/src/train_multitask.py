#!/usr/bin/env python3
"""
train_multitask.py
------------------
Fine-tuning multi-task della GNN:
  • head principale   → 3 classi (hc-ftd-ad)
  • head ausiliaria   → 2 classi (Good / Bad)

Richiede i moduli:
  multitask_model.py   (definisce MultiTaskGNNCWT2D_Mk11_1sec)
  multitask_dataset.py (definisce MultiTaskCWTGraphDataset)

TUTTI i file originali rimangono intatti.
"""
import os, argparse, ipdb, torch, csv
import pandas as pd, numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy.special import softmax
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------
from utils import seed_everything, write_tboard_dict
from multitask_dataset import MultiTaskCWTGraphDataset
from multitask_model   import MultiTaskGNNCWT2D_Mk11_1sec

# ============================= ARGPARSE ==============================
def argparser():
    p = argparse.ArgumentParser(
        description='Multi-task fine-tuning (GNN + Good/Bad head)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-n','--ds_name', required=True,
                   help='nome del dataset preprocessato')
    p.add_argument('-c','--classes', required=True,
                   help='classi in formato annot (es. hc-ftd-ad)')
    p.add_argument('-p','--ds_parent_dir', default='local/datasets/',
                   help='cartella padre del dataset')
    p.add_argument('-m','--monitor_dir', required=True,
                   help='folder con *_predictions_detailed.csv del monitor-CNN')
    p.add_argument('-k','--pretrained_ckpt', required=True,
                   help='checkpoint .pt della vecchia GNN da ri-usare')
    p.add_argument('-l','--lambda_aux', default=0.3, type=float,
                   help='peso λ per la loss ausiliaria Good/Bad')
    p.add_argument('-d','--device', default='cuda:0')
    p.add_argument('-s','--seed', default=1234, type=int)
    p.add_argument('-b','--batch_size', default=64, type=int)
    p.add_argument('-w','--num_workers', default=4, type=int)
    p.add_argument('-e','--num_epochs', default=30, type=int)
    p.add_argument('-r','--lr', default=1e-4, type=float)
    p.add_argument('-y','--weight_decay', default=1e-8, type=float)
    p.add_argument('--debug', action='store_true')
    p.set_defaults(debug=False)
    return p.parse_args()

# ======================= METRIC UTILITIES ============================
def compute_print_metrics(gt_array, pred_array):
    acc = accuracy_score(gt_array, pred_array)
    print(f'\nAccuracy: {acc:.6f}\n')
    print(classification_report(gt_array, pred_array))
    print(confusion_matrix(gt_array, pred_array))

# ====================== TRAIN / VAL ONE EPOCH =======================
def train_one_epoch(model, epoch, tb, loader, device, opt,
                    loss_fn, lambda_aux):
    model.train()
    running, correct = 0., 0
    for data in tqdm(loader, ncols=100, desc='  Train'):
        data = data.to(device)
        opt.zero_grad()
        logits_main, logits_aux = model(data.x, data.edge_index, data.batch)
        loss  = loss_fn(logits_main, data.y.squeeze())
        loss += lambda_aux * loss_fn(logits_aux, data.good_label.squeeze())
        running += loss.item()
        loss.backward(); opt.step()
        correct += (logits_main.argmax(1) == data.y.squeeze()).sum().item()

    avg_loss = running / len(loader)
    epoch_acc = correct / len(loader.dataset)
    tb.add_scalar('Loss/train', avg_loss, epoch)
    tb.add_scalar('Accuracy/train', epoch_acc, epoch)
    return avg_loss, epoch_acc

def val_one_epoch(model, epoch, tb, loader, device, loss_fn,
                  lambda_aux, tag):
    model.eval()
    running, correct = 0., 0
    desc = f'  {tag.capitalize()}'
    for data in tqdm(loader, ncols=100, desc=desc):
        data = data.to(device)
        logits_main, logits_aux = model(data.x, data.edge_index, data.batch)
        loss  = loss_fn(logits_main, data.y.squeeze())
        loss += lambda_aux * loss_fn(logits_aux, data.good_label.squeeze())
        running += loss.item()
        correct += (logits_main.argmax(1) == data.y.squeeze()).sum().item()

    avg_loss = running / len(loader)
    epoch_acc = correct / len(loader.dataset)
    tb.add_scalar(f'Loss/{tag}', avg_loss,  epoch)
    tb.add_scalar(f'Accuracy/{tag}', epoch_acc, epoch)
    return avg_loss, epoch_acc

# ================================ MAIN ===============================
def main():
    args = argparser()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ----------- cartelle output ------------------------------------
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'multitask_{ts}'
    tb = SummaryWriter(f'local/runs/{run_name}')
    ckpt_dir = f'local/checkpoints/{run_name}/'; os.makedirs(ckpt_dir, exist_ok=True)

    # ----------- dataset & split ------------------------------------
    annot_file = os.path.join(args.ds_parent_dir, args.ds_name,
                              f'annot_all_{args.classes}.csv')
    crop_path  = os.path.join(args.ds_parent_dir, args.ds_name, 'cwt')
    annot = pd.read_csv(annot_file)

    # soggetti hard-coded identici a single_fold.py
    if args.classes == "hc-ad":
        train_sub = [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        val_sub   = [54,55,56,57,58,59,22,23,24,25,26,27,28]
        test_sub  = [60,61,62,63,64,65,29,30,31,32,33,34,35,36]
    elif args.classes == "hc-ftd-ad":
        train_sub = [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        val_sub   = [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28]
        test_sub  = [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
    else:
        raise Exception('Splitting non disponibile per le classi richieste.')
    train_sub = [f'sub-{s:03d}' for s in train_sub]
    val_sub   = [f'sub-{s:03d}' for s in val_sub]
    test_sub  = [f'sub-{s:03d}' for s in test_sub]

    df_train = annot[annot.original_rec.isin(train_sub)]
    df_val   = annot[annot.original_rec.isin(val_sub)]
    df_test  = annot[annot.original_rec.isin(test_sub)]

    # ----------- csv monitor paths ----------------------------------
    csv_train = os.path.join(args.monitor_dir, 'train_predictions_detailed.csv')
    csv_val   = os.path.join(args.monitor_dir, 'val_predictions_detailed.csv')
    csv_test  = os.path.join(args.monitor_dir, 'test_predictions_detailed.csv')

    train_ds = MultiTaskCWTGraphDataset(df_train, crop_path, csv_train)
    val_ds   = MultiTaskCWTGraphDataset(df_val,   crop_path, csv_val)
    test_ds  = MultiTaskCWTGraphDataset(df_test,  crop_path, csv_test)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True,  num_workers=args.num_workers,
                        pin_memory=False)

    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=False)

    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=False)


    # ----------- modello & ottimizzatore ----------------------------
    model = MultiTaskGNNCWT2D_Mk11_1sec.from_pretrained(args.pretrained_ckpt, device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ----------- LOG di configurazione ------------------------------
    print(f'\nRun: {run_name}')
    print(f'Split sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}')
    print(f'λ (aux loss) = {args.lambda_aux}\n')

    # ======================== TRAIN LOOP ============================
    best_test = 0
    for epoch in range(args.num_epochs):
        print(f'\n===== Epoch {epoch:03d} =====')
        tr_loss, tr_acc = train_one_epoch(model, epoch, tb, train_dl,
                                          device, opt, loss_fn, args.lambda_aux)
        with torch.no_grad():
            vl_loss, vl_acc = val_one_epoch(model, epoch, tb, val_dl,
                                            device, loss_fn, args.lambda_aux, 'val')
            ts_loss, ts_acc = val_one_epoch(model, epoch, tb, test_dl,
                                            device, loss_fn, args.lambda_aux, 'test')
        print(f'Acc  (train/val/test): {tr_acc:.4f}/{vl_acc:.4f}/{ts_acc:.4f}')
        print(f'Loss (train/val/test): {tr_loss:.4f}/{vl_loss:.4f}/{ts_loss:.4f}')

        # checkpoint “last”
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'train_acc': tr_acc, 'val_acc': vl_acc, 'test_acc': ts_acc
        }
        torch.save(ckpt, os.path.join(ckpt_dir, 'last.pt'))

        # checkpoint “best” su test
        if ts_acc > best_test:
            best_test = ts_acc
            torch.save(ckpt, os.path.join(ckpt_dir, 'best.pt'))
            print('>>> nuovo best test acc salvato.')

    # =================== EVAL FINALE (best) =========================
    print('\n=== Valutazione finale con modello BEST ===')
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'best.pt'),
                                     map_location=device)['model_state_dict'])
    model.eval(); gt, pred = [], []
    with torch.no_grad():
        for data in tqdm(test_dl, ncols=100, desc='  Eval'):
            data = data.to(device)
            logits_main, _ = model(data.x, data.edge_index, data.batch)
            gt.extend(data.y.squeeze().cpu().numpy())
            pred.extend(logits_main.argmax(1).cpu().numpy())
    compute_print_metrics(np.array(gt), np.array(pred))

# --------------------------------------------------------------------
if __name__ == '__main__':
    main()
