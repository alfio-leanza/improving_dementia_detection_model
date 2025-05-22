#!/usr/bin/env python3
"""
Fine-tuning GNN con:
  • Re-weighting del loss in base a 'goodness'
  • Cross-entropy con label-smoothing (ε)

È sufficiente PyTorch ≥1.10, che supporta label_smoothing in F.cross_entropy.
"""
import os, argparse, torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils             import seed_everything
from reweight_dataset  import ReweightCWTGraphDataset
from models            import GNNCWT2D_Mk11_1sec

# ----------------------- ARGUMENTI -----------------------------------
def argparser():
    p = argparse.ArgumentParser('Re-weight + label-smooth fine-tuning')
    p.add_argument('-n','--ds_name', required=True)
    p.add_argument('-p','--ds_parent_dir', default='local/datasets/')
    p.add_argument('-m','--monitor_dir', required=True)
    p.add_argument('-k','--pretrained_ckpt', required=True)

    p.add_argument('--alpha', default=0.3, type=float,
                   help='w_i = α+(1-α)·goodness')
    p.add_argument('--invert', action='store_true',
                   help='usa w_i = 1-goodness')
    p.add_argument('--ls_eps', default=0.05, type=float,
                   help='label-smoothing ε')

    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num_epochs', default=15, type=int)
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--lr', default=3e-5,  type=float)
    p.add_argument('--weight_decay', default=1e-8, type=float)
    p.add_argument('-s','--seed', default=1234, type=int)
    return p.parse_args()

# ----------------------- HELPERS -------------------------------------
def run_epoch(model, loader, device, opt, alpha, invert, eps, train=True):
    if train:
        model.train()
    else:
        model.eval()
    run_loss = 0.; correct = 0
    for data in tqdm(loader, ncols=100, desc='  Train' if train else '  Val'):
        data = data.to(device)
        if train:
            opt.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)

        per_sample = F.cross_entropy(
            logits, data.y.squeeze(),
            reduction='none',
            label_smoothing=eps
        )
        w = 1.0 - data.goodness.squeeze() if invert \
            else alpha + (1-alpha)*data.goodness.squeeze()
        loss = (per_sample * w).mean()

        if train:
            loss.backward(); opt.step()
        run_loss += loss.item()
        correct  += (logits.argmax(1) == data.y.squeeze()).sum().item()

    return run_loss / len(loader), correct / len(loader.dataset)

# ----------------------------- MAIN ----------------------------------
def main():
    args = argparser()
    seed_everything(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---------- dataset & split (come prima) -------------
    annot_fp = os.path.join(args.ds_parent_dir, args.ds_name,
                            'annot_all_hc-ftd-ad.csv')
    crop_dir = os.path.join(args.ds_parent_dir, args.ds_name, 'cwt')
    annot = pd.read_csv(annot_fp)

    def sub(idx): return [f'sub-{s:03d}' for s in idx]
    tr = sub([37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
              66,67,68,69,70,71,72,73,74,75,76,77,78,
              1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])
    vl = sub([54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28])
    ts = sub([60,61,62,63,64,65,84,85,86,87,88,
              29,30,31,32,33,34,35,36])

    df_train = annot[annot.original_rec.isin(tr)]
    df_val   = annot[annot.original_rec.isin(vl)]
    df_test  = annot[annot.original_rec.isin(ts)]

    csv = lambda split: os.path.join(args.monitor_dir,f'{split}_predictions_detailed.csv')
    train_ds = ReweightCWTGraphDataset(df_train,crop_dir,csv('train'))
    val_ds   = ReweightCWTGraphDataset(df_val,  crop_dir,csv('val'))
    test_ds  = ReweightCWTGraphDataset(df_test, crop_dir,csv('test'))

    train_dl = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                          num_workers=4,pin_memory=False)
    val_dl   = DataLoader(val_ds,  batch_size=args.batch_size,shuffle=False,
                          num_workers=4,pin_memory=False)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size,shuffle=False,
                          num_workers=4,pin_memory=False)

    # ---------- modello  -------------------------------
    model = GNNCWT2D_Mk11_1sec(19,(40,500),3).to(dev)
    sd = torch.load(args.pretrained_ckpt, map_location=dev)
    if 'model_state_dict' in sd: sd = sd['model_state_dict']
    model.load_state_dict(sd, strict=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    writer = SummaryWriter(f'local/runs/reweight_ls_{datetime.now():%Y%m%d_%H%M%S}')

    best_val = 0.
    for ep in range(args.num_epochs):
        print(f'\n===== Epoch {ep:02d} =====')
        tr_loss,tr_acc = run_epoch(model,train_dl,dev,opt,args.alpha,
                                   args.invert,args.ls_eps,train=True)
        vl_loss,vl_acc = run_epoch(model,val_dl,dev,opt,args.alpha,
                                   args.invert,args.ls_eps,train=False)

        writer.add_scalars('Loss', {'train':tr_loss,'val':vl_loss}, ep)
        writer.add_scalars('Accuracy', {'train':tr_acc,'val':vl_acc}, ep)
        print(f'Acc (train/val): {tr_acc:.4f}/{vl_acc:.4f}   '
              f'Loss (train/val): {tr_loss:.4f}/{vl_loss:.4f}')

        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model.state_dict(),'best_reweight_ls.pt')
            print('>>> New best saved.')

    # ---------------- test finale ----------------------
    model.load_state_dict(torch.load('best_reweight_ls.pt',map_location=dev))
    _,ts_acc = run_epoch(model,test_dl,dev,opt,args.alpha,args.invert,
                         args.ls_eps,train=False)
    print(f'\n>>> Test accuracy (best): {ts_acc:.4f}')

if __name__ == '__main__':
    main()
