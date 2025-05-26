#!/usr/bin/env python3
"""
Fine-tuning del monitor Good/Bad usando la GNN congelata
e il file true_pred.csv per etichette e split.
"""

import os, argparse, torch, pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np

from utils                       import seed_everything
from monitor_dataset_fromcsv     import MonitorGraphDatasetCSV
from models                      import GNNCWT2D_Mk11_1sec

# ------------------ ARGPARSE -----------------------------------------
def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_true_pred', required=True,
                   help='path a true_pred.csv con softmax, pred, true ecc.')
    p.add_argument('-d','--device', default='cuda:0')
    p.add_argument('--epochs', default=10, type=int)
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--seed', default=1234, type=int)
    p.add_argument('--ckpt_gnn', required=True,
                   help='checkpoint GNN tre-classi')
    p.add_argument('--cwt_dir', required=True,
                   help='cartella con i crop CWT (*.npy)')
    p.add_argument('--out_dir', default='/home/alfio/improving_dementia_detection_model/results_monitor_gnn')
    return p.parse_args()

# ------------------ TRAIN / EVAL -------------------------------------
@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval(); corr=tot=0
    for d in loader:
        d=d.to(device)
        out=model(d.x,d.edge_index,d.batch)
        corr+=(out.argmax(1)==d.y.squeeze()).sum().item()
        tot += d.y.size(0)
    return corr/tot

@torch.no_grad()
def collect(model, loader, device):
    model.eval(); rows=[]
    for d in loader:
        d=d.to(device)
        out=model(d.x,d.edge_index,d.batch)
        sm = F.softmax(out, dim=1).cpu().numpy()
        lg = out.cpu().numpy()
        pred = sm.argmax(1)
        for fn,y,l,s in zip(d.crop_file,d.y.cpu(),lg,sm):
            rows.append([fn,int(y),int(pred[len(rows)]),l.tolist(),s.tolist(),float(s.max())])
    return rows

# ---------------------------------------------------------------------
def main():
    a = argparser()
    seed_everything(a.seed)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')

    # ---- leggi true_pred.csv per split -------------------------------
    tp = pd.read_csv(a.csv_true_pred)
    train_annot = tp[tp['dataset']=='train']
    val_annot   = tp[tp['dataset']=='val']
    test_annot  = tp[tp['dataset']=='test']

    train_ds = MonitorGraphDatasetCSV(train_annot, a.cwt_dir, a.csv_true_pred)
    val_ds   = MonitorGraphDatasetCSV(val_annot,   a.cwt_dir, a.csv_true_pred)
    test_ds  = MonitorGraphDatasetCSV(test_annot,  a.cwt_dir, a.csv_true_pred)

    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                          num_workers=4, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=a.batch_size, shuffle=False,
                          num_workers=4, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=a.batch_size, shuffle=False,
                          num_workers=4, pin_memory=False)

    # ---- modello -----------------------------------------------------
    model = GNNCWT2D_Mk11_1sec(19,(40,500),3).to(dev)
    sd=torch.load(a.ckpt_gnn, map_location=dev)
    sd=sd['model_state_dict'] if 'model_state_dict' in sd else sd
    model.load_state_dict(sd, strict=False)
    for p in model.parameters(): p.requires_grad=False
    in_feat=model.lin6.in_features
    model.lin6=torch.nn.Linear(in_feat,2).to(dev)
    torch.nn.init.xavier_uniform_(model.lin6.weight)

    opt=torch.optim.Adam(model.lin6.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.3,1.0]).to(dev))

    run_id=datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir=os.path.join(a.out_dir, run_id); os.makedirs(out_dir, exist_ok=True)
    tb=SummaryWriter(f'/home/alfio/improving_dementia_detection_model/results_monitor_gnn/runs/monitor_gnn_csv_{run_id}')

    best_val=0.0
    for ep in range(a.epochs):
        model.train(); run_loss=0; corr=tot=0
        for d in tqdm(train_dl, desc=f'Epoch {ep:02d}', ncols=100):
            d=d.to(dev); opt.zero_grad()
            out=model(d.x,d.edge_index,d.batch)
            loss=loss_fn(out,d.y.squeeze()); run_loss+=loss.item()
            loss.backward(); opt.step()
            corr+=(out.argmax(1)==d.y.squeeze()).sum().item(); tot+=d.y.size(0)

        tr_acc=corr/tot; tr_loss=run_loss/len(train_dl)
        val_acc=eval_acc(model,val_dl,dev)
        tb.add_scalars('Acc', {'train':tr_acc,'val':val_acc}, ep)
        tb.add_scalar('Loss/train', tr_loss, ep)
        print(f'[{ep:02d}] acc tr/val: {tr_acc:.3f}/{val_acc:.3f}')

        if val_acc>best_val:
            best_val=val_acc
            torch.save(model.state_dict(), os.path.join(out_dir,'best.pt'))
            print('>>> nuovo best (val) salvato')

    # ---- export risultati -------------------------------------------
    model.load_state_dict(torch.load(os.path.join(out_dir,'best.pt'), map_location=dev))

    for split,dl in zip(['train','val','test'],[train_dl,val_dl,test_dl]):
        rows=collect(model,dl,dev)
        # csv
        pd.DataFrame(rows, columns=['crop_file','true_label','pred_label',
                                    'logits','softmax','goodness']
        ).to_csv(os.path.join(out_dir,f'{split}_results.csv'), index=False)

        y_true=np.array([r[1] for r in rows]); y_pred=np.array([r[2] for r in rows])
        with open(os.path.join(out_dir,f'{split}_classification_report.txt'),'w') as f:
            f.write(classification_report(y_true,y_pred,digits=4))
        cm=confusion_matrix(y_true,y_pred)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                    xticklabels=['Bad','Good'],yticklabels=['Bad','Good'])
        plt.title(f'Confusion – {split}'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,f'{split}_confusion_matrix.png')); plt.close()

    print(f'\nRisultati salvati in {out_dir}')
    tb.close()

if __name__=='__main__':
    main()
