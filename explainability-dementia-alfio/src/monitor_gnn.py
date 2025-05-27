#!/usr/bin/env python3
"""
monitor_gnn.py
--------------
Costruisce il monitor Good/Bad usando il backbone GNN congelato
e i risultati di inferenza pre-esistenti in `true_pred.csv`.

• Calcola target Good (=1) se pred_label == true_label, altrimenti Bad (=0)
• Suddivide i crop in training / validation / test in base alla colonna `dataset`
• Congela tutti i layer, sostituisce lin6 con Linear(32→2)
• Ottimizza solo la nuova testa
• Salva il checkpoint con la migliore accuracy sul validation
• Produce CSV, classification report e confusion matrix per ogni split
"""

import os, argparse, torch, pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F

from utils   import seed_everything
from models  import GNNCWT2D_Mk11_1sec
from datasets import CWTGraphDataset
from torch_geometric.data import Data

# ------------------------------------------------------------------ #
#                         DATASET WRAPPER                            #
# ------------------------------------------------------------------ #
class MonitorGraphDatasetCSV(CWTGraphDataset):
    def __init__(self, annot_df, crop_dir, true_pred_csv):
        super().__init__(annot_df, crop_dir, norm_stats_path=None)

        tp = pd.read_csv(true_pred_csv)
        tp['good_label'] = (tp['pred_label'] == tp['true_label']).astype(int)
        self.good_map    = tp.set_index('crop_file')['good_label'].to_dict()

        # edge index copiato dall’implementazione originale
        self._edge_index = torch.tensor([[0, 0, 1, 1, 10, 10, 2, 2, 2, 2, 16, 16,
                                              16, 16, 16, 3, 3, 3, 3, 11, 11, 12, 12, 12,
                                              4, 4, 4, 4, 17, 17, 17, 17, 5, 5, 5, 5,
                                              13, 13, 13, 14, 14, 6, 6, 6, 6, 18, 18, 18,
                                              18, 18, 7, 7, 7, 7, 15, 15, 8, 8, 9, 9],
                                              [2, 16, 16, 3, 2, 12, 0, 16, 4, 10, 0, 1,
                                              3, 17, 2, 1, 11, 5, 16, 3, 13, 10, 4, 14,
                                              2, 17, 6, 12, 16, 5, 18, 4, 3, 13, 7, 17,
                                              11, 5, 15, 12, 6, 4, 18, 8, 14, 17, 7, 9,
                                              8, 6, 5, 15, 9, 18, 13, 7, 6, 18, 18, 7]])
    # --------------------------------------------------------------
    def get(self, idx):
        rec       = self.annot_df.iloc[idx]
        crop_path = os.path.join(self.dataset_crop_path, rec['crop_file'])
        cwt       = np.load(crop_path).astype(np.float32)

        x = np.moveaxis(cwt, 2, 0).reshape(19, -1)
        x = torch.tensor(x)
        y = torch.tensor([self.good_map[rec['crop_file']]], dtype=torch.long)

        data = Data(edge_index=self._edge_index, x=x, y=y)
        data.crop_file = rec['crop_file']  # necessario per l’export
        return data

# ------------------------------------------------------------------ #
#                       TRAIN / COLLECT UTILS                        #
# ------------------------------------------------------------------ #
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval(); correct = tot = 0
    for d in loader:
        d = d.to(device)
        pred = model(d.x, d.edge_index, d.batch).argmax(1)
        correct += (pred == d.y.squeeze()).sum().item()
        tot     += d.y.size(0)
    return correct / tot

@torch.no_grad()
def collect_results(model, loader, device):
    model.eval(); rows = []
    for d in loader:
        d = d.to(device)
        logits = model(d.x, d.edge_index, d.batch)
        soft   = F.softmax(logits, dim=1).cpu().numpy()
        out    = logits.cpu().numpy()
        pred   = soft.argmax(1)
        for i in range(pred.shape[0]):
            rows.append([
                d.crop_file[i],
                int(d.y[i].item()),
                int(pred[i]),
                out[i].tolist(),
                soft[i].tolist(),
                float(soft[i].max())
            ])
    y_true = np.array([r[1] for r in rows])
    y_pred = np.array([r[2] for r in rows])
    return rows, y_true, y_pred

# ------------------------------------------------------------------ #
#                             MAIN                                   #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--true_pred_csv', required=True)
    ap.add_argument('--cwt_dir',       required=True)
    ap.add_argument('--ckpt_gnn',      required=True)
    ap.add_argument('--device',        default='cuda:0')
    ap.add_argument('--epochs',        default=10, type=int)
    ap.add_argument('--batch_size',    default=64, type=int)
    ap.add_argument('--lr',            default=1e-3, type=float)
    ap.add_argument('--seed',          default=1234, type=int)
    ap.add_argument('--out_root',      default='/home/alfio/improving_dementia_detection_model/results_monitor_gnn')
    args = ap.parse_args()

    seed_everything(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ------------ prepara annotazioni e split -----------------------
    tp = pd.read_csv(args.true_pred_csv)
    tp['dataset'] = tp['dataset'].str.lower()
    tp['split']   = tp['dataset'].str.extract(r'(train|val|test)')
    # colonna fittizia label per compatibilità (non usata)
    tp['label']   = tp['true_label']

    split_df = {
        'train': tp[tp['split'] == 'train'],
        'val'  : tp[tp['split'] == 'val'],
        'test' : tp[tp['split'] == 'test']
    }

    ds = {k: MonitorGraphDatasetCSV(v, args.cwt_dir, args.true_pred_csv)
          for k,v in split_df.items()}
    dl = {k: DataLoader(ds[k], batch_size=args.batch_size, shuffle=(k=='train'),
                        num_workers=4, pin_memory=False) for k in ds}

    # ------------ modello ------------------------------------------
    model = GNNCWT2D_Mk11_1sec(19, (40,500), 3).to(dev)
    sd = torch.load(args.ckpt_gnn, map_location=dev)
    sd = sd['model_state_dict'] if 'model_state_dict' in sd else sd
    model.load_state_dict(sd, strict=False)

    for p in model.parameters(): p.requires_grad = False
    model.lin6 = torch.nn.Linear(model.lin6.in_features, 2).to(dev)
    torch.nn.init.xavier_uniform_(model.lin6.weight)

    opt = torch.optim.Adam(model.lin6.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.3, 1.0]).to(dev))

    run_id  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.out_root, run_id); os.makedirs(out_dir, exist_ok=True)
    tb      = SummaryWriter(f'/home/alfio/improving_dementia_detection_model/results_monitor_gnn/runs/monitor_gnn_{run_id}')

    best_val = 0.0
    for ep in range(args.epochs):
        model.train(); running = 0; corr = tot = 0
        for batch in tqdm(dl['train'], desc=f'Epoch {ep:02d}', ncols=100):
            batch = batch.to(dev); opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y.squeeze()); running += loss.item()
            loss.backward(); opt.step()
            corr += (out.argmax(1)==batch.y.squeeze()).sum().item(); tot += batch.y.size(0)

        tr_acc = corr / tot
        val_acc = accuracy(model, dl['val'], dev)
        tb.add_scalars('Acc', {'train': tr_acc, 'val': val_acc}, ep)
        print(f'[{ep:02d}] acc tr/val: {tr_acc:.3f}/{val_acc:.3f}')

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
            print('>>> nuovo best (val) salvato')

    # ------------ export risultati ---------------------------------
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best.pt'), map_location=dev))

    for split in ['train', 'val', 'test']:
        rows, y_true, y_pred = collect_results(model, dl[split], dev)
        # CSV
        pd.DataFrame(rows, columns=['crop_file','true_label','pred_label',
                                    'logits','softmax','goodness']
        ).to_csv(os.path.join(out_dir, f'{split}_results.csv'), index=False)
        # report
        with open(os.path.join(out_dir, f'{split}_classification_report.txt'), 'w') as f:
            f.write(classification_report(y_true, y_pred, digits=4))
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
        plt.title(f'Confusion – {split}'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{split}_confusion_matrix.png')); plt.close()

    print(f'\nRisultati salvati in {out_dir}')
    tb.close()

if __name__ == '__main__':
    main()
