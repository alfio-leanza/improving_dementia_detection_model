"""
Inference sul dataset: salva un Parquet con un embedding 32-D per ogni segmento.
"""
from pathlib import Path
import argparse, pandas as pd, numpy as np, torch
from torch_geometric.loader import DataLoader

from cwt_dataset import CWTGraphDataset
from model import GNNCWT2D_Mk11_1sec


def main(cfg):
    annot = pd.read_csv(cfg.annot_csv)
    ds = CWTGraphDataset(
        annot_df=annot,
        dataset_crop_path=cfg.crops_dir,
        norm_stats_path=cfg.norm_stats,
        augment=False,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    model = GNNCWT2D_Mk11_1sec(
        n_electrodes=19,
        cwt_size=(40, 500),
        num_classes=cfg.num_classes,
    )
    #model.load_state_dict(torch.load(cfg.checkpoint, map_location="cpu"))
        # --- CARICAMENTO SICURO DEL CHECKPOINT ---------------------------------
    ckpt = torch.load(cfg.checkpoint, map_location="cpu")

    # 1) se è stato salvato con torch.save({ ... 'model_state_dict': ... })
    state = ckpt.get("model_state_dict", ckpt)   # fallback al dict intero

    # 2) sostituisci i prefissi per compatibilità (gconv1 → g1, gconv2 → g2)
    from collections import OrderedDict
    state_fixed = OrderedDict()
    for k, v in state.items():
        if k.startswith("gconv1"):
            k = k.replace("gconv1", "g1", 1)
        elif k.startswith("gconv2"):
            k = k.replace("gconv2", "g2", 1)
        state_fixed[k] = v

    # 3) carica (strict=True assicura che ora tutto combaci)
    model.load_state_dict(state_fixed, strict=True)
    print("[✓] checkpoint caricato correttamente")

    model.cuda().eval()

    out = []
    with torch.no_grad():
        for batch in dl:
            x = batch.x.view(batch.num_graphs, 19, 40, 500).cuda()
            ei = batch.edge_index.cuda()
            b  = batch.batch.cuda()

            emb = model.embed(x, ei, b).cpu().numpy()

            for e, pid, f, s, t, lab in zip(
                emb,
                batch.pid,
                batch.crop_file,
                batch.start_sec,
                batch.end_sec,
                batch.gt_label,
            ):
                out.append(
                    {
                        "patient_id": pid,
                        "file": f,
                        "start_sec": s,
                        "end_sec": t,
                        "gt_label": lab,
                        "embedding": e.tolist(),  # 32-element list
                    }
                )

    df = pd.DataFrame(out)
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.output, index=False)
    print(f"[✓] Salvati {len(df)} segmenti in {cfg.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--annot_csv",  required=True)
    p.add_argument("--crops_dir",  required=True)
    p.add_argument("--norm_stats")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     default="embeddings.parquet")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=2)
    main(p.parse_args())
