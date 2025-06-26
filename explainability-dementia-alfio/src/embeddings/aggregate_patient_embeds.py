"""
Media (o altro pooling) degli embedding per paziente.
"""
import argparse, pandas as pd, numpy as np
from pathlib import Path


def main(cfg):
    seg = pd.read_parquet(cfg.input)
    pat = (
        seg.groupby("original_rec")["embedding"]
        .apply(lambda v: np.mean(np.stack(v), axis=0))
    )
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    pat.to_pickle(cfg.output)
    print(f"[✓] Salvato embedding per {len(pat)} pazienti → {cfg.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="embeddings.parquet")
    p.add_argument("--output", default="patient_embeds.pkl")
    main(p.parse_args())
