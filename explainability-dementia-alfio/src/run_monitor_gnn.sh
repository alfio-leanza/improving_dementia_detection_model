#!/usr/bin/env bash
set -euo pipefail

CSV_TP="/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/detailed_inference/true_pred.csv"
CKPT_GNN="/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt"
CWT_DIR="/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"

python3 monitor_gnn.py \
  --csv_true_pred "${CSV_TP}" \
  --ckpt_gnn "${CKPT_GNN}" \
  --cwt_dir "${CWT_DIR}" \
  --device cuda:0 \
  --epochs 10 --batch_size 64 --lr 1e-3
