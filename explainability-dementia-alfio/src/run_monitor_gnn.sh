#!/usr/bin/env bash
set -euo pipefail

python3 monitor_gnn.py \
  --true_pred_csv "/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/detailed_inference/true_pred.csv" \
  --cwt_dir       "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt" \
  --ckpt_gnn      "/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt" \
  --device cuda:0 --epochs 10 --batch_size 64 --lr 1e-3
