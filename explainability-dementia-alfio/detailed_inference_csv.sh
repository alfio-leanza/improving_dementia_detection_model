#!/bin/bash

python src/detailed_inference_csv.py \
    --timestamp "train_20250114_014153" \
    --ds_parent_dir "/home/tom/fast2/gnn-datasets" \
    --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
    --classes "hc-ftd-ad" \
    --device "cuda:0" \
