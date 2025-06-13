#!/bin/bash

# Dataset d1.0s o0.0s CWT
python3 src/single_ovo.py \
    --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
    --classes "hc-ftd-ad" \
    --ds_parent_dir "/home/tom/dataset_eeg" \
    --device "cuda:0" \
    --seed 1234 \
    --batch_size 256 \
    --num_workers 8 \
    --num_epochs 80 \
    --weight_decay 1e-7 \

