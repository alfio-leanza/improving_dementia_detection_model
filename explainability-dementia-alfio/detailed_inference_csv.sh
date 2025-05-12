#!/bin/bash

python3 src/detailed_inference_csv.py \
    --timestamp "train_20250510_172519" \
    --ds_parent_dir "/home/tom/dataset_eeg" \
    --ds_name "miltiadous_deriv_uV_d1.0s_o0.0s" \
    --classes "hc-ftd-ad" \
    --device "cuda:0" \
    --output_dir "/home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/detailed_inference/"
