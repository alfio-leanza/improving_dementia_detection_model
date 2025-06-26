# 1) Estrazione segmenti → Parquet
#python3 extract_embeddings.py \
 # --annot_csv  /home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv \
  #--crops_dir  /home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt \
  #--checkpoint /home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/checkpoints/train_20250510_172519/best_test_acc.pt \
  #--output /home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/embeddings/checkpoint/embeddings.parquet

# 2) Pooling per paziente
python3 aggregate_patient_embeds.py \
 --input /Volumes/Extreme SSD/progetto_tesi/improving_dementia_detection_model/explainability-dementia-alfio/local/embeddings/checkpoint/embeddings.parquet \
 --output /home/alfio/improving_dementia_detection_model/explainability-dementia-alfio/local/embeddings/checkpoint/patient_embeds.pkl