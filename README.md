# Data_science_thesis - Deep Learning in Healthcare: Dementia diagnosis through GNN

This repository contains the code and materials developed for my MSc thesis on **dementia diagnosis from resting-state EEG** using **Graph Neural Networks (GNNs)**.

**Task:** 3-class classification  
- **HC**: Healthy Controls  
- **AD**: Alzheimer’s Disease  
- **FTD**: Frontotemporal Dementia  

**Core ideas:**
- Convert EEG crops into **graph-structured samples** (nodes = electrodes).
- Compare a **baseline multiclass head (softmax)** vs an **One-vs-Rest (OVR) head**.
- Perform **subject-level diagnosis** via aggregation (voting) of crop-level predictions.
- Explore a **reliability monitor (Good/Bad)** experiment.

---

## Repository structure

- `baseline/`  
  Code for the **baseline GNN architecture** (multiclass head + softmax).

- `ovr/`  
  Code for the **OVR architecture** (3 independent binary heads: HC vs all, FTD vs all, AD vs all).

- `monitor/`  
  Code for the **exploratory reliability monitor** (Good/Bad) and related strategies (e.g., hybrid voting / discarding low-reliability crops).

- `results/`  
  Outputs for multiple runs/seeds (e.g., checkpoints, inference CSVs, metrics).

- `Leanza_Alfio_Thesis_LMDS.pdf`  
  Full thesis manuscript.

- `majority_voting.ipynb`  
  Notebook used for **subject-level aggregation** (e.g., weighted soft voting / majority voting) starting from crop-level inference files.

- `single_fold.sh`  
  Helper script to run **baseline**.

- `single_ovr.sh`  
  Helper script to run **OVR**.

---

## Method summary (high level)

1. **Crop extraction**
   - EEG is segmented into non-overlapping crops of duration `d = 1s`.
   - With sampling frequency `f_s = 500 Hz`, each crop has `N = d · f_s = 500` samples per channel.
   - Crop tensor (time-domain): `19 × 500` (19 electrodes).

2. **Time–frequency features**
   - Continuous Wavelet Transform (CWT) scalograms are computed per electrode (`F = 40` frequency bins).
   - A normalization step is applied.
   - Temporal compression is performed by splitting `N` into `B` blocks (`B = 20`) and averaging within each block.

3. **Graph construction**
   - Each crop becomes a graph `G = (V, E)`:
     - `|V| = 19` nodes (one per electrode).
     - `E` is a fixed sparse topology shared across samples.
   - Node features are built by vectorizing the processed time-frequency representation.

4. **Models**
   - **Baseline**: multiclass logits + softmax.
   - **OVR**: 3 binary heads + sigmoid; final class selected via argmax over the three scores.

5. **Subject-level decision**
   - Crop-level predictions are aggregated per subject (see `majority_voting.ipynb`).
