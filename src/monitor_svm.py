import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
import random

# --- Load activations ---
train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
test_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy', allow_pickle=True).item()
val_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy', allow_pickle=True).item()

for dataset_name, dataset in zip(['train_activations', 'test_activations', 'val_activations'], [train_activations, test_activations, val_activations]):
    list_of_data = [{'crop_file': key, 'valore': value} for key, value in dataset.items()]
    df = pd.DataFrame(list_of_data)
    globals()[f'{dataset_name}_df'] = df

train_activations_df['dataset'] = 'train'
test_activations_df['dataset'] = 'test'
val_activations_df['dataset'] = 'validation'

for df in [train_activations_df, test_activations_df, val_activations_df]:
    df['valore_softmax'] = df['valore'].apply(lambda x: softmax(x))
    df['pred_label'] = df['valore_softmax'].apply(lambda x: np.argmax(x))

all_activations_df = pd.concat([train_activations_df, test_activations_df, val_activations_df], ignore_index=True)

# --- Merge con etichette vere ---
annot = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv')
annot = annot.rename(columns={'label': 'true_label'})

true_pred = all_activations_df.merge(annot, on='crop_file')
true_pred = true_pred.rename(columns={'valore': 'activation_values', 'valore_softmax': 'softmax_values'})

# --- Setup per caricamento CWT ---
cwt_path = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
df_labels = true_pred.copy()

labels_dict = dict(zip(df_labels['crop_file'], (df_labels['true_label'] == df_labels['pred_label']).astype(int)))
dataset_dict = dict(zip(df_labels['crop_file'], df_labels['dataset']))

# --- Suddivisione crop_file per dataset ---
all_crop_files = [f for f in os.listdir(cwt_path) if f.endswith(".npy") and f in labels_dict]

from collections import defaultdict
dataset_files = defaultdict(list)

for f in all_crop_files:
    dataset_type = dataset_dict[f]
    dataset_files[dataset_type].append(f)

# --- Seleziona 10% casuale per ogni dataset ---
def sample_10_percent(file_list):
    k = max(1, int(len(file_list) * 0.1))
    return random.sample(file_list, k)

selected_files = []
for dataset_type in ['train', 'validation', 'test']:
    selected_files += sample_10_percent(dataset_files[dataset_type])

# --- Linearizzazione ---
def linearizza_batch(file_list, batch_size=1000):
    flattened_data = []
    labels = []
    datasets = []

    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i+batch_size]
        batch_data = []
        for f in batch_files:
            full_path = os.path.join(cwt_path, f)
            mat = np.load(full_path).astype(np.float32).flatten()
            batch_data.append(mat)
            labels.append(labels_dict[f])
            datasets.append(dataset_dict[f])
        flattened_data.append(np.array(batch_data))

    return np.vstack(flattened_data), np.array(labels), np.array(datasets)

print("🔄 Linearizzazione in corso solo su 10% di ciascun dataset...")
cwt_flattened, file_labels, file_datasets = linearizza_batch(selected_files)

# --- Creazione DataFrame finale ---
df_cwt_linearized = pd.DataFrame(cwt_flattened, dtype=np.float32)
df_cwt_linearized['label'] = file_labels
df_cwt_linearized['dataset'] = file_datasets
df_cwt_linearized.columns = df_cwt_linearized.columns.astype(str)

# --- Split train/val/test ---
train_df = df_cwt_linearized[df_cwt_linearized['dataset'] == 'train']
val_df = df_cwt_linearized[df_cwt_linearized['dataset'] == 'validation']
test_df = df_cwt_linearized[df_cwt_linearized['dataset'] == 'test']

X_train = train_df.drop(columns=['label', 'dataset'])
y_train = train_df['label']

X_val = val_df.drop(columns=['label', 'dataset'])
y_val = val_df['label']

X_test = test_df.drop(columns=['label', 'dataset'])
y_test = test_df['label']

# --- GridSearch SVM ---
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

print("GridSearchCV in corso...")
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Migliori parametri trovati: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# --- Valutazione su Validation ---
y_val_pred = best_model.predict(X_val)

print("\n📊 Metriche sul validation set:")
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_val_pred):.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))

# --- Retrain su Train + Validation ---
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

print("\n♻️ Riallenamento su train + validation...")
final_model = SVC(**grid_search.best_params_)
final_model.fit(X_trainval, y_trainval)

# --- Test finale ---
y_pred = final_model.predict(X_test)

print("\n📊 Metriche sul test set:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: 0', 'Pred: 1'],
            yticklabels=['True: 0', 'True: 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
