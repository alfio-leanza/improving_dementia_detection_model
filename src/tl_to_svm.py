import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

# Caricamento delle attivazioni
train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
test_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy', allow_pickle=True).item()
val_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy', allow_pickle=True).item()

for dataset_name, dataset in zip(['train_activations', 'test_activations', 'val_activations'], 
                                 [train_activations, test_activations, val_activations]):
    list_of_data = [{'crop_file': key, 'valore': value} for key, value in dataset.items()]
    df = pd.DataFrame(list_of_data)
    globals()[f'{dataset_name}_df'] = df

train_activations_df['dataset'] = 'training'
test_activations_df['dataset'] = 'test'
val_activations_df['dataset'] = 'validation'

for dataset_name, dataset in zip(['train_activations_df', 'test_activations_df', 'val_activations_df'], 
                                 [train_activations_df, test_activations_df, val_activations_df]):
    globals()[f'{dataset_name}']['valore_softmax'] = globals()[f'{dataset_name}']['valore'].apply(lambda x: softmax(x))
    globals()[f'{dataset_name}']['pred_label'] = globals()[f'{dataset_name}']['valore_softmax'].apply(lambda x: np.argmax(x))

all_activations_df = pd.concat([train_activations_df, test_activations_df, val_activations_df], ignore_index=True)

# Caricamento annotazioni
annot = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv')
annot = annot.rename(columns={'label': 'true_label'})

true_pred = all_activations_df.merge(annot, on='crop_file')
true_pred = true_pred.rename(columns={'valore': 'activation_values'})
true_pred = true_pred.rename(columns={'valore_softmax': 'softmax_values'})


# === CWT Dataset ===
class CWT_Dataset(Dataset):
    def __init__(self, files, labels, root):
        self.files = files
        self.labels = labels
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cwt = np.load(os.path.join(self.root, self.files[idx]))  # shape: (40,500,19)
        cwt_img = np.mean(cwt, axis=2)  # media sui canali per avere (40,500)
        cwt_img = np.stack([cwt_img]*3, axis=-1)  # replica per 3 canali RGB
        img = self.transform(cwt_img).float()
        return img, self.labels[idx]

# === Caricamento dati ===
root_dir = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
df_labels = true_pred.copy()
df_labels['train_label'] = (df_labels['pred_label'] == df_labels['true_label']).astype(int)

splits = {
    "train": [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    "val": [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    "test": [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
}

loaders = {}
for split in splits:
    subset = df_labels[df_labels['original_rec'].isin([f'sub-{s:03d}' for s in splits[split]])]
    loaders[split] = DataLoader(CWT_Dataset(subset['crop_file'].values, subset['train_label'].values, root_dir), batch_size=16, shuffle=(split=='train'))

# === Estrazione feature con ResNet50 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(device)
resnet.fc = nn.Identity()  # rimuove la FC per estrazione feature
resnet.eval()

features, labels = {}, {}
for phase in ['train', 'val', 'test']:
    feats, labs = [], []
    for inputs, lbls in tqdm(loaders[phase], desc=f"Extracting {phase} features"):
        inputs = inputs.to(device)
        with torch.no_grad():
            f = resnet(inputs)
        feats.append(f.cpu().numpy())
        labs.extend(lbls.numpy())
    features[phase] = np.vstack(feats)
    labels[phase] = np.array(labs)

# === Grid Search SVM ===
param_grid = {
    'C': [0.1, 1,], # 0.01,10, 100],              # regularisation
    'gamma': [0.1, 1], # 0.01, 10, 100],          # kernel coefficient
    'kernel': ['linear', 'rbf'], # 'rbf', 'poly', 'sigmoid'],
    'degree': [2]# 3]                           # for polynomial kernel
}
svm = GridSearchCV(
    SVC(),
    param_grid,
    scoring='accuracy',
    verbose=3,
    n_jobs=-1
)
svm.fit(features['train'], labels['train'])

# === Valutazione e salvataggio risultati ===
y_pred = svm.predict(features['test'])
acc = accuracy_score(labels['test'], y_pred)
report = classification_report(labels['test'], y_pred)
cm = confusion_matrix(labels['test'], y_pred)

output_dir = "/home/alfio/improving_dementia_detection_model/results_tl_to_svm_grid"
os.makedirs(output_dir, exist_ok=True)

# Salvataggio confusion matrix come immagine
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

print(f"Test Accuracy: {acc}")
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

print(f"Risultati salvati in {output_dir}")