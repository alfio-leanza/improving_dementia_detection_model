import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

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
        cwt_img = np.mean(cwt, axis=2)  # average over channels to get (40,500)
        cwt_img = np.stack([cwt_img]*3, axis=-1)  # replicate as 3-channel RGB
        img = self.transform(cwt_img).float()
        return img, self.labels[idx]

# === Load activations and labels ===
train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle=True).item()
test_activations  = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy', allow_pickle=True).item()
val_activations   = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy', allow_pickle=True).item()

# Convert to DataFrames
def build_df(act_dict, split_name):
    data = [{'crop_file':k, 'valore':v} for k,v in act_dict.items()]
    df = pd.DataFrame(data)
    df['dataset'] = split_name
    df['valore_softmax'] = df['valore'].apply(lambda x: softmax(x))
    df['pred_label'] = df['valore_softmax'].apply(lambda x: np.argmax(x))
    return df

train_df = build_df(train_activations, 'train')
test_df  = build_df(test_activations, 'test')
val_df   = build_df(val_activations, 'val')
all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Load true labels
a = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv')
a = a.rename(columns={'label':'true_label'})
true_pred = all_df.merge(a, on='crop_file')
true_pred['train_label'] = (true_pred['pred_label'] == true_pred['true_label']).astype(int)

# Create DataLoaders
root_dir = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
splits = {
    'train': [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    'val':   [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
    'test':  [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
}

def create_loader(split, batch_size=16, augment=False):
    recs = [f'sub-{s:03d}' for s in splits[split]]
    df_sub = true_pred[true_pred['original_rec'].isin(recs)]
    files = df_sub['crop_file'].tolist()
    labels = df_sub['train_label'].tolist()
    ds = CWT_Dataset(files, labels, root_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'))

train_loader = create_loader('train', augment=True)
val_loader   = create_loader('val')
test_loader  = create_loader('test')

# === Feature extraction with ResNet50 ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
resnet.fc = nn.Identity()
resnet.eval()

def extract_features(loader):
    feats, labs = [], []
    for imgs, lbl in tqdm(loader, desc='Extracting'):
        imgs = imgs.to(device)
        with torch.no_grad(): out = resnet(imgs)
        feats.append(out.cpu().numpy())
        labs.extend(lbl.numpy())
    return np.vstack(feats), np.array(labs)

X_train, y_train = extract_features(train_loader)
X_val,   y_val   = extract_features(val_loader)
X_test,  y_test  = extract_features(test_loader)

# === SVM Classification (fixed hyperparameters) with verbose progress ===
# Use verbose=True to see libsvm iteration progress
svm = SVC(C=0.01, kernel='poly', gamma=0.01, degree=4, verbose=True)
# Fit will print progress messages to stdout
svm.fit(X_train, y_train)

# === Test evaluation ===
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# === Save results and plots ===
output_dir = '/home/alfio/improving_dementia_detection_model/results_tl_to_svm'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Pred')
plt.savefig(os.path.join(output_dir,'confusion_matrix.png'))
plt.close()

with open(os.path.join(output_dir,'classification_report.txt'),'w') as f:
    f.write(report)

print(f'Test Accuracy: {acc:.4f}')
print(f'Results saved in {output_dir}')
