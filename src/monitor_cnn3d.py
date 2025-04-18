import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


# ======== Modello: CNN3D con Channel Attention ========
class CNN3D_ChannelAttention(nn.Module):
    def __init__(self, in_channels=1, depth=19, num_classes=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),  # pool solo su freq/time
            nn.Dropout3d(0.2),

            nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Dropout3d(0.2),

            nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Dropout3d(0.3),

            nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((depth,1,1))  # mantieni depth dimension
        )
        self.channel_attention = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=(depth,1,1), bias=False),
            nn.ReLU(),
            nn.Conv3d(64, 256, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Dropout3d(0.6),
            nn.Conv3d(256, num_classes, kernel_size=(1,1,1))
        )

    def forward(self, x):
        # x shape: [B,1,19,40,500]
        feat = self.conv_block(x)          # [B,256,19,1,1]
        attn = self.channel_attention(feat)  # [B,256,1,1,1]
        weighted = feat * attn
        out = self.classifier(weighted)   # [B,2,19,1,1]
        return out.view(out.size(0), -1)  # [B,2]

# ======== Dataset 3D CWT ========
class CWT3D_Dataset(Dataset):
    def __init__(self, file_list, labels, root_dir, augment=False):
        self.file_list = file_list
        self.labels = labels
        self.root_dir = root_dir
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        label = self.labels[idx]
        data = np.load(os.path.join(self.root_dir, fname))  # shape (40,500,19)
        if self.augment:
            # rumore gaussiano
            data += np.random.normal(0, 0.01, data.shape)
            # masking frequenze
            if np.random.rand() < 0.5:
                m = np.random.randint(1, data.shape[2]//5)
                s = np.random.randint(0, data.shape[2]-m)
                data[:,:,s:s+m] = 0
            # masking tempo
            if np.random.rand() < 0.5:
                m = np.random.randint(1, data.shape[1]//5)
                s = np.random.randint(0, data.shape[1]-m)
                data[:,s:s+m,:] = 0
        # permute to [channels, freq, time] then add channel dim
        tensor = torch.tensor(data, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        return tensor, label

# ======== Evaluate ========
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return running_loss/len(loader), correct/total

# ======== Training ========
def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        running_loss, correct, total = 0.0,0,0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        tl, ta = running_loss/len(train_loader), correct/total
        vl, va = evaluate_model(model, val_loader, criterion, device)
        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)
        print(f"Epoch {epoch}: Train Loss={tl:.4f}, Train Acc={ta:.4f}, Val Loss={vl:.4f}, Val Acc={va:.4f}")
    return history

# ======== Plot History ========
def save_history_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(output_dir,'training_history_loss.png'))
    plt.close()
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig(os.path.join(output_dir,'training_history_acc.png'))
    plt.close()

# ======== Test & Save ========
def test_and_save_predictions(model, loader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, loader, criterion, device)
    preds, truths = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p)
            truths.extend(y.numpy())
    pd.DataFrame({'True':truths,'Predicted':preds}).to_csv(os.path.join(output_dir,'test_predictions.csv'), index=False)
    with open(os.path.join(output_dir,'classification_report.txt'),'w') as f:
        f.write(classification_report(truths,preds))
    cm = confusion_matrix(truths,preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
    plt.title('Confusion Matrix'); plt.ylabel('True'); plt.xlabel('Pred')
    plt.savefig(os.path.join(output_dir,'confusion_matrix.png'))
    plt.close()
    print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

# ======== Main ========
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Prepara df_labels e percorso CWT
    df_labels = true_pred.copy()
    df_labels['crop_file'] = df_labels['crop_file'].apply(lambda x: os.path.basename(x))
    df_labels['train_label'] = (df_labels['pred_label']==df_labels['true_label']).astype(int)
    cwt_root = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
    data_split = {
        'train': [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,66,67,68,69,70,71,72,73,74,75,76,77,78,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
        'val':   [54,55,56,57,58,59,79,80,81,82,83,22,23,24,25,26,27,28],
        'test':  [60,61,62,63,64,65,84,85,86,87,88,29,30,31,32,33,34,35,36]
    }
    def create_loader(split, augment=False):
        subset = df_labels[df_labels['original_rec'].isin([f"sub-{s:03d}" for s in data_split[split]])]
        files = list(subset['crop_file'])
        labels = list(subset['train_label'])
        return DataLoader(CWT3D_Dataset(files, labels, cwt_root, augment), batch_size=64, shuffle=True)
    train_loader = create_loader('train', augment=True)
    val_loader = create_loader('val')
    test_loader = create_loader('test')
    model = CNN3D_ChannelAttention().to(device)
    history = train_model(model, train_loader, val_loader, epochs=20, device=device)
    save_history_plots(history, output_dir="/home/alfio/improving_dementia_detection_model/results_cnn3d")
    test_and_save_predictions(model, test_loader, device, output_dir="/home/alfio/improving_dementia_detection_model/results_cnn3d")

if __name__=='__main__':
    main()
