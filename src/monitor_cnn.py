import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

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

# ======== Modello: CNN Profonda con Channel Attention e Maggiore Regolarizzazione ========
class CNN_ChannelAttention(nn.Module):
    def __init__(self, num_channels=19, num_classes=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d(1)
        )
        self.channel_attention = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 256, bias=False),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        attn_weights = self.channel_attention(x)
        x = x * attn_weights
        return self.classifier(x)

# ======== Dataset con EEG-specific Data Augmentation ========
class CWT_Dataset(Dataset):
    def __init__(self, file_list, labels, root_dir, augment=False):
        self.file_list = file_list
        self.labels = labels
        self.root_dir = root_dir
        self.augment = augment
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        label = self.labels[idx]
        data = np.load(os.path.join(self.root_dir, file_name))
        if self.augment:
            noise = np.random.normal(0, 0.01, data.shape)
            data += noise
            if np.random.rand() < 0.5:
                fmask = np.random.randint(0, data.shape[0]//5)
                start = np.random.randint(0, data.shape[0]-fmask)
                data[start:start+fmask,:,:] = 0
            if np.random.rand() < 0.5:
                tmask = np.random.randint(0, data.shape[1]//5)
                start = np.random.randint(0, data.shape[1]-tmask)
                data[:,start:start+tmask,:] = 0
        tensor = torch.tensor(data, dtype=torch.float32).permute(2,0,1)
        return tensor, label

# ======== Evaluate con loss e accuracy ========
def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss/len(loader), correct/total

# ======== Training con history ========
def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        running_loss, correct, total = 0.0,0,0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
        tl, ta = running_loss/len(train_loader), correct/total
        vl, va = evaluate_model(model, val_loader, criterion, device)
        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)
        print(f"Epoch {epoch}: Train Loss={tl:.4f}, Train Acc={ta:.4f}, Val Loss={vl:.4f}, Val Acc={va:.4f}")
    return history

# ======== Test e Salvataggio Output con plot immagine CM e history ========
def test_and_save_predictions(model, loader, device, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, loader, criterion, device)
    preds, truths = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            p = outputs.argmax(dim=1).cpu().numpy()
            preds.extend(p)
            truths.extend(labels.numpy())
    # Salva predizioni e report
    pd.DataFrame({"True":truths, "Predicted":preds}).to_csv(os.path.join(output_dir,"test_predictions.csv"), index=False)
    with open(os.path.join(output_dir,"classification_report.txt"),"w") as f:
        f.write(classification_report(truths,preds))
    # Plot e salva confusion matrix come immagine
    cm = confusion_matrix(truths, preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad','Good'], yticklabels=['Bad','Good'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'confusion_matrix.png'))
    plt.close()
    print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
    print(f"Results saved in {output_dir}")

# ======== Salvataggio plot history ========
def save_history_plots(history, output_dir="/home/alfio/improving_dementia_detection_model/results_cnn"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss'])+1)
    # Loss
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir,'training_history_loss.png'))
    plt.close()
    # Accuracy
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir,'training_history_acc.png'))
    plt.close()

# ======== Main ========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_labels = true_pred.copy()
    df_labels['crop_file'] = df_labels['crop_file'].apply(lambda x: os.path.basename(x))
    df_labels['train_label'] = (df_labels['pred_label']==df_labels['true_label']).astype(int)
    cwt_root = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"

    # Data split
    data_split = {
        "train": [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "val": [54, 55, 56, 57, 58, 59, 79, 80, 81, 82, 83, 22, 23, 24, 25, 26, 27, 28],
        "test": [60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 29, 30, 31, 32, 33, 34, 35, 36]
    }
    def create_loader(split, batch_size=64, augment=False):
        subset = df_labels[df_labels['original_rec'].isin([f'sub-{s:03d}' for s in data_split[split]])]
        files = list(subset['crop_file'])
        labels = list(subset['train_label'])
        return DataLoader(CWT_Dataset(files, labels, cwt_root, augment), batch_size=batch_size, shuffle=True)
    
    train_loader = create_loader('train', augment=True)
    val_loader = create_loader('val')
    test_loader = create_loader('test')
    model = CNN_ChannelAttention()
    history = train_model(model, train_loader, val_loader, epochs=20, device=device)
    save_history_plots(history)
    test_and_save_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()

