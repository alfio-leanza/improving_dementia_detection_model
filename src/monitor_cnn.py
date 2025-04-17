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

# ======== Modello: CNN con Channel Attention ========
class CNN_ChannelAttention(nn.Module):
    def __init__(self, num_channels=19, num_classes=2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.channel_attention = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        attn_weights = self.channel_attention(x)
        x = x * attn_weights
        logits = self.classifier(x)
        return logits

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
        file_path = os.path.join(self.root_dir, file_name)
        cwt_data = np.load(file_path)

        if self.augment:
            noise = np.random.normal(0, 0.01, cwt_data.shape)
            cwt_data += noise

            if np.random.rand() < 0.5:
                freq_mask = np.random.randint(0, cwt_data.shape[0] // 5)
                freq_start = np.random.randint(0, cwt_data.shape[0] - freq_mask)
                cwt_data[freq_start:freq_start + freq_mask, :, :] = 0

            if np.random.rand() < 0.5:
                time_mask = np.random.randint(0, cwt_data.shape[1] // 5)
                time_start = np.random.randint(0, cwt_data.shape[1] - time_mask)
                cwt_data[:, time_start:time_start + time_mask, :] = 0

        cwt_data = torch.tensor(cwt_data, dtype=torch.float32).permute(2, 0, 1)
        return cwt_data, label

# ======== Training e Valutazione ========
def train_model(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

        evaluate_model(model, val_loader, device)

def evaluate_model(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            truths.extend(labels.cpu().numpy())

    accuracy = accuracy_score(truths, preds)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(classification_report(truths, preds))
    print(confusion_matrix(truths, preds))

# ======== Predizione sul Test e Salvataggio Output ========
def test_and_save_predictions(model, loader, device, output_dir="/home/alfio/improving_dementia_detection_model/results_cnn"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            truths.extend(labels.numpy())

    results_df = pd.DataFrame({"True": truths, "Predicted": preds})
    results_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    report = classification_report(truths, preds)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    conf_matrix = confusion_matrix(truths, preds)
    pd.DataFrame(conf_matrix).to_csv(os.path.join(output_dir, "confusion_matrix.csv"), index=False)

    print(f"Test results saved in {output_dir}")

# ======== Esecuzione Completa Integrata ========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_labels = true_pred.copy()
    df_labels['crop_file'] = df_labels['crop_file'].apply(lambda x: os.path.basename(x))
    df_labels['train_label'] = (df_labels['pred_label'] == df_labels['true_label']).astype(int)
    cwt_path = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"

    # Data split
    data_split = {
        "train": [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "val": [54, 55, 56, 57, 58, 59, 79, 80, 81, 82, 83, 22, 23, 24, 25, 26, 27, 28],
        "test": [60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 29, 30, 31, 32, 33, 34, 35, 36]
    }
    def create_dataloader(split, batch_size=64, augment=False):
        subset = df_labels[df_labels['original_rec'].isin([f'sub-{s:03d}' for s in data_split[split]])]
        dataset = CWT_Dataset(list(subset["crop_file"]), list(subset["train_label"]), cwt_path, augment)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_dataloader("train", augment=True)
    val_loader = create_dataloader("val")
    test_loader = create_dataloader("test")

    model = CNN_ChannelAttention()
    train_model(model, train_loader, val_loader, epochs=20, device=device)

    test_and_save_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()
