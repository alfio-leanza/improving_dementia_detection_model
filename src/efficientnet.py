import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
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

# Path per CWT
cwt_path = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"

# Data split
data_split = {
    "train": [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "val": [54, 55, 56, 57, 58, 59, 79, 80, 81, 82, 83, 22, 23, 24, 25, 26, 27, 28],
    "test": [60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 29, 30, 31, 32, 33, 34, 35, 36]
}

df_labels = true_pred.copy()
df_labels['crop_file'] = df_labels['crop_file'].apply(lambda x: os.path.basename(x))
df_labels['train_label'] = (df_labels['pred_label'] == df_labels['true_label']).astype(int)
df_labels['train_label'].unique()

# Dataset per CWT
class CWT_Dataset(Dataset):
    def __init__(self, file_list, labels, root_dir):
        self.file_list = file_list
        self.labels = labels
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        label = self.labels[idx]
        file_path = os.path.join(self.root_dir, file_name)
        cwt_data = np.load(file_path)
        cwt_data = torch.tensor(cwt_data, dtype=torch.float32)
        cwt_data = cwt_data.permute(2, 0, 1)  # da (40,500,19) a (19,40,500)
        return cwt_data, label

def create_dataloader(split, batch_size=16):
    subset = df_labels[df_labels['original_rec'].isin([f'sub-{s:03d}' for s in data_split[split]])]
    file_list = list(subset["crop_file"])
    labels = list(subset["train_label"])
    dataset = CWT_Dataset(file_list, labels, cwt_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modello EfficientNet-B0 modificato: ingresso a 19 canali, dropout nel classifier
class EfficientNet19Channels(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(EfficientNet19Channels, self).__init__()
        # Carica EfficientNet-B0 pre-addestrato usando il parametro 'weights'
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Modifica il primo layer: EfficientNet-B0 ha self.model.features[0][0] come conv iniziale
        self.model.features[0][0] = nn.Conv2d(19, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Aggiorna il classificatore: EfficientNet-B0 normalmente usa un Dropout + Linear
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Funzione di training che tiene traccia della history
def train_model(model, train_loader, val_loader, epochs=10):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        val_loss, val_acc = evaluate_model(model, val_loader, return_loss=True)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
    
    return history

# Funzione di evaluation
def evaluate_model(model, loader, return_loss=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item()

    acc = 100. * correct / total
    avg_loss = running_loss / len(loader)

    if return_loss:
        return avg_loss, acc
    else:
        return acc, all_labels, all_preds

# Funzione per calcolare le metriche
def compute_metrics(true_labels, predictions):
    conf_matrix = confusion_matrix(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return {
        'conf_matrix': conf_matrix,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Funzione per salvare outputs e grafici nei risultati
def save_all_outputs(history, conf_matrix, true_labels, pred_labels, output_dir="/home/alfio/improving_dementia_detection_model/results_efficientnet"):
    os.makedirs(output_dir, exist_ok=True)

    # === Plot Training History ===
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    history_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(history_path)
    print(f"[INFO] Training history salvata in: {history_path}")
    plt.close()

    # === Plot Confusion Matrix ===
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[INFO] Confusion matrix salvata in: {cm_path}")
    plt.close()

    # === Classification Report (text) ===
    report = classification_report(true_labels, pred_labels, target_names=['Class 0', 'Class 1'])
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[INFO] Classification report salvato in: {report_path}")

    # === CSV: summary metrics ===
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    summary_df = pd.DataFrame([{
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }])
    metrics_path = os.path.join(output_dir, "metrics_summary.csv")
    summary_df.to_csv(metrics_path, index=False)
    print(f"[INFO] Metriche salvate in: {metrics_path}")

# Creazione dei dataloader
train_loader = create_dataloader("train")
val_loader = create_dataloader("val")
test_loader = create_dataloader("test")

# Setup dispositivo, modello, criterio e optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet19Channels(num_classes=2, dropout_rate=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training e valutazione
history = train_model(model, train_loader, val_loader, epochs=10)

test_acc, test_labels, test_preds = evaluate_model(model, test_loader)
metrics = compute_metrics(test_labels, test_preds)

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1 Score: {metrics['f1']:.2f}")

# Salvataggio di outputs e grafici
save_all_outputs(history, metrics['conf_matrix'], test_labels, test_preds)