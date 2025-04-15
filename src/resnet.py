import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax

train_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/train_activations.npy', allow_pickle = True,).item()
test_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/test_activations.npy', allow_pickle = True,).item()
val_activations = np.load('/home/tom/dataset_eeg/inference_20250327_171717/val_activations.npy', allow_pickle = True,).item()

for dataset_name, dataset in zip(['train_activations', 'test_activations', 'val_activations'], [train_activations, test_activations, val_activations]):
    list_of_data = [{'crop_file': key, 'valore': value} for key, value in dataset.items()]
    df = pd.DataFrame(list_of_data)
    globals()[f'{dataset_name}_df'] = df

train_activations_df['dataset'] = 'training'
test_activations_df['dataset'] = 'test'
val_activations_df['dataset'] = 'validation'

for dataset_name, dataset in zip(['train_activations_df', 'test_activations_df', 'val_activations_df'], [train_activations_df, test_activations_df, val_activations_df]):
    globals()[f'{dataset_name}']['valore_softmax'] = globals()[f'{dataset_name}']['valore'].apply(lambda x: softmax(x))
    globals()[f'{dataset_name}']['pred_label'] = globals()[f'{dataset_name}']['valore_softmax'].apply(lambda x: np.argmax(x))

all_activations_df = pd.concat([train_activations_df,test_activations_df,val_activations_df], ignore_index = True)

annot = pd.read_csv('/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/annot_all_hc-ftd-ad.csv')

annot = annot.rename(columns={'label':'true_label'})

true_pred = all_activations_df.merge(annot, on = 'crop_file')

true_pred = true_pred.rename(columns={'valore':'activation_values'})
true_pred = true_pred.rename(columns={'valore_softmax':'softmax_values'})
#root = tk.Tk()
#root.withdraw()

cwt_path = "/home/tom/dataset_eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt"
#labels_path = '/home/tom/dataset_eeg/inference_20250327_171717/true_pred.csv'

data_split = {
    "train": [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "val": [54, 55, 56, 57, 58, 59, 79, 80, 81, 82, 83, 22, 23, 24, 25, 26, 27, 28],
    "test": [60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 29, 30, 31, 32, 33, 34, 35, 36]
}

df_labels = true_pred.copy()


df_labels['crop_file'] = df_labels['crop_file'].apply(lambda x: os.path.basename(x))

df_labels['train_label'] = (df_labels['pred_label'] == df_labels['true_label']).astype(int)
df_labels['train_label'].unique()

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
        cwt_data = cwt_data.permute(2, 0, 1)
        return cwt_data, label

def create_dataloader(split, batch_size=16):
    subset = df_labels[df_labels['original_rec'].isin([f'sub-{s:03d}' for s in data_split[split]])]
    file_list = list(subset["crop_file"])
    labels = list(subset["train_label"])
    dataset = CWT_Dataset(file_list, labels, cwt_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class ResNet18_19Channels(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18_19Channels, self).__init__()
        self.model = models.resnet18(pretrained=True)
        
        self.model.conv1 = nn.Conv2d(19, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, val_loader, epochs=10):
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
        
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Train Acc={100.*correct/total:.2f}%, Val Acc={val_acc:.2f}%")

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

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

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(6,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

train_loader = create_dataloader("train")
val_loader = create_dataloader("val")
test_loader = create_dataloader("test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18_19Channels(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, val_loader, epochs=10)

test_acc, test_labels, test_preds = evaluate_model(model, test_loader)
metrics = compute_metrics(test_labels, test_preds)

print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"F1 Score: {metrics['f1']:.2f}")

plot_confusion_matrix(metrics['conf_matrix'])
