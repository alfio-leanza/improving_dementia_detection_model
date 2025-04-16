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

cwt_path = '/Users/alfioleanza/progetto_tesi/dataset-eeg/miltiadous_deriv_uV_d1.0s_o0.0s/cwt'
labels_path = '/Users/alfioleanza/progetto_tesi/dataset-eeg/inference_20250327_171717/true_pred.csv'

df_labels = pd.read_csv(labels_path)
labels_dict = dict(zip(df_labels['crop_file'], (df_labels['true_label'] == df_labels['pred_label']).astype(int)))

cwt_matrices = []
file_labels = []
file_datasets = []

for file_name in tqdm(os.listdir(cwt_path)):
    if file_name.endswith(".npy") and file_name in labels_dict:
        file_path = os.path.join(cwt_path, file_name)
        
        cwt_matrix = np.load(file_path).astype(np.float32)
        cwt_matrices.append(cwt_matrix)
        file_labels.append(labels_dict[file_name])
        file_datasets.append(labels_dict[file_name]['dataset'])

def linearizza_cwt(cwt_list):
    n = len(cwt_list)
    flattened_size = cwt_list[0].size 
    output_array = np.empty((n, flattened_size), dtype=np.float32)
    
    return output_array

if cwt_matrices:
    cwt_flattened = linearizza_cwt(cwt_matrices)

    df_cwt_linearized = pd.DataFrame(cwt_flattened, dtype=np.float32)
    df_cwt_linearized['label'] = file_labels
    df_cwt_linearized.columns = df_cwt_linearized.columns.astype(str)


train_df = df_cwt_linearized[df_cwt_linearized['dataset'] == 'train']
val_df = df_cwt_linearized[df_cwt_linearized['dataset'] == 'validation']
test_df = df_cwt_linearized[df_cwt_linearized['dataset'] == 'test']

X_train = train_df.drop(columns=['label', 'dataset'])
y_train = train_df['label']

X_val = val_df.drop(columns=['label', 'dataset'])
y_val = val_df['label']

X_test = test_df.drop(columns=['label', 'dataset'])
y_test = test_df['label']


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


y_val_pred = best_model.predict(X_val)

print("\n Metriche sul validation set:")
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_val_pred):.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_val, y_val_pred, zero_division=0))


X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

print("\n Riallenamento su train + validation...")
final_model = SVC(**grid_search.best_params_)
final_model.fit(X_trainval, y_trainval)


y_pred = final_model.predict(X_test)

print("\n📊 Metriche sul test set:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


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
