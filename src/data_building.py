import pandas as pd
import numpy as np
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

true_pred.to_csv('/home/tom/dataset_eeg/inference_20250327_171717/true_pred.csv', index = False)