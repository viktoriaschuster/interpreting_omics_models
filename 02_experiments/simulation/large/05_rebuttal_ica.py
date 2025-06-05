import os
import torch
import pandas as pd
import numpy as np
import random
import tqdm
import gc

import sys
sys.path.append(".")
sys.path.append('src')

from src.functions.sae_analysis_sim3 import *

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

complexity = 'high'
n_samples = 100000
data_dir = '/home/vschuste/data/simulation/'

for seed in range(10):
    temp_y = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed), weights_only=False)
    temp_x0 = torch.load(data_dir+'large_{}-complexity_rs{}_x0.pt'.format(complexity, seed), weights_only=False)
    temp_x1 = torch.load(data_dir+'large_{}-complexity_rs{}_x1.pt'.format(complexity, seed), weights_only=False)
    temp_x2 = torch.load(data_dir+'large_{}-complexity_rs{}_x2.pt'.format(complexity, seed), weights_only=False)
    temp_ct = torch.load(data_dir+'large_{}-complexity_rs{}_ct.pt'.format(complexity, seed), weights_only=False)
    temp_cov = torch.load(data_dir+'large_{}-complexity_rs{}_co.pt'.format(complexity, seed), weights_only=False)
    if seed == 0:
        rna_counts = temp_y
        x0 = temp_x0
        x1 = temp_x1
        x2 = temp_x2
        ct = temp_ct
        co = temp_cov
    else:
        rna_counts = torch.cat((rna_counts, temp_y), dim=0)
        x0 = torch.cat((x0, temp_x0), dim=0)
        x1 = torch.cat((x1, temp_x1), dim=0)
        x2 = torch.cat((x2, temp_x2), dim=0)
        ct = torch.cat((ct, temp_ct), dim=0)
        co = torch.cat((co, temp_cov), dim=0)
# limit to the training data
n_samples_train = int(n_samples*0.9)
rna_counts = rna_counts[:n_samples_train]
x0 = x0[:n_samples_train]
x1 = x1[:n_samples_train]
x2 = x2[:n_samples_train]
ct = ct[:n_samples_train]
co = co[:n_samples_train]
# also make this faster by taking every 10th sample
rna_counts = rna_counts[::3]
x0 = x0[::3]
x1 = x1[::3]
x2 = x2[::3]
ct = ct[::3]
co = co[::3]

print("Data loaded.")
print(f"Running on a subset with {rna_counts.shape[0]} samples.")

def pearsonr(a,b):
    #cov = torch.mean((a - a.mean(dim=0).unsqueeze(0)).unsqueeze(1) * (b - b.mean(dim=0).unsqueeze(0)).unsqueeze(-1), dim=0)
    cov = torch.mean((a - a.mean(dim=0)) * (b - b.mean()).unsqueeze(-1), dim=0)
    #std_a = a.std(dim=0)
    std_a = a.std(dim=0)
    #std_b = b.std(dim=0)
    std_b = b.std()
    return cov / (std_a * std_b)

def get_correlations_with_data(activations, unique_activs, comparison_data, device='cpu'):
    correlations_p = torch.zeros((len(unique_activs), comparison_data.shape[1]))
    
    # Move data to GPU once, not in every loop iteration
    with torch.no_grad():        
        # Process in smaller batches to avoid memory issues
        batch_size = 20000  # Adjust based on your GPU memory
        for start_idx in range(0, comparison_data.shape[1], batch_size):
            end_idx = min(start_idx + batch_size, comparison_data.shape[1])
            
            # Calculate correlations for the batch
            for j in tqdm.tqdm(range(0, comparison_data.shape[1])):
                correlations_p[start_idx:end_idx, j] = pearsonr(activations[:,start_idx:end_idx].to(device), comparison_data[:, j].to(device)).cpu()
            
            gc.collect()
            torch.cuda.empty_cache()
        
        gc.collect()
        torch.cuda.empty_cache()
        
    return correlations_p.numpy()

def get_number_of_redundant_features(activations, threshold=0.95, device='cpu'):
    # compute correlations between all active features
    redundant_set = set()
    # Move data to GPU once, not in every loop iteration
    with torch.no_grad():        
        # Process in smaller batches to avoid memory issues
        batch_size = 20000  # Adjust based on your GPU memory
        for j in tqdm.tqdm(range(0, activations.shape[1])):
            corr_temp = torch.zeros(activations.shape[1])
            for start_idx in range(0, activations.shape[1], batch_size):
                end_idx = min(start_idx + batch_size, activations.shape[1])
                corr_temp[start_idx:end_idx] = pearsonr(activations[:,start_idx:end_idx].to(device), activations[:, j].to(device)).cpu()
                gc.collect()
                torch.cuda.empty_cache()
        
        redundant_set.update([j for j in np.where(corr_temp.numpy() > threshold)[0]])
    n_redundant = len(redundant_set)
    return n_redundant

def analyze_dimreduction_methods(latent, comparison_data, redundant=False, device='cpu'):
    if redundant:
        print("Computing redundant features")
        n_redundant = get_number_of_redundant_features(latent, threshold=0.95, device=device)
    else:
        n_redundant = None
    gc.collect()
    torch.cuda.empty_cache()
    corrs = get_correlations_with_data(latent, np.arange(latent.shape[1]), comparison_data, device=device)
    n_per_attribute = get_n_features_per_attribute(corrs)
    highest_corrs = get_highest_corr_per_attribute(corrs)
    return n_redundant, n_per_attribute, highest_corrs

##########################################

#n_dim = rna_counts.shape[1]
n_dim = 150

# run ICA on the data
print("Running ICA...")
from sklearn.decomposition import FastICA
ica = FastICA(n_components=n_dim, random_state=0)
ica.fit(rna_counts)
embed = torch.tensor(ica.transform(rna_counts))

print("Running y")
y_metrics = analyze_dimreduction_methods(embed, rna_counts, redundant=True, device=device)
gc.collect()
torch.cuda.empty_cache()
print("Running x0")
x0_metrics = analyze_dimreduction_methods(embed, x0, redundant=False, device=device)
gc.collect()
torch.cuda.empty_cache()
print("Running x1")
x1_metrics = analyze_dimreduction_methods(embed, x1, redundant=False, device=device)
gc.collect()
torch.cuda.empty_cache()
print("Running x2")
x2_metrics = analyze_dimreduction_methods(embed, x2, redundant=False, device=device)
gc.collect()
torch.cuda.empty_cache()
print("Running ct")
ct_metrics = analyze_dimreduction_methods(embed, ct.float().unsqueeze(1), redundant=False, device=device)
gc.collect()
torch.cuda.empty_cache()
print("Running co")
co_metrics = analyze_dimreduction_methods(embed, co.float().unsqueeze(1), redundant=False, device=device)
gc.collect()
torch.cuda.empty_cache()

# save metrics
df_y = pd.DataFrame({'n_redundant': [y_metrics[0]], 'n_per_attribute (mean)': [y_metrics[1].mean()], 'n_per_attribute (max)': [y_metrics[1].max()], 'highest_corrs (mean)': [y_metrics[2].mean()], 'highest_corrs (max)': [y_metrics[2].max()], 'variable': 'y'})
df_x0 = pd.DataFrame({'n_redundant': [y_metrics[0]], 'n_per_attribute (mean)': [x0_metrics[1].mean()], 'n_per_attribute (max)': [x0_metrics[1].max()], 'highest_corrs (mean)': [x0_metrics[2].mean()], 'highest_corrs (max)': [x0_metrics[2].max()], 'variable': 'x0'})
df_x1 = pd.DataFrame({'n_redundant': [y_metrics[0]], 'n_per_attribute (mean)': [x1_metrics[1].mean()], 'n_per_attribute (max)': [x1_metrics[1].max()], 'highest_corrs (mean)': [x1_metrics[2].mean()], 'highest_corrs (max)': [x1_metrics[2].max()], 'variable': 'x1'})
df_x2 = pd.DataFrame({'n_redundant': [y_metrics[0]], 'n_per_attribute (mean)': [x2_metrics[1].mean()], 'n_per_attribute (max)': [x2_metrics[1].max()], 'highest_corrs (mean)': [x2_metrics[2].mean()], 'highest_corrs (max)': [x2_metrics[2].max()], 'variable': 'x2'})
df_ct = pd.DataFrame({'n_redundant': [y_metrics[0]], 'n_per_attribute (mean)': [ct_metrics[1][0]], 'n_per_attribute (max)': [ct_metrics[1][0]], 'highest_corrs (mean)': [ct_metrics[2][0]], 'highest_corrs (max)': [ct_metrics[2][0]], 'variable': 'ct'})
df_co = pd.DataFrame({'n_redundant': [y_metrics[0]], 'n_per_attribute (mean)': [co_metrics[1][0]], 'n_per_attribute (max)': [co_metrics[1][0]], 'highest_corrs (mean)': [co_metrics[2][0]], 'highest_corrs (max)': [co_metrics[2][0]], 'variable': 'co'})
df_pca = pd.concat([df_y, df_x0, df_x1, df_x2, df_ct, df_co], axis=0)
df_pca.to_csv(f'03_results/reports/files/sim2L_ica_metrics_{n_dim}components.csv', index=False)