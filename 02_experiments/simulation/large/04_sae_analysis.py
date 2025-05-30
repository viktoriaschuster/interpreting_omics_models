import os
import torch
import pandas as pd
import numpy as np
import random
import gc
import tqdm

import sys
sys.path.append(".")
sys.path.append('src')

from src.functions.sae_analysis_sim3 import *

results_dir = '03_results/models/'
latent_dims = [20, 100, 150]
depth = 2
width = 'wide'
batch_size = 128

dev_id = 0
device = torch.device(f'cuda:{dev_id}' if torch.cuda.is_available() else 'cpu')

# set a random seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

############################################
# load data
############################################

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
rna_counts = rna_counts[::20]
x0 = x0[::20]
x1 = x1[::20]
x2 = x2[::20]
ct = ct[::20]
co = co[::20]

print("Data loaded.")
print(f"Running on a subset with {rna_counts.shape[0]} samples.")

############################################
# prepare output
############################################

sae_metrics_dict_template = {
    'latent_dim': [],
    'hidden_factor': [],
    'lr': [],
    'l1_weight': [],
    'n_activation_features': [],
    'avg_active_hidden_units': [],
    'n_unique': [],
    'n_redundant': [],
    'n_per_y': [],
    'highest_corr_y': [],
    'n_per_x0': [],
    'highest_corr_x0': [],
    'n_per_x1': [],
    'highest_corr_x1': [],
    'n_per_x2': [],
    'highest_corr_x2': [],
    'n_per_ct': [],
    'highest_corr_ct': [],
    'n_per_co': [],
    'highest_corr_co': []
}

############################################
# run analysis
############################################

seed = 0
for latent_dim in latent_dims:
    sae_metrics_dict = sae_metrics_dict_template.copy()
    print(f"Running analysis for latent dimension {latent_dim}.")
    ae_dir = results_dir + 'largesim_ae_latent-' + str(latent_dim) + '_depth-' + str(depth) + '_width-' + width + '_seed-' + str(seed) + '/'
    file_dir = ae_dir + 'sae/'
    # get all subdirectories starting with 'largesim_ae'
    filenames = [x for x in os.listdir(file_dir) if os.path.isfile(file_dir + x) and x.startswith('sae')]
    # only keep the ones that are for lr 1e-6
    filenames = [x for x in filenames if 'lr1e-05' in x]
    filenames = [x for x in filenames if 'l1w0.001' in x]
    print(f"Found {len(filenames)} models.")
    #filenames = ['sae_20x_l1w0.001_lr1e-05.pth']

    # load the encoder
    encoder = torch.load(ae_dir +  'encoder.pth', weights_only=False).to(device)
    encoder.eval()
    # get embeddings
    reps = encoder(rna_counts.to(device)).detach()
    # prep the data loader
    input_size = latent_dim
    train_loader = torch.utils.data.DataLoader(reps.to(device), batch_size=batch_size, shuffle=True)

    for filename in filenames:
        modelname = filename.split('sae_')[1].split('.pth')[0]
        scaling_factor = modelname.split('x_')[0]
        l1_weight = modelname.split('_l1w')[1].split('_')[0]
        lr = modelname.split('_lr')[1]

        sae_setup = [latent_dim, scaling_factor, lr, l1_weight]
        sae_model = torch.load(file_dir + filename, weights_only=False).to(device)

        # get the activations in chunks
        print("Getting activations.")
        chunksize = 1000
        activations = []
        for i in range(0, reps.shape[0], chunksize):
            reps_reconstructed, activations_chunk = sae_model(reps[i:i+chunksize].to(device))
            activations.append(activations_chunk)
            del reps_reconstructed
        activations = torch.cat(activations, dim=0)
        print(f"Got activations of shape {activations.shape}")

        #"""
        print("Running y")
        y_metrics = run_sae_analysis(sae_model, train_loader, activations, rna_counts, get_all=True)
        sae_setup.append(y_metrics[0])
        sae_setup.append(y_metrics[1])
        sae_setup.append(y_metrics[2])
        sae_setup.append(y_metrics[3])
        sae_setup.append(np.mean(y_metrics[4]))
        sae_setup.append(np.mean(y_metrics[5]))
        del y_metrics
        #"""
        print("Running x0")
        #x0_metrics = run_sae_analysis(sae_model, train_loader, activations, x0)
        #sae_setup.append(x0_metrics[0])
        #sae_setup.append(x0_metrics[1])
        #sae_setup.append(x0_metrics[2])
        #sae_setup.append(x0_metrics[3])
        x0_metrics = run_sae_analysis(sae_model, train_loader, activations, x0, get_all=False)
        sae_setup.append(x0_metrics[4])
        sae_setup.append(x0_metrics[5])
        del x0_metrics
        print("Running x1")
        x1_metrics = run_sae_analysis(sae_model, train_loader, activations, x1, get_all=False)
        sae_setup.append(x1_metrics[4])
        sae_setup.append(x1_metrics[5])
        del x1_metrics
        print("Running x2")
        x2_metrics = run_sae_analysis(sae_model, train_loader, activations, x2, get_all=False)
        sae_setup.append(x2_metrics[4])
        sae_setup.append(x2_metrics[5])
        del x2_metrics
        print("Running ct")
        ct_metrics = run_sae_analysis(sae_model, train_loader, activations, ct.unsqueeze(1), get_all=False)
        sae_setup.append(ct_metrics[4])
        sae_setup.append(ct_metrics[5])
        del ct_metrics
        print("Running co")
        co_metrics = run_sae_analysis(sae_model, train_loader, activations, co.unsqueeze(1), get_all=False)
        sae_setup.append(co_metrics[4])
        sae_setup.append(co_metrics[5])
        del co_metrics

        del sae_model, activations

        for i, key in enumerate(sae_metrics_dict.keys()):
            sae_metrics_dict[key].append(sae_setup[i])

        # free um memory
        gc.collect()
        torch.cuda.empty_cache()

    # make the dict into a pandas dataframe and save it
    df_sae_metrics = pd.DataFrame(sae_metrics_dict)
    df_sae_metrics.to_csv('03_results/reports/files/sim2L_sae_metrics_pearson_latent-{}.csv'.format(latent_dim), index=False)
    print(f"Saved metrics for latent dimension {latent_dim}.")