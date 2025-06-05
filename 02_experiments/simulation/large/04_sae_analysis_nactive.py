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
#latent_dims = [20, 100, 150]
latent_dims = [150]
depth = 2
width = 'wide'
batch_size = 128

#dev_id = 2
#device = torch.device(f'cuda:{dev_id}' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

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
    'n_hidden': [],
    'reconstruction loss': [],
    'number of active neurons': []
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
    #filenames = [x for x in filenames if 'lr1e-05' in x]
    #filenames = [x for x in filenames if 'l1w0.001' in x]
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
        if int(sae_model.encoder_linear.weight.shape[0]) > (int(latent_dim) * int(scaling_factor)):
            print(f"Warning: encoder weight shape {sae_model.encoder_linear.weight.shape[0]} is larger than latent_dim * scaling_factor {int(latent_dim) * int(scaling_factor)}.")
            continue

        # get the activations in chunks
        print("Getting activations.")
        chunksize = 100
        activations = []
        recon_loss = 0
        for i in range(0, reps.shape[0], chunksize):
            reps_reconstructed, activations_chunk = sae_model(reps[i:i+chunksize].to(device))
            activations.append(activations_chunk)
            recon_loss += torch.nn.functional.mse_loss(reps_reconstructed, reps[i:i+chunksize].to(device), reduction='sum').item()
            del reps_reconstructed
        activations = torch.cat(activations, dim=0).detach().cpu()
        recon_loss /= reps.shape[0]
        print(f"Got activations of shape {activations.shape}")
        sae_setup.append(activations.shape[1])
        sae_setup.append(recon_loss)

        #"""
        print("Running y")
        unique_active_indices = set()  # Use a set to store unique active neuron indices
        # Find the indices of the active neurons (where activation > threshold)
        active_indices_batch = (activations > 1e-5).nonzero(as_tuple=False)  # Get nonzero indices
        # This returns indices in the form [batch_idx, neuron_idx]
        # We are only interested in neuron_idx for overall unique active units
        unique_active_indices.update(active_indices_batch[:, 1].tolist())
        sae_setup.append(len(unique_active_indices))

        del sae_model, activations

        for i, key in enumerate(sae_metrics_dict.keys()):
            sae_metrics_dict[key].append(sae_setup[i])

        # free um memory
        gc.collect()
        torch.cuda.empty_cache()

    # make the dict into a pandas dataframe and save it
    df_sae_metrics = pd.DataFrame(sae_metrics_dict)
    df_sae_metrics.to_csv('03_results/reports/files/sim2L_sae_metrics_pearson_latent-{}_nactive.csv'.format(latent_dim), index=False)
    print(f"Saved metrics for latent dimension {latent_dim}.")