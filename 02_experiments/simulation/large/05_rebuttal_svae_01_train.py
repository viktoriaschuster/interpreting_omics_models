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

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

# add keyword arguments for the type of prior and the hidden dimensions
import argparse
parser = argparse.ArgumentParser(description='Train a sparse VAE on simulated data.')
parser.add_argument('--prior', type=str, default='gaussian', help='Type of prior to use (gaussian, laplace, cauchy).')
#parser.add_argument('--latent_scaling_factor', type=int, default=1, help='Scaling factor for latent dim 150.') # options are for comparison to SAE: [1, 20, 100, 200, 500] * 150
args = parser.parse_args()
#scaling_factor = args.latent_scaling_factor

complexity = 'high'
n_samples = 100000
data_dir = '/home/vschuste/data/simulation/'

for seed in range(10):
    temp_y = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed), weights_only=False)
    if seed == 0:
        rna_counts = temp_y
    else:
        rna_counts = torch.cat((rna_counts, temp_y), dim=0)
# limit to the training data
n_samples_train = int(n_samples*0.9)
rna_counts = rna_counts[:n_samples_train]
# also make this faster by taking every 10th sample
rna_counts = rna_counts[::3]

print("Data loaded.")
print(f"Running on a subset with {rna_counts.shape[0]} samples.")

############################

# decide on a few betas to run with
beta_options = [1.0, 0.1]
scaling_factor_options = [1, 20, 100, 200]

from src.models.sparse_vae import *

# Create a VAE with Laplace prior
input_dim = rna_counts.shape[1]
from torch.utils.data import DataLoader, TensorDataset
# Assuming rna_counts is a PyTorch tensor of shape (n_samples, n_features)
batch_size = 512
n_train = int(rna_counts.shape[0] * 0.9)
data_loader = DataLoader(
    TensorDataset(rna_counts[:n_train]),
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(rna_counts[n_train:]),
    batch_size=batch_size,
    shuffle=True
)
# set a random seed
torch.manual_seed(0)

for beta in beta_options:
    for scaling_factor in scaling_factor_options:
        print(f"Training with beta: {beta}, scaling factor: {scaling_factor}")
        latent_dim = int(scaling_factor * 150)
        svae = PriorVAE(
            input_dim=input_dim,
            hidden_dim=int(abs(input_dim - latent_dim) / 2),
            latent_dim=latent_dim,
            prior_type=args.prior,
            beta=beta,
        )
        # write a dataloader for the rna_counts data

        svae.to(device)
        # Set up optimizer
        optimizer = torch.optim.Adam(svae.parameters(), lr=1e-4)

        svae, losses = train_vae(svae, optimizer, data_loader, val_loader, epochs=1000, device=device, early_stopping=50)
        df_losses = pd.DataFrame(losses)
        # save the model and losses
        model_name = f"svae_b{beta}_h{latent_dim}_{args.prior}.pt"
        torch.save(svae.state_dict(), '03_results/models/comparison/{}'.format(model_name))
        df_losses.to_csv('03_results/models/comparison/{}_losses.csv'.format(model_name), index=False)

        # delete everything to save memory
        del svae
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()