import numpy as np
import torch
import pandas as pd
import torch
import random
from tqdm import tqdm

# add src to the system path
import sys
sys.path.append(".")
sys.path.append('src')

from src.functions.sae_training import *
from src.functions.sae_analysis_sim2 import *
from src.models.autoencoder import *
from src.visualization.plotting import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set a random seed
seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

data_dir = '/projects/heads/data/simulation/singlecell/'

############################################

sae_type = 'bricken'

import argparse

parser = argparse.ArgumentParser(description='SAE training')
parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples to train on.')
parser.add_argument('--latent_dim', type=int, default=150, help='Dimension of the latent space.')
parser.add_argument('--model_depth', type=int, default=2, help='Number of layers in the encoder and decoder.')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for the encoder and decoder.')
args = parser.parse_args()

n_samples = args.n_samples
latent_dim = args.latent_dim
model_depth = args.model_depth
dropout_rate = args.dropout_rate
#n_samples = 1000
#latent_dim = 150
#model_depth = 2

############################################

# load data
complexity = 'high'

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
n_samples_validation = int(n_samples*0.1)
rna_counts = rna_counts[:n_samples_validation]
x0 = x0[:n_samples_validation]
x1 = x1[:n_samples_validation]
x2 = x2[:n_samples_validation]
ct = ct[:n_samples_validation]
co = co[:n_samples_validation]

# load encoder
if dropout_rate > 0:
    if n_samples == 10000:
        model_name = 'large_{}-complexity_{}-depth_{}-latent_{}-samples_{}-dropout'.format(complexity, model_depth, latent_dim,  n_samples, dropout_rate)
    else:
        model_name = 'large_{}-complexity_{}-depth_{}-latent_{}-dropout_{}-samples'.format(complexity, model_depth, latent_dim, dropout_rate, n_samples)
else:
    model_name = 'large_{}-complexity_{}-depth_{}-latent_{}-samples'.format(complexity, model_depth, latent_dim, n_samples)

encoder = torch.load('03_results/models/sim2_{}_encoder.pth'.format(model_name)).cpu()

# get representations
reps = encoder(rna_counts).detach()
del encoder
input_size = latent_dim
batch_size = 128
train_data = reps.clone().to(device)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

############################################

sae_metrics_dict = {
    'hidden_factor': [],
    'lr': [],
    'l1_weight': [],
    'loss': [],
    'n_activation_features': [],
    'avg_active_hidden_units': [],
    'n_unique': [],
    'n_redundant': [],
    #'n_per_y': [],
    #'highest_corr_y': [],
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

#hidden_factor_options = [2, 5, 10, 20, 50, 100, 200, 1000]
hidden_factor_options = [5, 10, 20, 50, 100, 200]
#lr_options = [1e-2, 1e-3, 1e-4, 1e-5]
lr_options = [1e-3, 1e-4, 1e-5]
#l1_weight_options = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
l1_weight_options = [1e-2, 1e-3, 1e-4, 1e-5]

# add the progress bar to the outer loop
for hidden_factor in tqdm(hidden_factor_options):
    for lr in lr_options:
        for l1_weight in l1_weight_options:
            sae_setup = [hidden_factor, lr, l1_weight]
            sae_model, loss = train_sae(train_loader, input_size=input_size, hidden_factor=hidden_factor, lr=lr, l1_weight=l1_weight, n_epochs=500, sae_type=sae_type)
            sae_setup.append(loss)

            reps_reconstructed, activations = sae_model(reps.to(device))
            del reps_reconstructed

            """
            y_metrics = run_sae_analysis(sae_model, train_loader, activations, rna_counts)
            sae_setup.append(y_metrics[0])
            sae_setup.append(y_metrics[1])
            sae_setup.append(y_metrics[2])
            sae_setup.append(y_metrics[3])
            sae_setup.append(np.mean(y_metrics[4]))
            sae_setup.append(np.mean(y_metrics[5]))
            del y_metrics
            """
            x0_metrics = run_sae_analysis(sae_model, train_loader, activations, x0)
            sae_setup.append(x0_metrics[0])
            sae_setup.append(x0_metrics[1])
            sae_setup.append(x0_metrics[2])
            sae_setup.append(x0_metrics[3])
            sae_setup.append(x0_metrics[4])
            sae_setup.append(x0_metrics[5])
            del x0_metrics
            x1_metrics = run_sae_analysis(sae_model, train_loader, activations, x1, get_all=False)
            sae_setup.append(x1_metrics[4])
            sae_setup.append(x1_metrics[5])
            del x1_metrics
            x2_metrics = run_sae_analysis(sae_model, train_loader, activations, x2, get_all=False)
            sae_setup.append(x2_metrics[4])
            sae_setup.append(x2_metrics[5])
            del x2_metrics
            ct_metrics = run_sae_analysis(sae_model, train_loader, activations, ct.unsqueeze(1), get_all=False)
            sae_setup.append(ct_metrics[4])
            sae_setup.append(ct_metrics[5])
            del ct_metrics
            co_metrics = run_sae_analysis(sae_model, train_loader, activations, co.unsqueeze(1), get_all=False)
            sae_setup.append(co_metrics[4])
            sae_setup.append(co_metrics[5])
            del co_metrics

            del sae_model, activations

# make the dict into a pandas dataframe and save it
df_sae_metrics = pd.DataFrame(sae_metrics_dict)
df_sae_metrics.to_csv('03_results/reports/files/sim2L_'+model_name+'_'+sae_type+'_metrics.csv', index=False)