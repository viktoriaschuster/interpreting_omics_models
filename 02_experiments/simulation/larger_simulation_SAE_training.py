import numpy as np
import torch
import pandas as pd
import random
import gc
import os

# add src to the system path
import sys
sys.path.append(".")
sys.path.append('src')

from src.models.sparse_autoencoder import *
from src.models.autoencoder import *
from src.functions.sae_training import *
from src.visualization.plotting import *

###
# argparse
###

import argparse

parser = argparse.ArgumentParser(description='Train an autoencoder on the simulated data.')
parser.add_argument('--latent_dim', type=int, default=150, help='Dimension of the latent space.')
parser.add_argument('--model_depth', type=int, default=2, help='Number of layers in the encoder and decoder.')
parser.add_argument('--width', type=str, default='narrow', help='AE can be wide or narrow')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use.')
args = parser.parse_args()

latent_dim = args.latent_dim
model_depth = args.model_depth
model_width = args.width
seed = args.seed
complexity = 'high'
n_samples = 100000
#n_samples = 1000

################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    # specify the gpu id
    torch.cuda.set_device(args.gpu)
    print('Using GPU', torch.cuda.current_device())

# set a random seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

################################

# load the data

data_dir = '/home/vschuste/data/simulation/'

for d_seed in range(10):
    if d_seed == 0:
        rna_counts = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, d_seed), weights_only=False)
    else:
        temp = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, d_seed), weights_only=False)
        rna_counts = torch.cat((rna_counts, temp), dim=0)
# now subsample the data if necessary
if n_samples < rna_counts.shape[0]:
    rna_counts = rna_counts[:n_samples]
rna_counts = rna_counts.to(device)
n_samples_validation = int(n_samples*0.1)
n_samples_train = n_samples - n_samples_validation
print('Data loaded.')
print('Data shape:', rna_counts.shape)

hyperparams = {
    '20-2-narrow': {'dropout': 0.0, 'learning_rate': 1e-4, 'weight_decay': 0.001, 'batch_size': 128}, # in this style
    '20-2-wide': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 0.001, 'batch_size': 128},
    '20-4-narrow': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 0.0, 'batch_size': 128},
    '20-4-wide': {'dropout': 0.0, 'learning_rate': 1e-6, 'weight_decay': 1e-5, 'batch_size': 128},
    '20-6-narrow': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 1e-5, 'batch_size': 128},
    '20-6-wide': {'dropout': 0.0, 'learning_rate': 1e-6, 'weight_decay': 1e-5, 'batch_size': 512},
    '100-2-narrow': {'dropout': 0.0, 'learning_rate': 1e-4, 'weight_decay': 1e-5, 'batch_size': 128},
    '100-2-wide': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 1e-5, 'batch_size': 128},
    '100-4-narrow': {'dropout': 0.0, 'learning_rate': 1e-4, 'weight_decay': 1e-7, 'batch_size': 128},
    '100-4-wide': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 1e-5, 'batch_size': 128},
    '150-2-narrow': {'dropout': 0.0, 'learning_rate': 1e-4, 'weight_decay': 1e-5, 'batch_size': 128},
    '150-2-wide': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 1e-5, 'batch_size': 128},
    '150-4-narrow': {'dropout': 0.0, 'learning_rate': 1e-4, 'weight_decay': 1e-7, 'batch_size': 128},
    '150-4-wide': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 1e-5, 'batch_size': 128},
    '150-6-narrow': {'dropout': 0.0, 'learning_rate': 1e-4, 'weight_decay': 1e-5, 'batch_size': 512},
    '150-6-wide': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 0.1, 'batch_size': 256},
    '1000-2-narrow': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 1e-5, 'batch_size': 128},
    '1000-2-wide': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 1e-5, 'batch_size': 128},
    '1000-4-narrow': {'dropout': 0.0, 'learning_rate': 1e-4, 'weight_decay': 0.0, 'batch_size': 128},
    '1000-4-wide': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 1e-7, 'batch_size': 128},
    '1000-6-narrow': {'dropout': 0.0, 'learning_rate': 1e-6, 'weight_decay': 0.0, 'batch_size': 512},
    '1000-6-wide': {'dropout': 0.0, 'learning_rate': 1e-5, 'weight_decay': 1e-7, 'batch_size': 512},
}
try:
    hyperparams = hyperparams[f"{latent_dim}-{model_depth}-{model_width}"]
except ValueError:
    print("Hyperparameters not found, using default.")
    hyperparams = hyperparams['20-2-narrow']
hyperparams['latent_dim'] = latent_dim
hyperparams['model_depth'] = model_depth
hyperparams['model_width'] = model_width
hyperparams['seed'] = seed

################################

run_name = 'largesim_ae_latent-{}_depth-{}_width-{}_seed-{}'.format(latent_dim, model_depth, model_width, seed)

# load encoder and decoder
try:
    encoder = torch.load('03_results/models/'+run_name+'/encoder.pth').to(device)
    #decoder = torch.load('03_results/models/'+run_name+'/decoder.pth').to(device)
except:
    # if the file was not found, look for a directory that contains the run_name
    import os
    run_name = [f for f in os.listdir('03_results/models/') if run_name in f][0]
    encoder = torch.load('03_results/models/'+run_name+'/encoder.pth').to(device)
    #decoder = torch.load('03_results/models/'+run_name+'/decoder.pth').to(device)
    print('Loaded model from directory ', run_name)

################################
# create data loaders for SAE
################################

# get the latent space representation
reps = encoder(rna_counts).detach()
del encoder
input_size = latent_dim
batch_size = 128
train_data = reps[:n_samples_train].clone().to(device)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = reps[n_samples_train:].clone().to(device)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

################################
# train the SAEs
################################

# make a subdirectory for the saes
os.makedirs('03_results/models/'+run_name+'/sae', exist_ok=True)

scaling_factors = [20, 100, 200, 500]
l1_weights = [0.1, 1e-2, 1e-3, 1e-4]
if latent_dim == 20:
    l1_weights = [0.1, 1e-2, 1e-3, 1e-4]
    lrs = [1e-4, 1e-5, 1e-6]
else:
    l1_weights = [1e-2, 1e-3]
    lrs = [1e-5]
sae_type = 'bricken'

#import wandb

for scaling_factor in scaling_factors:
    for l1_weight in l1_weights:
        for lr in lrs:
            print('Training SAE with scaling factor {}, l1 weight {}, and lr {}'.format(scaling_factor, l1_weight, lr))
            #wandb.init(project="sc_simulation", entity="vschuster-broad-institute", config=hyperparams)
            #wandb.run.name = 'sae-' + str(scaling_factor) + 'x-l1w-' + str(l1_weight) + '_' + run_name

            #hidden_factor = scaling_factor*latent_dim
            hidden_factor = scaling_factor

            sae_model, losses, val_losses = train_sae(
                train_loader, 
                val_loader=val_loader,
                input_size=input_size, 
                hidden_factor=hidden_factor, 
                lr=lr, 
                l1_weight=l1_weight, 
                n_epochs=1000, 
                sae_type=sae_type,
                early_stopping=20,
                return_all_losses=True
            )

            # save model and history
            torch.save(sae_model, '03_results/models/'+run_name+'/sae/sae_{}x_l1w{}_lr{}.pth'.format(scaling_factor, l1_weight, lr))
            history = pd.DataFrame({'loss': losses, 'val_loss': val_losses})
            history.to_csv('03_results/models/'+run_name+'/sae/losses_{}x_l1w{}_lr{}.csv'.format(scaling_factor, l1_weight, lr))

            del sae_model
            del history
            torch.cuda.empty_cache()
            gc.collect()