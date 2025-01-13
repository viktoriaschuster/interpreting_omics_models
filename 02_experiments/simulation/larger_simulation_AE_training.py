import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import random

# add src to the system path
import sys
sys.path.append(".")
sys.path.append('src')

from src.models.sparse_autoencoder import *
from src.models.autoencoder import *
from src.functions.ae_training import *
from src.visualization.plotting import *

###
# argparse
###

import argparse

parser = argparse.ArgumentParser(description='Train an autoencoder on the simulated data.')
parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples to train on.')
parser.add_argument('--latent_dim', type=int, default=150, help='Dimension of the latent space.')
parser.add_argument('--model_depth', type=int, default=2, help='Number of layers in the encoder and decoder.')
parser.add_argument('--complexity', type=str, default='high', help='Complexity of the data.')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for the encoder and decoder.')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use.')
args = parser.parse_args()

n_samples = args.n_samples
latent_dim = args.latent_dim
model_depth = args.model_depth
complexity = args.complexity
dropout_rate = args.dropout_rate

################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    # specify the gpu id
    torch.cuda.set_device(args.gpu)
    print('Using GPU', torch.cuda.current_device())

# set a random seed
seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

################################

# load the data

data_dir = '/projects/heads/data/simulation/singlecell/'

for seed in range(10):
    if seed == 0:
        rna_counts = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed), weights_only=False)
    else:
        temp = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed), weights_only=False)
        rna_counts = torch.cat((rna_counts, temp), dim=0)
# now subsample the data if necessary
if n_samples < rna_counts.shape[0]:
    rna_counts = rna_counts[:n_samples]
rna_counts = rna_counts.to(device)
n_samples_validation = int(n_samples*0.1)
print('Data loaded.')
print('Data shape:', rna_counts.shape)

################################

encoder = Encoder(20000, latent_dim, model_depth, dropout=dropout_rate).to(device)
decoder = Decoder(latent_dim, 20000, model_depth, dropout=dropout_rate).to(device)

encoder, decoder, history = train_and_eval_model(encoder, decoder, rna_counts, n_samples, n_samples_validation, n_epochs=10000)

# save this model as the best one
if dropout_rate > 0:
    model_name = 'large_{}-complexity_{}-depth_{}-latent_{}-dropout_{}-samples'.format(complexity, model_depth, latent_dim, dropout_rate, n_samples)
else:
    model_name = 'large_{}-complexity_{}-depth_{}-latent_{}-samples'.format(complexity, model_depth, latent_dim, n_samples)

torch.save(encoder, '03_results/models/sim2_'+model_name+'_encoder.pth')
torch.save(decoder, '03_results/models/sim2_'+model_name+'_decoder.pth')

# also save the history of the training
history.to_csv('03_results/models/sim2_'+model_name+'_history.csv', index=False)