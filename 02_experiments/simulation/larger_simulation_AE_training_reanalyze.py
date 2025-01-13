import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
parser.add_argument('--latent_dim', type=int, default=150, help='Dimension of the latent space.')
parser.add_argument('--model_depth', type=int, default=2, help='Number of layers in the encoder and decoder.')
parser.add_argument('--width', type=str, default='narrow', help='AE can be wide or narrow')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use.')
#parser.add_argument('--run_name', type=str, default=None, help='Name of the run.')
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
    decoder = torch.load('03_results/models/'+run_name+'/decoder.pth').to(device)
except:
    # if the file was not found, look for a directory that contains the run_name
    import os
    run_name = [f for f in os.listdir('03_results/models/') if run_name in f][0]
    encoder = torch.load('03_results/models/'+run_name+'/encoder.pth').to(device)
    decoder = torch.load('03_results/models/'+run_name+'/decoder.pth').to(device)
    print('Loaded model from directory ', run_name)

################################
# perform the evaluation
################################

# get the latent space representation
reps = encoder(rna_counts).detach()

# load all the generative hidden variables
complexity = 'high'

for d_seed in range(10):
    temp_x0 = torch.load(data_dir+'large_{}-complexity_rs{}_x0.pt'.format(complexity, d_seed), weights_only=False)
    temp_x1 = torch.load(data_dir+'large_{}-complexity_rs{}_x1.pt'.format(complexity, d_seed), weights_only=False)
    temp_x2 = torch.load(data_dir+'large_{}-complexity_rs{}_x2.pt'.format(complexity, d_seed), weights_only=False)
    temp_ct = torch.load(data_dir+'large_{}-complexity_rs{}_ct.pt'.format(complexity, d_seed), weights_only=False)
    temp_cov = torch.load(data_dir+'large_{}-complexity_rs{}_co.pt'.format(complexity, d_seed), weights_only=False)
    if d_seed == 0:
        x0 = temp_x0
        x1 = temp_x1
        x2 = temp_x2
        ct = temp_ct
        co = temp_cov
    else:
        x0 = torch.cat((x0, temp_x0), dim=0)
        x1 = torch.cat((x1, temp_x1), dim=0)
        x2 = torch.cat((x2, temp_x2), dim=0)
        ct = torch.cat((ct, temp_ct), dim=0)
        co = torch.cat((co, temp_cov), dim=0)
x0 = x0[:n_samples]
x1 = x1[:n_samples]
x2 = x2[:n_samples]
ct = ct[:n_samples]
co = co[:n_samples]

all_values = torch.cat((rna_counts.cpu(), x0, x1, x2, ct.unsqueeze(1), co.unsqueeze(1)), dim=1).detach()
value_columns = ['y_{}'.format(i) for i in range(rna_counts.shape[1])] + ['x0_{}'.format(i) for i in range(x0.shape[1])] + ['x1_{}'.format(i) for i in range(x1.shape[1])] + ['x2_{}'.format(i) for i in range(x2.shape[1])] + ['ct'] + ['co']

print('Collected all data variables: ', all_values.shape)

assert all_values.shape[1] == len(value_columns)

print('Now performing parallel linear regression btw latent representation and all variables.')

def parallel_linear_regression(x, y, n_samples, n_samples_train, n_epochs=100, early_stopping=10):
    import tqdm
    y_mean = y[n_samples_train:n_samples].mean(dim=0)

    # loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[:n_samples_train], y[:n_samples_train]), batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[n_samples_train:n_samples], y[n_samples_train:n_samples]), batch_size=128, shuffle=False)

    # set up a linear layer to use for parallel regression
    linear = nn.Linear(x.shape[1], y.shape[1]).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.0001, weight_decay=0)
    loss_fn = nn.MSELoss()

    # train the linear layer
    val_losses = []
    pbar = tqdm.tqdm(range(n_epochs))
    for epoch in pbar:
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = linear(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        val_loss = 0
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            y_pred = linear(x_val)
            val_loss += loss_fn(y_pred, y_val).item()
        val_losses.append(val_loss / len(val_loader))
        if epoch > early_stopping and min(val_losses[-early_stopping:]) > min(val_losses):
            print("Early stopping in linear regression at epoch ", epoch)
            break
        pbar.set_postfix({'val loss': round(val_loss / len(val_loader), 4)})
    
    y_pred = linear(x[n_samples_train:n_samples]).cpu()
    y_pred = y_pred.detach()
    r_squares = 1 - (((y[n_samples_train:n_samples] - y_pred)**2).sum(0) / ((y[n_samples_train:n_samples] - y_mean)**2).sum(0))
    return r_squares

r_squares = parallel_linear_regression(reps, all_values, n_samples, n_samples_train, n_epochs=100, early_stopping=10)
# I also want to know what the number of features encoded in the latent space is. Since I don't know the best threshold, I will use the auc

print('Computing summary statistics.')

thresholds = np.linspace(0, 1, 100)
n_features = []
for threshold in thresholds:
    n_features.append(sum(r_squares >= threshold))
n_features = np.array(n_features)

# log the r_squares values as summary metrics
start_idx = 0
stop_idx = rna_counts.shape[1]
r_square_y = r_squares[start_idx:stop_idx]
# remove infinity values with nan
r_square_y = r_square_y[~torch.isinf(r_square_y)].flatten()
# remove nan values
r_square_y = r_square_y[~torch.isnan(r_square_y)]
r_square_y = r_square_y.mean().item()
start_idx = stop_idx
stop_idx += x0.shape[1]
r_square_x0 = r_squares[start_idx:stop_idx]#.mean().item()
r_square_x0 = r_square_x0[~torch.isinf(r_square_x0)].flatten()
r_square_x0 = r_square_x0[~torch.isnan(r_square_x0)]
r_square_x0 = r_square_x0.mean().item()
start_idx = stop_idx
stop_idx += x1.shape[1]
r_square_x1 = r_squares[start_idx:stop_idx]#.mean().item()
r_square_x1 = r_square_x1[~torch.isinf(r_square_x1)].flatten()
r_square_x1 = r_square_x1[~torch.isnan(r_square_x1)]
r_square_x1 = r_square_x1.mean().item()
start_idx = stop_idx
stop_idx += x2.shape[1]
r_square_x2 = r_squares[start_idx:stop_idx]#.mean().item()
r_square_x2 = r_square_x2[~torch.isinf(r_square_x2)].flatten()
r_square_x2 = r_square_x2[~torch.isnan(r_square_x2)]
r_square_x2 = r_square_x2.mean().item()
r_square_ct = r_squares[stop_idx].item()
r_square_co = r_squares[stop_idx+1].item()
r_squares = {
    'mean r_square_y': r_square_y,
    'mean r_square_x0': r_square_x0,
    'mean r_square_x1': r_square_x1,
    'mean r_square_x2': r_square_x2,
    'r_square_ct': r_square_ct,
    'r_square_co': r_square_co
}
print(r_squares)
# save the r_squares values
r_squares = pd.Series(r_squares)
r_squares.to_csv('03_results/models/'+run_name+'/r_squares.csv')
# save the number of features
df_n_features = pd.DataFrame({'threshold': thresholds, 'n_features': n_features})
df_n_features.to_csv('03_results/models/'+run_name+'/n_features.csv')
print("done")