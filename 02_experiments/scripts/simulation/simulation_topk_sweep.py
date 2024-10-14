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
from src.visualization.plotting import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set a random seed
seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

############################################

sae_type = 'topk'

############################################

import torch
import torch.nn as nn
import math
from typing import Callable
from scipy.stats import spearmanr
import torch.nn.functional as F

class BiasLayer(torch.nn.Module):
    def __init__(self, input_size, init='standard') -> None:
        super().__init__()
        if init == 'standard':
            stdv = 1. / math.sqrt(input_size)
            # init bias to a uniform distribution
            bias_value = torch.empty((input_size)).uniform_(-stdv, stdv)
        else:
            bias_value = torch.zeros(input_size)

        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return x + self.bias_layer

# Sparse Autoencoder Model with Mechanistic Interpretability
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, input_bias: bool = False, activation: Callable = nn.ReLU(), bias_type='standard', tied: bool = False):
        super(SparseAutoencoder, self).__init__()
        # Encoder
        self.pre_bias = BiasLayer(input_size, init=bias_type)
        self.latent_bias = BiasLayer(hidden_size, init=bias_type)
        self.encoder_linear = nn.Linear(input_size, hidden_size, bias=False)
        self.activation = activation
        if input_bias:
            self.encoder = nn.Sequential(
                self.pre_bias,
                self.encoder_linear,
                self.latent_bias,
                self.activation
            )
        else:
            self.encoder = nn.Sequential(
                self.encoder_linear,
                self.latent_bias,
                self.activation
            )
        #decoder
        if tied:
            self.decoder: nn.Linear | TiedTranspose = TiedTranspose(self.encoder)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, input_size, bias=False)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, input_size, bias=False),
                BiasLayer(input_size, init=bias_type)
            )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update({prefix + "k": self.k, prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)


ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}

def train_sae(train_loader, input_size, hidden_factor, lr, l1_weight=1e-3, sparsity_penalty=0, n_epochs=500, sae_type='vanilla', k_latent_percent=100):
    # Initialize model
    hidden_size = input_size * hidden_factor
    k_latent = max(1, int(hidden_size * k_latent_percent / 100)) # make sure there is at least 1 latent feature
    if sae_type == 'vanilla':
        sae_model = SparseAutoencoder(input_size, hidden_size)
    elif sae_type == 'bricken':
        sae_model = SparseAutoencoder(input_size, hidden_size, input_bias=True, bias_type='zero')
    elif sae_type == 'topk':
        sae_model = SparseAutoencoder(input_size, hidden_size, input_bias=True, bias_type='zero', activation=TopK(k_latent))
    else:
        raise ValueError('Invalid SAE type')
    sae_model = sae_model.to(device)

    # Optimizer
    optimizer = optim.Adam(sae_model.parameters(), lr=lr)

    # Training loop
    losses = []
    for epoch in range(n_epochs):
        total_loss = 0
        for x in train_loader:
            # Get inputs and convert to torch Variable
            inputs = x
            inputs = inputs.to(device)
            
            # Forward pass
            outputs, encoded = sae_model(inputs)
            
            # Compute loss
            if sae_type != 'vanilla':
                loss = loss_function(
                    outputs, 
                    inputs, 
                    encoded, 
                    sae_model.encoder[1].weight,  # Pass the encoder's weights for weight sparsity
                    sparsity_penalty, 
                    l1_weight
                )
            else:
                loss = loss_function(
                    outputs, 
                    inputs, 
                    encoded, 
                    sae_model.encoder[0].weight,  # Pass the encoder's weights for weight sparsity
                    sparsity_penalty, 
                    l1_weight
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        losses.append(total_loss / len(train_loader))
    
    return sae_model, losses[-1]

############################################

def get_n_features(activations, threshold=1e-10):
    n_activation_features = activations.shape[1]
    unique_active_unit_indices = get_unique_active_unit_indices(sae_model, train_loader, threshold=threshold)
    avg_active_hidden_units = count_active_hidden_units(sae_model, train_loader, threshold=threshold, avg=True)
    return n_activation_features, unique_active_unit_indices, avg_active_hidden_units

def get_number_of_redundant_features(activations, unique_activs, threshold=0.95):
    # compute correlations between all active features
    redundant_set = set()
    for i in range(len(unique_activs)):
        for j in range(i+1, len(unique_activs)):
            corr = np.corrcoef(activations[:, unique_activs[i]].cpu().detach().numpy(), activations[:, unique_activs[j]].cpu().detach().numpy())[0, 1]
            if corr > threshold:
                # add the feature to the redundant set
                redundant_set.add(unique_activs[j])
    n_redundant = len(redundant_set)
    return n_redundant

def get_correlations_with_data(activations, unique_activs, comparison_data):
    correlations_p = np.zeros((len(unique_activs), comparison_data.shape[1]))
    correlations_s = np.zeros((len(unique_activs), comparison_data.shape[1]))
    for i, feat in enumerate(unique_activs):
        # get the activations
        feat_activation = activations[:, feat]
        for j in range(comparison_data.shape[1]):
            corr = np.corrcoef(feat_activation.cpu().detach().numpy(), comparison_data[:, j])[0, 1]
            correlations_p[i, j] = corr
            # now spearman
            corr, _ = spearmanr(feat_activation.cpu().detach().numpy(), comparison_data[:, j])
            correlations_s[i, j] = corr
    return correlations_p, correlations_s

def get_n_features_per_attribute(correlations, threshold=0.95):
    n_per_attribute = np.zeros(correlations.shape[1])
    for i in range(correlations.shape[1]):
        n_per_attribute[i] = np.sum(np.abs(correlations[:, i]) > threshold)
    return n_per_attribute

def get_highest_corr_per_attribute(correlations):
    best_corrs = np.zeros(correlations.shape[1])
    for i in range(correlations.shape[1]):
        best_id = np.argmax(np.abs(correlations[:, i]))
        best_corrs[i] = correlations[best_id, i]
    return best_corrs

def run_sae_analysis(activations, comparison_data):
    n_activation_features, unique_active_unit_indices, avg_active_hidden_units = get_n_features(activations)
    n_unique = len(unique_active_unit_indices)
    n_redundant = get_number_of_redundant_features(activations, unique_active_unit_indices)
    if len(unique_active_unit_indices) > 0:
        correlations_p, correlations_s = get_correlations_with_data(activations, unique_active_unit_indices, comparison_data)
        n_per_attribute = get_n_features_per_attribute(correlations_s)
        n_per_rna = n_per_attribute[:5].mean()
        n_per_tf = n_per_attribute[5:8].mean()
        n_per_activity = n_per_attribute[8]
        n_per_accessibility = n_per_attribute[9:].mean()
        highest_corrs = get_highest_corr_per_attribute(correlations_s)
        highest_corrs_rna = highest_corrs[:5].mean()
        highest_corrs_tf = highest_corrs[5:8].mean()
        highest_corrs_activity = highest_corrs[8]
        highest_corrs_accessibility = highest_corrs[9:].mean()
    else:
        n_per_rna = 0
        n_per_tf = 0
        n_per_activity = 0
        n_per_accessibility = 0
        highest_corrs_rna = 0
        highest_corrs_tf = 0
        highest_corrs_activity = 0
        highest_corrs_accessibility = 0
    return n_activation_features, n_unique, avg_active_hidden_units, n_redundant, n_per_rna, n_per_tf, n_per_activity, n_per_accessibility, highest_corrs_rna, highest_corrs_tf, highest_corrs_activity, highest_corrs_accessibility

############################################

# load the data
rna_counts = torch.tensor(np.load("01_data/sim_rna_counts.npy"))
tf_scores = torch.tensor(np.load("01_data/sim_tf_scores.npy"))
activity_score = torch.tensor(np.load("01_data/sim_activity_scores.npy"))
accessibility_scores = torch.tensor(np.load("01_data/sim_accessibility_scores.npy"))

model_name = 'layer1_latent4'
latent_size = 4

encoder = torch.load('03_results/models/sim1_'+model_name+'_encoder.pth')

decoder = torch.load('03_results/models/sim1_'+model_name+'_decoder.pth')

input_size = latent_size
batch_size = 128
reps = encoder(rna_counts).detach()
train_data = reps.clone().to(device)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

comparison_data = np.concatenate((rna_counts, tf_scores, activity_score, accessibility_scores), axis=1)

############################################

sae_metrics_dict = {
    'hidden_factor': [],
    'lr': [],
    'k': [],
    'sparsity_penalty': [],
    'loss': [],
    'n_activation_features': [],
    'n_unique': [],
    'avg_active_hidden_units': [],
    'n_redundant': [],
    'n_per_rna': [],
    'n_per_tf': [],
    'n_per_activity': [],
    'n_per_accessibility': [],
    'highest_corrs_rna': [],
    'highest_corrs_tf': [],
    'highest_corrs_activity': [],
    'highest_corrs_accessibility': []
}

hidden_factor_options = [2, 5, 10, 20, 50, 100, 200, 1000]
#lr_options = [1e-2, 1e-3, 1e-4, 1e-5]
lr_options = [1e-4]
k_options = [1, 5, 10, 20, 50, 75, 100]
l1_weight = 0
sparsity_penalty = 0

# add a progress bar
from tqdm import tqdm
# add the progress bar to the outer loop
for hidden_factor in tqdm(hidden_factor_options):
    for lr in lr_options:
        for k in k_options:
            sae_setup = [hidden_factor, lr, l1_weight, sparsity_penalty]
            sae_model, loss = train_sae(train_loader, input_size=input_size, hidden_factor=hidden_factor, lr=lr, l1_weight=l1_weight, sparsity_penalty=sparsity_penalty, n_epochs=500, sae_type='topk', k_latent_percent=k)
            sae_setup.append(loss)

            reps_reconstructed, activations = sae_model(reps.to(device))
            sae_metrics = run_sae_analysis(activations, comparison_data)
            sae_metrics = sae_setup + list(sae_metrics)
            for i, key in enumerate(sae_metrics_dict.keys()):
                sae_metrics_dict[key].append(sae_metrics[i])

# make the dict into a pandas dataframe and save it
df_sae_metrics = pd.DataFrame(sae_metrics_dict)
df_sae_metrics.to_csv('03_results/reports/files/sim_modl1l4_sae-'+sae_type+'_metrics_spearman.csv', index=False)