import torch
import numpy as np
import pandas as pd
import anndata as ad
import multiDGD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import math

import sys
sys.path.append(".")
sys.path.append('src')
from src.models.sparse_autoencoder import *
from src.visualization.plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cuda:1'

####################################################################################################
# load data and models
####################################################################################################

data = ad.read_h5ad('/projects/heads/data/singlecell/human_bonemarrow.h5ad')
data = data[data.obs["train_val_test"] == "train"]

activations = torch.load('03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')

####################################################################################################
# calc ct-specific activations and determine locality
####################################################################################################

# go through all cell types and compute each features mean activation and std for ct positive and negative
cell_types = [x for x in data.obs['cell_type'].unique()]
means_positive = torch.zeros((len(cell_types), activations.shape[1]))
se_positive = torch.zeros((len(cell_types), activations.shape[1]))
means_negative = torch.zeros((len(cell_types), activations.shape[1]))
se_negative = torch.zeros((len(cell_types), activations.shape[1]))
for i, ct in enumerate(cell_types):
    data_indices = data.obs['cell_type'] == ct
    activations_pos = activations[data_indices,:]
    activations_neg = activations[~data_indices,:]
    means_positive[i,:] = torch.mean(activations_pos, dim=0).detach().cpu()
    # compute the standard error of the mean
    se_positive[i,:] = torch.std(activations_pos, dim=0).detach().cpu() / math.sqrt(activations_pos.shape[0])
    means_negative[i,:] = torch.mean(activations_neg, dim=0).detach().cpu()
    se_negative[i,:] = torch.std(activations_neg, dim=0).detach().cpu() / math.sqrt(activations_neg.shape[0])
# that gives the "significant features" per cell type
significant_features = torch.BoolTensor((means_positive - means_negative) > 1.96*(se_positive + se_negative))
sum_significant_features = torch.sum(significant_features, dim=0)

###
# unique (local) vs shared (global) features, and there is a thing in the middle which I call regional
###
# all active features
active_features = (torch.where(torch.sum(activations, dim=0) != 0)[0]).tolist()
# local features are the ct specific ones
local_features = (torch.where(sum_significant_features == 1)[0]).tolist()
# are there any features that are significant in all cts?
all_ct_features = (torch.where(torch.sum(significant_features, dim=0) == len(cell_types))[0]).tolist()
print('Number of features significant in all cell types: {}'.format(len(all_ct_features))) # was zero
regional_features = (torch.where(sum_significant_features > 1)[0]).tolist()
# all remaining active features are global
global_features = list(set(active_features).difference(set(local_features).union(set(regional_features))))

assert len(active_features) == len(local_features) + len(regional_features) + len(global_features)
assert len(set(local_features).intersection(set(regional_features))) == 0
assert len(set(local_features).intersection(set(global_features))) == 0

print('{} active features, {} local features, {} regional features, {} global features'.format(len(active_features), len(local_features), len(regional_features), len(global_features)))

# put it into a dataframe for nice plotting
df_sae_feat_activations = pd.DataFrame({
    'mean_activation': torch.mean(activations, dim=0).detach().cpu().numpy(),
    'max_activation': torch.max(activations, dim=0).values.detach().cpu().numpy(),
    'std_activation': torch.std(activations, dim=0).detach().cpu().numpy(),
    })
df_sae_feat_activations['feature_type'] = 'dead'
df_sae_feat_activations['feature_type'] = ['local' if i in local_features else x for i, x in enumerate(df_sae_feat_activations['feature_type'].values)]
df_sae_feat_activations['feature_type'] = ['regional' if i in regional_features else x for i, x in enumerate(df_sae_feat_activations['feature_type'].values)]
df_sae_feat_activations['feature_type'] = ['global' if i in global_features else x for i, x in enumerate(df_sae_feat_activations['feature_type'].values)]
df_sae_feat_activations = df_sae_feat_activations[df_sae_feat_activations['feature_type'] != 'dead']
# sort global regional local
df_sae_feat_activations['feature_type'] = pd.Categorical(df_sae_feat_activations['feature_type'], categories=['global', 'regional', 'local'], ordered=True)
# sort by feature type
df_sae_feat_activations = df_sae_feat_activations.sort_values('feature_type')

####################################################################################################
# make plots
####################################################################################################

plt.rcParams.update({'font.size': 15})
#sc_categorical_palette = ['#003f5c','#2f4b7c','#665191','#a05195','#d45087','#f95d6a','#ff7c43','#ffa600']
# subsample for 3 categories
#sc_categorical_palette = ['#ff7c43','#a05195','#2f4b7c']
sc_categorical_palette = sns.color_palette('colorblind', 3) # maybe stick to the other 3-col palette for now

# make a figure with 2 subplots

fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
# add space between subplots
plt.subplots_adjust(wspace=1.0)

# first subplot is a histogram of dead vs active neurons
sns.histplot(df_sae_feat_activations, x='mean_activation', hue='feature_type', bins=100, ax=axs[0], linewidth=0, alpha=0.8, palette=sc_categorical_palette)
axs[0].set_xlabel('Activation')
axs[0].set_ylabel('Number of neurons')
axs[0].set_yscale('log')
axs[0].set_title('Mean activation')
#axs[0].legend(['local', 'regional', 'global'], title='Neuron space', frameon=False)
axs[0].legend(['global', 'regional', 'local'], title='Neuron space', frameon=False)

# second subplot is a histogram of the number of active neurons per cell
sns.histplot(df_sae_feat_activations, x='max_activation', hue='feature_type', bins=100, ax=axs[1], linewidth=0, alpha=0.8, palette=sc_categorical_palette)
#plt.axvline(x=avg_active_hidden_units, color='black', linestyle='--')
# annotate the average number of active neurons
#plt.text(avg_active_hidden_units+20, 5000, f'Mean: {avg_active_hidden_units:.1f}', color='black')
axs[1].set_xlabel('Activation')
axs[1].set_ylabel('Number of neurons')
axs[1].set_yscale('log')
axs[1].set_title('Max activation')
axs[1].legend().remove()

plt.tight_layout()
# save
plt.savefig("03_results/figures/sc_activations_by_locality.pdf")