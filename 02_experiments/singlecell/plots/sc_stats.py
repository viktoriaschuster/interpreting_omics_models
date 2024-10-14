import torch
import numpy as np
import pandas as pd
import anndata as ad
import multiDGD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

data = ad.read_h5ad("./01_data/human_bonemarrow.h5ad")

model = multiDGD.DGD.load(data=data, save_dir="./03_results/models/", model_name="human_bonemarrow_l20_h2-3_test50e")
reps = model.representation.z.detach()
data = data[data.obs["train_val_test"] == "train"]

sae_model_save_name = '03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs'

input_size = reps.shape[1]
hidden_size = 10**4
sae_model = SparseAutoencoder(input_size, hidden_size)
sae_model.load_state_dict(torch.load(sae_model_save_name+'.pt'))
sae_model.to(device)

batch_size = 128
train_data = reps.clone()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

####################################################################################################
# calc activations
####################################################################################################

threshold = 1e-10

unique_active_unit_indices = get_unique_active_unit_indices(sae_model, train_loader, threshold=threshold)

avg_active_hidden_units = count_active_hidden_units(sae_model, train_loader, threshold=threshold, avg=True)

reps_reconstructed, activations = sae_model(torch.tensor(reps, dtype=torch.float32).to(device))

# stuff for plots
avg_activation_per_neuron = activations.detach().cpu().mean(dim=0).numpy()
df_avg_activs = pd.DataFrame(data=avg_activation_per_neuron, columns=['mean activation'])
df_avg_activs['active'] = ['active' if i in unique_active_unit_indices else 'dead' for i in range(avg_activation_per_neuron.shape[0])]

num_active_neurons = (activations > threshold).sum(dim=1).detach().cpu().numpy()

####################################################################################################
# make plots
####################################################################################################

plt.rcParams.update({'font.size': 15})

# make a figure with 2 subplots

fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
# add space between subplots
plt.subplots_adjust(wspace=1.0)

# first subplot is a histogram of dead vs active neurons
sns.histplot(df_avg_activs, x='mean activation', hue='active', bins=100, ax=axs[0], linewidth=0, alpha=0.7)
axs[0].set_xlabel('Activation')
axs[0].set_ylabel('Number of neurons')
axs[0].set_yscale('log')
axs[0].set_title('Mean activation per neuron')
axs[0].legend(['active', 'dead'], title='Neuron status', frameon=False)

# second subplot is a histogram of the number of active neurons per cell
sns.histplot(num_active_neurons, bins=100, ax=axs[1], linewidth=0, alpha=0.7)
plt.axvline(x=avg_active_hidden_units, color='black', linestyle='--')
# annotate the average number of active neurons
plt.text(avg_active_hidden_units+20, 5000, f'Mean: {avg_active_hidden_units:.1f}', color='black')
axs[1].set_xlabel('Number of firing neurons')
axs[1].set_ylabel('Number of cells')
axs[1].set_yscale('log')
axs[1].set_title('Active neurons per cell')

plt.tight_layout()
# save
plt.savefig("03_results/figures/sc_avg_activations_and_num_active_neurons.pdf")