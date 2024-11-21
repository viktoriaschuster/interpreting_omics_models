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

plt.rcParams.update({'font.size': 14})

####################################################################################################
# load data and models
####################################################################################################

data = ad.read_h5ad("./01_data/human_bonemarrow.h5ad")

model = multiDGD.DGD.load(data=data, save_dir="./03_results/models/", model_name="human_bonemarrow_l20_h2-3_test50e")
reps = model.representation.z.detach()
data = data[data.obs["train_val_test"] == "train"]
celltypes = np.unique(data.obs['cell_type'].values)

sae_model_save_name = '03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs'

input_size = reps.shape[1]
hidden_size = 10**4
sae_model = SparseAutoencoder(input_size, hidden_size)
sae_model.load_state_dict(torch.load(sae_model_save_name+'.pt'))
sae_model.to(device)

batch_size = 128
train_data = reps.clone()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

threshold = 1e-10
unique_active_unit_indices = get_unique_active_unit_indices(sae_model, train_loader, threshold=threshold)
avg_active_hidden_units = count_active_hidden_units(sae_model, train_loader, threshold=threshold)

reps_reconstructed, activations = sae_model(torch.tensor(reps, dtype=torch.float32).to(device))

####################################################################################################
# plot 1
####################################################################################################

fig_dir = '03_results/figures/supplementaries/'
fig_name = 'sc_activation_heatmap.png'

range_start_end = None
indices = unique_active_unit_indices
obs = 'cell_type'
plot_size=(16,10)
obs_order = None

if range_start_end is not None:
    activs = activations[:, range_start_end[0]:range_start_end[1]].detach().cpu().numpy()
elif indices is not None:
    activs = activations[:, indices].detach().cpu().numpy()
else:
    activs = activations.detach().cpu().numpy()

if obs is not None:
    # sort by obs
    if range_start_end is not None:
        df_temp = pd.DataFrame(activs, columns=range(range_start_end[0], range_start_end[1]))
    elif indices is not None:
        df_temp = pd.DataFrame(activs, columns=indices)
    else:
        df_temp = pd.DataFrame(activs, columns=range(activs.shape[1]))
    df_temp['obs'] = data.obs[obs].values
    df_temp.index = data.obs[obs].values
    if obs_order is not None:
        df_temp['obs'] = pd.Categorical(df_temp['obs'], categories=obs_order, ordered=True)
    df_temp = df_temp.sort_values(by='obs')

    fig, ax = plt.subplots(figsize=plot_size)
    sns.heatmap(df_temp.drop(columns='obs'), cmap='rocket_r')
    if range_start_end is not None:
        plt.title(f'Activations for range {range_start_end[0]}-{range_start_end[1]}')
    else:
        plt.title(f'Activations')
    plt.xlabel('Hidden neuron')
    plt.ylabel(obs)
else:
    fig, ax = plt.subplots(figsize=plot_size)
    sns.heatmap(activs, cmap='rocket_r')
    if range_start_end is not None:
        plt.title(f'Activations for range {range_start_end[0]}-{range_start_end[1]}')
    else:
        plt.title(f'Activations')
    plt.xlabel('Hidden neuron')
plt.ylabel('Cell type')

fig.savefig(fig_dir+fig_name, bbox_inches='tight', dpi=300)

####################################################################################################
# plot 2
####################################################################################################

fig_name = 'sc_active_hidden_units_per_celltype.pdf'

# Assuming 'model' is your trained SparseAutoencoder and 'data_loader' is your dataset loader
def count_active_hidden_units(model, reps, threshold=1e-5, avg=True):

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():  # Disable gradient calculations for efficiency
            # Forward pass through the encoder to get the hidden layer activations
        _, encoded = model(torch.tensor(reps, dtype=torch.float32).to(device))
            
        # Count how many activations are above the threshold (active neurons)
        active_units_per_sample = (encoded > threshold).sum(dim=1)
    
    if avg:
        avg_active_units = sum(active_units_per_sample) / encoded.shape[0]
        return avg_active_units
    else:
        return active_units_per_sample

active_hidden_units = count_active_hidden_units(sae_model, reps, threshold=threshold, avg=False)

df_temp = pd.DataFrame(data=active_hidden_units.detach().cpu().numpy(), columns=['active_hidden_units'])
df_temp['celltype'] = data.obs['cell_type'].values

# plot a bar plot of the average number of active hidden units per celltype
plt.figure(figsize=(16, 4))
sns.barplot(x='celltype', y='active_hidden_units', data=df_temp)
plt.xlabel('Cell type')
plt.xticks(rotation=90)
plt.ylabel('Number of active hidden neurons')
plt.title('Average number of active hidden neurons per celltype')

plt.savefig(fig_dir+fig_name, bbox_inches='tight')