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
fig_name = 'sc_activation_heatmap_erythro.png'

hematopoiesis_features = unique_active_unit_indices.copy()

# conditions for features to be met

# 1. the average value of the feature must be higher in the hematopoietic cells than in the non-hematopoietic cells
celltypes = np.unique(data.obs['cell_type'].values)
hematopoietic_cells = ['MK/E prog', 'Proerythroblast', 'Erythroblast', 'Normoblast']
stem_cells = ['HSC']
non_hematopoietic_cells = [x for x in celltypes if x not in hematopoietic_cells + stem_cells]
temp_indices_true = np.where(data.obs['cell_type'].isin(hematopoietic_cells))[0]
temp_indices_false = np.where(data.obs['cell_type'].isin(non_hematopoietic_cells))[0]
avg_activations_true = np.mean(activations[temp_indices_true,:].detach().cpu().numpy(), axis=0)
avg_activations_false = np.mean(activations[temp_indices_false,:].detach().cpu().numpy(), axis=0)
condition_accepted = avg_activations_true > avg_activations_false
print(f'Condition 1: {np.sum(condition_accepted)} features meet the condition')
hematopoiesis_features = np.intersect1d(hematopoiesis_features, np.where(condition_accepted)[0])

# 2. the average value of the feature must be higher in the hematopoietic cells than in the stem cells
temp_indices_false = np.where(data.obs['cell_type'].isin(stem_cells))[0]
avg_activations_false = np.mean(activations[temp_indices_false,:].detach().cpu().numpy(), axis=0)
condition_accepted = avg_activations_true > avg_activations_false
print(f'Condition 2: {np.sum(condition_accepted)} features meet the condition')
hematopoiesis_features = np.intersect1d(hematopoiesis_features, np.where(condition_accepted)[0])
print(f'Number of hematopoiesis features: {len(hematopoiesis_features)}')
temp_indices_true, temp_indices_false, avg_activations_true, avg_activations_false = None, None, None, None

# 3. the average value must increase consistently from the stem cells to the hematopoietic cells
ct_1 = 'HSC'
for ct_2 in hematopoietic_cells:
    temp_indices_low = np.where(data.obs['cell_type'] == ct_1)[0]
    temp_indices_high = np.where(data.obs['cell_type'] == ct_2)[0]
    avg_activations_low = np.mean(activations[temp_indices_low,:].detach().cpu().numpy(), axis=0)
    avg_activations_high = np.mean(activations[temp_indices_high,:].detach().cpu().numpy(), axis=0)
    condition_accepted = avg_activations_low < avg_activations_high
    hematopoiesis_features = np.intersect1d(hematopoiesis_features, np.where(condition_accepted)[0])
    ct_1 = ct_2
print(f'Condition 3: Number of hematopoiesis features: {len(hematopoiesis_features)}')


range_start_end = None
indices = hematopoiesis_features
obs = 'cell_type'
plot_size=(16,8)
obs_order = stem_cells + hematopoietic_cells + non_hematopoietic_cells

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
# plot 1
####################################################################################################

fig_dir = '03_results/figures/supplementaries/'
fig_name = 'sc_erythro_perturbations.png'

pca = PCA(n_components=2)
pca.fit(reps.cpu().numpy())
reps_transformed = pca.transform(reps.cpu().numpy())

stricter_selection = [1422, 2019, 2091, 2306, 4525, 4621, 4881, 5557, 5687, 5793, 6352, 7491, 7494, 7965, 8026, 8048]

hsc_indices = np.where(data.obs['cell_type'] == 'HSC')[0]
latents_non_hsc = reps[np.where(data.obs['cell_type'] != 'HSC')[0], :].cpu().numpy()
latents_hsc = reps[hsc_indices, :].cpu().numpy()
activs_hsc = activations[hsc_indices, :]
active_values_normoblast = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == 'Normoblast')[0],:].detach().cpu().numpy(), axis=0)).to(device)

# for each feature, get the HSC samples and plot the predicted latents before and after perturbation

# make a figure with 4 columns and as many rows as needed for the features

if len(stricter_selection) % 4 == 0:
    n_rows = len(stricter_selection) // 4
else:
    n_rows = len(stricter_selection) // 4 + 1

fig, axs = plt.subplots(n_rows, 4, figsize=(16, 3.5*n_rows))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

i = 0
for feat in stricter_selection:
    activs_perturbed = activs_hsc.clone()
    activs_perturbed[:, feat] = active_values_normoblast[feat]
    latents_perturbed = sae_model.decoder(activs_perturbed.to(device)).detach().cpu().numpy()

    df_temp = pd.DataFrame(reps_transformed, columns=['PC 1', 'PC 2'])
    df_temp['type'] = 'other'
    df_temp2 = pd.DataFrame(pca.transform(latents_hsc), columns=['PC 1', 'PC 2'])
    df_temp2['type'] = 'HSC'
    df_temp = pd.concat([df_temp, df_temp2])
    df_temp2 = pd.DataFrame(pca.transform(latents_perturbed), columns=['PC 1', 'PC 2'])
    df_temp2['type'] = 'HSC perturbed'
    df_temp = pd.concat([df_temp, df_temp2])

    col, row = i % 4, i // 4
    sns.scatterplot(df_temp, x='PC 1', y='PC 2', hue='type', ax=axs[row, col], s=1, alpha=0.7, palette=['grey', 'blue', 'red'])
    axs[row, col].get_legend().remove()
    axs[row, col].set_title(f'Neuron {feat}')
    i = i + 1

fig.savefig(fig_dir+fig_name, bbox_inches='tight', dpi=300)