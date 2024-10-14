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
# PCA
####################################################################################################

pca = PCA(n_components=2)
pca.fit(reps.cpu().numpy())
# plot the representations
reps_transformed = pca.transform(reps.cpu().numpy())
clusters = model.gmm.clustering(torch.tensor(reps)).detach().cpu().numpy()

df_celltypes = pd.DataFrame(reps_transformed, columns=["PC 1", "PC 2"])
df_celltypes["type"] = "original"
df_celltypes["component"] = clusters
df_celltypes["component"] = df_celltypes["component"].astype(str)
df_celltypes["celltype"] = data.obs["cell_type"].values
unique_values = data.obs["cell_type"].cat.categories

feature = 2306
df_celltypes['feat2306'] = activations[:, feature].detach().cpu().numpy()

####################################################################################################
# make plots
####################################################################################################

plt.rcParams.update({'font.size': 15})
palette = ['lightgrey', 'black', 'lightseagreen']
point_size = 2
edgecolor = None
alpha=1
marker_scale = 4
handletextpad = 0.2

# make a figure with 2 subplots

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
# add space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.3)

###
# HSC
###
i, j = 0, 0
ct1 = 'HSC'
ct2 = 'Normoblast'
ct1_indices = np.where(data.obs['cell_type'] == ct1)[0]
latents_non_ct1 = reps[np.where(data.obs['cell_type'] != ct1)[0], :].cpu().numpy()
latents_ct1 = reps[ct1_indices, :].cpu().numpy()
activs_ct1 = activations[ct1_indices, :]
active_values_ct2 = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == ct2)[0],:].detach().cpu().numpy(), axis=0)).to(device)
activs_perturbed = activs_ct1.clone()
activs_perturbed[:, feature] = active_values_ct2[feature]
latents_perturbed = sae_model.decoder(activs_perturbed.to(device)).detach().cpu().numpy()

df_temp = pd.DataFrame(reps_transformed, columns=['PC 1', 'PC 2'])
df_temp['type'] = 'other'
df_temp2 = pd.DataFrame(pca.transform(latents_ct1), columns=['PC 1', 'PC 2'])
df_temp2['type'] = ct1
df_temp = pd.concat([df_temp, df_temp2])
df_temp2 = pd.DataFrame(pca.transform(latents_perturbed), columns=['PC 1', 'PC 2'])
df_temp2['type'] = ct1 + ' perturbed'
df_temp = pd.concat([df_temp, df_temp2])

sns.scatterplot(data=df_temp, x='PC 1', y='PC 2', hue='type', ax=axs[i, j], palette=palette, ec=edgecolor, s=point_size, alpha=alpha)
axs[i, j].set_xticks([])
axs[i, j].set_yticks([])
axs[i, j].set_title(ct1)
# remove the legend
axs[i, j].get_legend().remove()

###
# Proerythroblast
###
i, j = 0, 1
ct1 = 'Proerythroblast'
ct2 = 'Normoblast'
ct1_indices = np.where(data.obs['cell_type'] == ct1)[0]
latents_non_ct1 = reps[np.where(data.obs['cell_type'] != ct1)[0], :].cpu().numpy()
latents_ct1 = reps[ct1_indices, :].cpu().numpy()
activs_ct1 = activations[ct1_indices, :]
active_values_ct2 = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == ct2)[0],:].detach().cpu().numpy(), axis=0)).to(device) * 1.5
activs_perturbed = activs_ct1.clone()
activs_perturbed[:, feature] = active_values_ct2[feature]
latents_perturbed = sae_model.decoder(activs_perturbed.to(device)).detach().cpu().numpy()

df_temp = pd.DataFrame(reps_transformed, columns=['PC 1', 'PC 2'])
df_temp['type'] = 'other'
df_temp2 = pd.DataFrame(pca.transform(latents_ct1), columns=['PC 1', 'PC 2'])
df_temp2['type'] = 'normal'
df_temp = pd.concat([df_temp, df_temp2])
df_temp2 = pd.DataFrame(pca.transform(latents_perturbed), columns=['PC 1', 'PC 2'])
df_temp2['type'] = 'perturbed'
df_temp = pd.concat([df_temp, df_temp2])

sns.scatterplot(data=df_temp, x='PC 1', y='PC 2', hue='type', ax=axs[i, j], palette=palette, ec=edgecolor, s=point_size, alpha=alpha)
axs[i, j].set_xticks([])
axs[i, j].set_yticks([])
axs[i, j].set_title(ct1)
# legend to the right
# remove the first handle and label
handles, labels = axs[i, j].get_legend_handles_labels()
axs[i, j].legend(handles=handles[1:], labels=labels[1:], loc='center left', bbox_to_anchor=(1, 0.5), title='Sample', frameon=False, markerscale=marker_scale, handletextpad=handletextpad, alignment='left')

###
# Proerythroblast
###
i, j = 1, 0
ct1 = 'NK'
ct2 = 'Normoblast'
ct1_indices = np.where(data.obs['cell_type'] == ct1)[0]
latents_non_ct1 = reps[np.where(data.obs['cell_type'] != ct1)[0], :].cpu().numpy()
latents_ct1 = reps[ct1_indices, :].cpu().numpy()
activs_ct1 = activations[ct1_indices, :]
active_values_ct2 = torch.Tensor(np.mean(activations[np.where(data.obs['cell_type'] == ct2)[0],:].detach().cpu().numpy(), axis=0)).to(device) * 1.5
activs_perturbed = activs_ct1.clone()
activs_perturbed[:, feature] = active_values_ct2[feature]
latents_perturbed = sae_model.decoder(activs_perturbed.to(device)).detach().cpu().numpy()

df_temp = pd.DataFrame(reps_transformed, columns=['PC 1', 'PC 2'])
df_temp['type'] = 'other'
df_temp2 = pd.DataFrame(pca.transform(latents_ct1), columns=['PC 1', 'PC 2'])
df_temp2['type'] = ct1
df_temp = pd.concat([df_temp, df_temp2])
df_temp2 = pd.DataFrame(pca.transform(latents_perturbed), columns=['PC 1', 'PC 2'])
df_temp2['type'] = ct1 + ' perturbed'
df_temp = pd.concat([df_temp, df_temp2])

sns.scatterplot(data=df_temp, x='PC 1', y='PC 2', hue='type', ax=axs[i, j], palette=palette, ec=edgecolor, s=point_size, alpha=alpha)
axs[i, j].set_xticks([])
axs[i, j].set_yticks([])
axs[i, j].set_title(ct1)
# remove the legend
axs[i, j].get_legend().remove()

###
# CD8+ T
###
palette = ['lightgrey', 'peachpuff', 'firebrick']
i, j = 1, 1
ct1 = 'CD8+ T'
cd8_indices = np.where(data.obs['cell_type'] == 'CD8+ T')[0]
activs_feat = activations[cd8_indices, feature].detach().cpu().numpy()
# kick out the zeros (this is where the feature is not active, and this screws up the quantiles)
activs_feat = activs_feat[activs_feat > 0]
quantile_steps = [0.05, 0.1, 0.25, 0.5, 0.75, 0.95]
quantiles = np.quantile(activs_feat, quantile_steps)
thresholds = [quantiles[0], quantiles[-1]]
# get indices of CD8+T cells and indices of where the feature is active (> 0.01)
ct_indices = np.where(data.obs['cell_type'] == 'CD8+ T')[0]
feat_indices_pos = np.where(activations.detach().cpu()[:, feature] > thresholds[1])[0]
feat_indices_neg = np.where(activations.detach().cpu()[:, feature] < thresholds[0])[0]

ct_pos_indices = np.intersect1d(ct_indices, feat_indices_pos)
#ct_neg_indices = np.setdiff1d(ct_indices, ct_pos_indices)
ct_neg_indices = np.intersect1d(ct_indices, feat_indices_neg)

latents_pos = reps[ct_pos_indices, :].cpu().numpy()
latents_neg = reps[ct_neg_indices, :].cpu().numpy()
activs_pos = activations[ct_pos_indices, :]
activs_neg = activations[ct_neg_indices, :]

df_temp = pd.DataFrame(reps_transformed, columns=['PC 1', 'PC 2'])
df_temp['type'] = 'other'
df_temp2 = pd.DataFrame(pca.transform(latents_neg), columns=['PC 1', 'PC 2'])
df_temp2['type'] = 'low'
df_temp = pd.concat([df_temp, df_temp2])
df_temp2 = pd.DataFrame(pca.transform(latents_pos), columns=['PC 1', 'PC 2'])
df_temp2['type'] = 'high'
df_temp = pd.concat([df_temp, df_temp2])

sns.scatterplot(data=df_temp, x='PC 1', y='PC 2', hue='type', ax=axs[i, j], palette=palette, ec=edgecolor, s=point_size, alpha=alpha)
axs[i, j].set_xticks([])
axs[i, j].set_yticks([])
axs[i, j].set_title(ct1)
# remove the legend
# move the legend to the right
handles, labels = axs[i, j].get_legend_handles_labels()
axs[i, j].legend(handles=handles[1:], labels=labels[1:], loc='center left', bbox_to_anchor=(1, 0.5), title='Activation\nstatus', frameon=False, markerscale=marker_scale, handletextpad=handletextpad, alignment='left')

plt.tight_layout()
# save the figure
plt.savefig('03_results/figures/sc_latent_perturb.pdf')