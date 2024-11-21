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

df_celltypes['feat2306'] = activations[:, 2306].detach().cpu().numpy()

####################################################################################################
# make plots
####################################################################################################

plt.rcParams.update({'font.size': 15})

# make a figure with 2 subplots

fig, axs = plt.subplots(1, 2, figsize=(8, 6))
# add space between subplots
plt.subplots_adjust(wspace=0.5, bottom=0.5, left=0.07)

# remove the white lines arount the dots
sns.scatterplot(data=df_celltypes, x="PC 1", y="PC 2", hue="celltype", s=1, alpha=0.7, ec=None, ax=axs[0])
axs[0].set_title("Representations")
# change the legend labels so that they show the index of the celltype from the unique_values list
handles, labels = axs[0].get_legend_handles_labels()
"""
axs[0].legend(
    handles,
    [str(np.where(unique_values == label)[0][0]) + " (" + label + ")" for label in labels],
    title="Cell type",
    bbox_to_anchor=(1, 1.05),
    loc="upper left",
    markerscale=4,
    ncol=1,
    frameon=False,
    handletextpad=0.2,
    columnspacing=0.8,
)
"""
# put the legend underneat the plot
axs[0].legend(
    handles,
    [str(np.where(unique_values == label)[0][0]) + " (" + label + ")" for label in labels],
    title="Cell type",
    loc="lower left",
    bbox_to_anchor=(-0.15, -1.3),
    alignment='left',
    markerscale=6,
    fontsize=12,
    ncol=3,
    frameon=False,
    handletextpad=0.2,
    columnspacing=0.8,
)#.remove()
# remove the ticks
axs[0].set_xticks([])
axs[0].set_yticks([])

# annotate the plot with the celltype names (on means)
for i, celltype in enumerate(unique_values):
    mean = np.mean(reps_transformed[data.obs["cell_type"] == celltype], axis=0)
    # also plot a black dot
    axs[0].scatter(mean[0], mean[1], color='black', s=5)
    axs[0].annotate(i, (mean[0], mean[1]), fontsize=12)

feat_palette = sns.color_palette('rocket_r', as_cmap=True)
# the second plot will be the same latent space but colored by the activation of feature 2306
sns.scatterplot(data=df_celltypes, x="PC 1", y="PC 2", hue="feat2306", s=1, alpha=0.7, ec=None, ax=axs[1], palette=feat_palette)
axs[1].set_title("Neuron 2306 activation")
# remove the ticks
axs[1].set_xticks([])
axs[1].set_yticks([])
# put the leend outside the plot
cbar_points = plt.scatter([], [], c=[], vmin=df_celltypes['feat2306'].min(), vmax=df_celltypes['feat2306'].max(), cmap=feat_palette)
plt.colorbar(cbar_points, ax=axs[1], label='Activation')
# remove the normal legend
axs[1].get_legend().remove()
"""
axs[1].legend(
    title='Activation',
    bbox_to_anchor=(1, 1.05),
    loc='upper left',
    #fontsize=10,
    frameon=False,
    handletextpad=0.2,
)
"""

#plt.tight_layout()
# save
plt.savefig("03_results/figures/sc_latent.pdf")