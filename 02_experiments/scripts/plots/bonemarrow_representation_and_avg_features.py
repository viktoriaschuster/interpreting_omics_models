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
# calc PCA
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

####################################################################################################
# get avg feature values
####################################################################################################

reps_reconstructed, activations = sae_model(torch.tensor(reps, dtype=torch.float32).to(device))
threshold = 1e-10
unique_active_unit_indices = get_unique_active_unit_indices(sae_model, train_loader, threshold=threshold)
print(f"Number of active units: {len(unique_active_unit_indices)}")

celltype_values = data.obs["cell_type"].values
ct_activ_means = np.zeros((len(unique_values), len(unique_active_unit_indices)))
for i, ct in enumerate(unique_values):
    ids_ct = np.where(celltype_values == ct)[0]
    activations_ct = activations[ids_ct, :].detach().cpu().numpy()
    # normalize the activations to [0, 1]
    activations_ct = (activations_ct - np.min(activations_ct, axis=0)) / (np.max(activations_ct, axis=0) - np.min(activations_ct, axis=0))
    avg_activations = np.mean(activations_ct, axis=0)
    ct_activ_means[i, :] = avg_activations[unique_active_unit_indices]
    del activations_ct
    del avg_activations

# sort the mean activation columns by their own correlations and get the new indices
correlations = np.corrcoef(ct_activ_means.T)
sorted_indices = np.argsort(np.sum(correlations, axis=0))
#unique_active_unit_indices = unique_active_unit_indices[sorted_indices] #TypeError: only integer scalar arrays can be converted to a scalar index
unique_active_unit_indices = [unique_active_unit_indices[i] for i in sorted_indices]
ct_activ_means = ct_activ_means[:, sorted_indices]
df_ct_means = pd.DataFrame(ct_activ_means, index=unique_values, columns=unique_active_unit_indices)

####################################################################################################
# plot
####################################################################################################

# specify the font size
plt.rcParams.update({"font.size": 8})

# make the right column wider than the left column
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(10, 4))
gs = GridSpec(1, 2, width_ratios=[1, 2])
axs = [plt.subplot(gs[i]) for i in range(2)]

###
# pca
###

# remove the white lines arount the dots
sns.scatterplot(data=df_celltypes, x="PC 1", y="PC 2", hue="celltype", s=2, alpha=0.5, ec=None, ax=axs[0])
axs[0].set_title("Bone marrow representations")
# change the legend labels so that they show the index of the celltype from the unique_values list
handles, labels = axs[0].get_legend_handles_labels()
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
# remove the ticks
axs[0].set_xticks([])
axs[0].set_yticks([])

# annotate the plot with the celltype names (on means)
for i, celltype in enumerate(unique_values):
    mean = np.mean(reps_transformed[data.obs["cell_type"] == celltype], axis=0)
    axs[0].annotate(i, (mean[0], mean[1]), fontsize=10)

###
# heatmap
###

sns.heatmap(df_ct_means, cmap="rocket_r", ax=axs[1])
axs[1].set_title("Average SAE activation")
axs[1].set_xlabel("Active SAE neuron")
axs[1].set_ylabel("Cell type")
# change the yticks to show the index of the celltype from the unique_values list
axs[1].set_yticks(np.arange(len(unique_values))+0.5)
axs[1].set_yticklabels([str(np.where(unique_values == label)[0][0]) for label in unique_values])
plt.tight_layout()
# save the plot
plt.savefig("./03_results/figures/bonemarrow_celltypes.png", dpi=300)
