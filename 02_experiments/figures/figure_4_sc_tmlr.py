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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cuda:1'

####################################################################################################
# load data and models
####################################################################################################

data = ad.read_h5ad("/home/vschuste/data/singlecell/human_bonemarrow.h5ad")

model = multiDGD.DGD.load(data=data, save_dir="./03_results/models/", model_name="human_bonemarrow_l20_h2-3_test50e")
reps = model.representation.z.detach()
data = data[data.obs["train_val_test"] == "train"]

sae_model_save_name = '03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs'
input_size = reps.shape[1]
hidden_size = 10**4
import torch.nn as nn
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            #nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
sae_model = SparseAutoencoder(input_size, hidden_size)
sae_model.load_state_dict(torch.load(sae_model_save_name+'.pt'))
sae_model.to(device)

activations = torch.load('03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')
#active_features = (torch.where(torch.sum(activations, dim=0) != 0)[0]).tolist()

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
# adjust the visualized cell types. we only want to keep all B cell types and those along the erythroid lineage
cts_to_keep = ['HSC', 'MK/E prog', 'Proerythroblast', 'Erythroblast', 'Normoblast', 'Transitional B', 'Naive CD20+ B', 'B1 B']
#df_celltypes["celltype"] = data.obs["cell_type"].values
df_celltypes["celltype"] = [ct if ct in cts_to_keep else "Other" for ct in data.obs["cell_type"].values]
# set it to a category and order it
df_celltypes["celltype"] = pd.Categorical(df_celltypes["celltype"], categories=["Other"]+cts_to_keep, ordered=True)
unique_values = cts_to_keep
celltype_palette = ["grey"] + sns.color_palette("YlOrBr", n_colors=5) + sns.color_palette("crest", n_colors=3)

df_celltypes['feat2306'] = activations[:, 2306].detach().cpu().numpy()

####################################################################################################
# make plots
####################################################################################################

plt.rcParams.update({'font.size': 14})

# make a figure with 2 subplots

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
# add space between subplots
#plt.subplots_adjust(wspace=0.5, bottom=0.5, left=0.07)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
palette = ['lightgrey'] + [sns.color_palette('rocket_r', n_colors=10)[i] for i in [3,8]]
point_size = 1
edgecolor = None
alpha=0.7
marker_scale = 6
handletextpad = 0.2

###
# first row: global features
###

# remove the white lines arount the dots
sns.scatterplot(data=df_celltypes.sort_values(by='celltype'), x="PC 1", y="PC 2", hue="celltype", s=1, alpha=0.7, ec=None, ax=axs[0,0], palette=celltype_palette)
# add "A" and "B" to the plots
axs[0,0].text(-0.3, 1.1, "A", fontsize=18, transform=axs[0,0].transAxes, fontweight='bold')
axs[0,0].set_title("Representations")
# change the legend labels so that they show the index of the celltype from the unique_values list
handles, labels = axs[0,0].get_legend_handles_labels()
# put the legend underneat the plot
#labels = [str(i-1) + ": " + label if label != "Other" else "Other" for i, label in enumerate(labels)]
axs[0,0].legend(
    handles,
    labels,
    title="Cell type",
    loc="upper left",
    bbox_to_anchor=(3.6, 1.1),
    alignment='left',
    markerscale=marker_scale,
    #fontsize=12,
    ncol=1,
    frameon=False,
    handletextpad=handletextpad,
    columnspacing=0.8,
)#.remove()
# remove the ticks
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])

# annotate the plot with the celltype names (on means)
#for i, celltype in enumerate(unique_values):
#    mean = np.mean(reps_transformed[df_celltypes["celltype"] == celltype], axis=0)
#    # also plot a black dot
#    axs[0,0].scatter(mean[0], mean[1], color='black', s=5)
#    axs[0,0].annotate(i, (mean[0], mean[1]), fontsize=12)

feat_palette = sns.color_palette('rocket_r', as_cmap=True)
# the second plot will be the same latent space but colored by the activation of feature 2306
sns.scatterplot(data=df_celltypes.sort_values(by='feat2306', ascending=True), x="PC 1", y="PC 2", hue="feat2306", s=1, alpha=0.7, ec=None, ax=axs[0,1], palette=feat_palette)
axs[0,1].set_title("Neuron 2306")
# remove the ticks
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
# put the leend outside the plot
#cbar_points = plt.scatter([], [], c=[], vmin=df_celltypes['feat2306'].min(), vmax=df_celltypes['feat2306'].max(), cmap=feat_palette)
#plt.colorbar(cbar_points, ax=axs[0,1], label='Activation')
# remove the normal legend
axs[0,1].get_legend().remove()

# proerythroblast perturbation
feature = 2306
i, j = 0, 2
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
axs[i, j].set_title('Feature pertrubation')
# legend to the right
# remove the first handle and label
handles, labels = axs[i, j].get_legend_handles_labels()
axs[i, j].legend(handles=handles[1:], labels=labels[1:], loc='center left', bbox_to_anchor=(1, 0.5), title='Sample', frameon=False, markerscale=marker_scale, handletextpad=handletextpad, alignment='left').remove()

###
# second row: local features
###
i, j = 1, 0
axs[i,j].text(-0.3, 1.1, "B", fontsize=18, transform=axs[i,j].transAxes, fontweight='bold')
#feature = 1238
feature = 1500
df_celltypes[str(feature)] = activations[:, feature].detach().cpu().numpy()
feat_palette = sns.color_palette('rocket_r', as_cmap=True)
# the second plot will be the same latent space but colored by the activation of feature 2306
sns.scatterplot(data=df_celltypes.sort_values(by=str(feature), ascending=True), x="PC 1", y="PC 2", hue=str(feature), size=str(feature), sizes=(1,10), alpha=0.7, ec=None, ax=axs[i,j], palette=feat_palette)
axs[i,j].set_title("Neuron {}".format(feature))
# remove the ticks
axs[i,j].set_xticks([])
axs[i,j].set_yticks([])
# put the leend outside the plot
#cbar_points = plt.scatter([], [], c=[], vmin=df_celltypes[str(feature)].min(), vmax=df_celltypes[str(feature)].max(), cmap=feat_palette)
#plt.colorbar(cbar_points, ax=axs[i,j], label='Activation')
# remove the normal legend
axs[i,j].get_legend().remove()

i, j = 1, 1
#feature = 5205
feature = 1238
df_celltypes[str(feature)] = activations[:, feature].detach().cpu().numpy()
feat_palette = sns.color_palette('rocket_r', as_cmap=True)
# the second plot will be the same latent space but colored by the activation of feature 2306
sns.scatterplot(data=df_celltypes.sort_values(by=str(feature), ascending=True), x="PC 1", y="PC 2", hue=str(feature), size=str(feature), sizes=(1,10), alpha=0.7, ec=None, ax=axs[i,j], palette=feat_palette)
axs[i,j].set_title("Neuron {}".format(feature))
# remove the ticks
axs[i,j].set_xticks([])
axs[i,j].set_yticks([])
# put the leend outside the plot
#cbar_points = plt.scatter([], [], c=[], vmin=df_celltypes[str(feature)].min(), vmax=df_celltypes[str(feature)].max(), cmap=feat_palette)
#plt.colorbar(cbar_points, ax=axs[i,j], label='Activation')
# remove the normal legend
axs[i,j].get_legend().remove()

i, j = 1, 2
#feature = 1500
feature = 5205
df_celltypes[str(feature)] = activations[:, feature].detach().cpu().numpy()
feat_palette = sns.color_palette('rocket_r', as_cmap=True)
# the second plot will be the same latent space but colored by the activation of feature 2306
sns.scatterplot(data=df_celltypes.sort_values(by=str(feature), ascending=True), x="PC 1", y="PC 2", hue=str(feature), size=str(feature), sizes=(1,10), alpha=0.7, ec=None, ax=axs[i,j], palette=feat_palette)
axs[i,j].set_title("Neuron {}".format(feature))
# remove the ticks
axs[i,j].set_xticks([])
axs[i,j].set_yticks([])
# put the leend outside the plot
#cbar_points = plt.scatter([], [], c=[], vmin=df_celltypes[str(feature)].min(), vmax=df_celltypes[str(feature)].max(), cmap=feat_palette)
#plt.colorbar(cbar_points, ax=axs[i,j], label='Activation')
# remove the normal legend
axs[i,j].get_legend().remove()

#plt.tight_layout()
# save
plt.savefig("03_results/figures/sc_reps_all_v3.png", dpi=300, bbox_inches='tight')