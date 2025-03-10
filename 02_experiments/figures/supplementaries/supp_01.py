import os
import torch
import pandas as pd
import numpy as np
import random
import gc
import tqdm

import sys
sys.path.append(".")
sys.path.append('src')

# set a random seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

############################################
# load data
############################################

complexity = 'high'
n_samples = 100000
data_dir = '/home/vschuste/data/simulation/'

for seed in range(10):
    temp_y = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed), weights_only=False)
    temp_x0 = torch.load(data_dir+'large_{}-complexity_rs{}_x0.pt'.format(complexity, seed), weights_only=False)
    temp_x1 = torch.load(data_dir+'large_{}-complexity_rs{}_x1.pt'.format(complexity, seed), weights_only=False)
    temp_x2 = torch.load(data_dir+'large_{}-complexity_rs{}_x2.pt'.format(complexity, seed), weights_only=False)
    temp_ct = torch.load(data_dir+'large_{}-complexity_rs{}_ct.pt'.format(complexity, seed), weights_only=False)
    temp_cov = torch.load(data_dir+'large_{}-complexity_rs{}_co.pt'.format(complexity, seed), weights_only=False)
    if seed == 0:
        rna_counts = temp_y
        x0 = temp_x0
        x1 = temp_x1
        x2 = temp_x2
        ct = temp_ct
        co = temp_cov
    else:
        rna_counts = torch.cat((rna_counts, temp_y), dim=0)
        x0 = torch.cat((x0, temp_x0), dim=0)
        x1 = torch.cat((x1, temp_x1), dim=0)
        x2 = torch.cat((x2, temp_x2), dim=0)
        ct = torch.cat((ct, temp_ct), dim=0)
        co = torch.cat((co, temp_cov), dim=0)
# limit to the training data
n_samples_train = int(n_samples*0.9)
rna_counts = rna_counts[:n_samples_train]
x0 = x0[:n_samples_train]
x1 = x1[:n_samples_train]
x2 = x2[:n_samples_train]
ct = ct[:n_samples_train]
co = co[:n_samples_train]

# now the real data
import anndata as ad
import multiDGD
data = ad.read_h5ad("/home/vschuste/data/singlecell/human_bonemarrow.h5ad")
model = multiDGD.DGD.load(data=data, save_dir="./03_results/models/", model_name="human_bonemarrow_l20_h2-3_test50e")
reps = model.representation.z.detach()
data = data[data.obs["train_val_test"] == "train"]

############################################
# prep the pcas
############################################

from sklearn.decomposition import PCA

y_pca = PCA(n_components=2).fit_transform(rna_counts)
y_pca = pd.DataFrame(y_pca, columns=['PC1', 'PC2'])
y_pca['A'] = ct
y_pca['B'] = co
x2_pca = PCA(n_components=2).fit_transform(x2)
x2_pca = pd.DataFrame(x2_pca, columns=['PC1', 'PC2'])
x2_pca['A'] = ct
x2_pca['B'] = co

data_pca = PCA(n_components=2).fit_transform(np.log1p(np.asarray(data.X[:,data.var['modality'] == 'GEX'].todense())))
data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
data_pca['celltype'] = data.obs['observable'].values
data_pca['covariate'] = data.obs['covariate_Site'].values
reps_pca = PCA(n_components=2).fit_transform(reps.cpu().numpy())
reps_pca = pd.DataFrame(reps_pca, columns=['PC1', 'PC2'])
reps_pca['celltype'] = data.obs['observable'].values
reps_pca['covariate'] = data.obs['covariate_Site'].values

############################################
# plot the data
############################################

# I want 3 rows and 4 columns in gridspec

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

fontsize = 14
plt.rcParams.update({'font.size': fontsize})

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(3, 4)
# add spacing
gs.update(wspace=0.4, hspace=0.4)

# first row: PCAs of Y colored by ct, cov, and then real data colored by cell type and covariate
ax1 = fig.add_subplot(gs[0, 0])
# add A to the figure
ax1.text(-0.35, 1.1, 'A', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=16, fontweight='bold')
sns.scatterplot(data=y_pca, x='PC1', y='PC2', hue='A', ax=ax1, palette='tab20', s=1, ec=None)
# remove axis ticks and legend
ax1.get_legend().remove()
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Y (c=activity A)')

ax2 = fig.add_subplot(gs[0, 1])
sns.scatterplot(data=y_pca, x='PC1', y='PC2', hue='B', ax=ax2, palette='tab20', s=1, ec=None)
ax2.get_legend().remove()
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('Y (c=noise B)')

ax3 = fig.add_subplot(gs[0, 2])
sns.scatterplot(data=data_pca, x='PC1', y='PC2', hue='celltype', ax=ax3, palette='tab20', s=1, ec=None)
ax3.get_legend().remove()
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_title('sc counts (c=cell type)')

ax4 = fig.add_subplot(gs[0, 3])
sns.scatterplot(data=data_pca, x='PC1', y='PC2', hue='covariate', ax=ax4, palette='tab20', s=1, ec=None)
ax4.get_legend().remove()
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_title('sc counts (c=covariate)')

# second row is the same but for x2 and reps
ax5 = fig.add_subplot(gs[1, 0])
ax5.text(-0.35, 1.1, 'B', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes, fontsize=16, fontweight='bold')
sns.scatterplot(data=x2_pca, x='PC1', y='PC2', hue='A', ax=ax5, palette='tab20', s=1, ec=None)
ax5.get_legend().remove()
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_title("X'' (c=activity A)")

ax6 = fig.add_subplot(gs[1, 1])
sns.scatterplot(data=x2_pca, x='PC1', y='PC2', hue='B', ax=ax6, palette='tab20', s=1, ec=None)
ax6.get_legend().remove()
ax6.set_xticks([])
ax6.set_yticks([])
ax6.set_title("X'' (c=noise B)")

ax7 = fig.add_subplot(gs[1, 2])
sns.scatterplot(data=reps_pca, x='PC1', y='PC2', hue='celltype', ax=ax7, palette='tab20', s=1, ec=None)
ax7.get_legend().remove()
ax7.set_xticks([])
ax7.set_yticks([])
ax7.set_title('sc latent (c=cell type)')

ax8 = fig.add_subplot(gs[1, 3])
sns.scatterplot(data=reps_pca.sort_values(by='celltype'), x='PC1', y='PC2', hue='covariate', ax=ax8, palette='tab20', s=1, ec=None)
ax8.get_legend().remove()
ax8.set_xticks([])
ax8.set_yticks([])
ax8.set_title('sc latent (c=covariate)')

# the last row will have two plots spanning over 2 columns each, depicting count histograms of the data
ax9 = fig.add_subplot(gs[2, 0:2])
ax9.text(-0.15, 1.1, 'C', horizontalalignment='center', verticalalignment='center', transform=ax9.transAxes, fontsize=16, fontweight='bold')
sns.histplot(rna_counts[:10,:].flatten(), ax=ax9, bins=100, color='tab:blue')
ax9.set_yscale('log')
ax9.set_title('Y value counts')

ax10 = fig.add_subplot(gs[2, 2:4])
sns.histplot(np.log1p(np.asarray(data.X[:10,:].todense()).flatten()), ax=ax10, bins=100, color='tab:orange')
ax10.set_yscale('log')
ax10.set_title('sc value counts')

# save the figure
fig.savefig('03_results/figures/simL_data.png', dpi=300, bbox_inches='tight')