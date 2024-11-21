import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import random

import sys
sys.path.append(".")
sys.path.append('src')
from src.models.sparse_autoencoder import *
from src.visualization.plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set a random seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

####################################################################################################
# load data
####################################################################################################

df_sae_metrics = pd.read_csv('03_results/reports/files/sim_modl1l4_sae_metrics.csv')
# remove all rows with sparsity penalty > 0
df_sae_metrics = df_sae_metrics[df_sae_metrics['sparsity_penalty'] == 0.0]
df_sae_metrics['type'] = 'Vanilla'

df_sae_metrics_relu = pd.read_csv('03_results/reports/files/sim_modl1l4_sae-bricken2_metrics.csv')
df_sae_metrics_relu['type'] = 'ReLU'
df_sae_metrics = pd.concat([df_sae_metrics, df_sae_metrics_relu])

df_sae_metrics_topk = pd.read_csv('03_results/reports/files/sim_modl1l4_sae-topK_metrics.csv')
df_sae_metrics_topk['type'] = 'TopK'
df_sae_metrics = pd.concat([df_sae_metrics, df_sae_metrics_topk])

hidden_factor_options = [2, 5, 10, 20, 50, 100, 200, 1000]
lr_options = [1e-2, 1e-3, 1e-4, 1e-5]
l1_weight_options = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
k_options = [1, 5, 10, 20, 50, 75, 100]

df_sae_metrics['k'] = df_sae_metrics['k'].fillna(0)
df_sae_metrics['k'] = df_sae_metrics['k'].astype(int)
df_sae_metrics['k'] = df_sae_metrics['k'].astype(str)
df_sae_metrics['l1_weight'] = df_sae_metrics['l1_weight'].astype(str)
df_sae_metrics = df_sae_metrics[df_sae_metrics['lr'] == 1e-4]
df_sae_metrics = df_sae_metrics[df_sae_metrics['k'] != '1']
df_sae_metrics['fraction_unique'] = df_sae_metrics['n_unique'] / df_sae_metrics['n_activation_features']
df_sae_metrics['fraction_dead'] = 1 - df_sae_metrics['fraction_unique']

####################################################################################################
# set up figure
####################################################################################################

figure_kwargs = {
    'fig_width': 16,
    'fig_height': 12,
    'font_size': 14,
    'point_size': 20,
    'line_width_background': 0.5,
    'wspace': 1.0,
    'hspace': 0.7,
    'errorbar_linewidth': 1.0,
    'error_capsize': 0.3,
}
fig_dir = '03_results/figures/supplementaries/'
fig_name = 'sim_sae_performance2.pdf'

plt.rcParams.update({'font.size': figure_kwargs['font_size']})
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

###############

n_cols = 4

x = 'n_activation_features'
y_options = ['loss', 'fraction_dead', 'avg_active_hidden_units', 'highest_corrs_tf']
y_labels = ['MSE loss', 'Fraction of\ndead neurons', 'Avg. active\nhidden neurons', 'Highest correlation\nwith X']
y_lims = []
for y in y_options:
    y_lims.append((df_sae_metrics[y].min(), df_sae_metrics[y].max()))
n_rows = len(y_options)

#model_palette = sns.color_palette('colorblind', n_colors=3)
model_palette = ['grey','lightseagreen','darkorange']
l1_palette = sns.color_palette('viridis_r', n_colors=len(l1_weight_options))
k_palette = sns.color_palette('rocket_r', n_colors=len(k_options))

fig, axs = plt.subplots(n_rows, n_cols, figsize=(figure_kwargs['fig_width'], figure_kwargs['fig_height']))
plt.subplots_adjust(wspace=0.3, hspace=0.2)

# first column is generalized over models
j = 0
for i, y in enumerate(y_options):
    # print the number of unique data points per point
    print('Number of unique data points per point:')
    print(df_sae_metrics.groupby(['type', x]).size())
    sns.lineplot(data=df_sae_metrics, x=x, y=y, ax=axs[i,j], hue='type', style='type', markers=True, palette=model_palette)
    axs[i,j].set_xscale('log')
    if i == 1:
        axs[i,j].set_ylim([0, 1.02])
    elif i == 3:
        axs[i,j].set_ylim([0.5, 1.02])
    else:
        axs[i,j].set_ylim(y_lims[i])
    if i in [0,2]:
        axs[i,j].set_yscale('log')
    if i == 0:
        axs[i,j].set_title('All')
    if i == len(y_options) - 1:
        axs[i,j].set_xlabel('SAE hidden dimensionality')
    else:
        axs[i,j].set_xlabel('')
        axs[i,j].set_xticklabels([])
    axs[i,j].set_ylabel(y_labels[i])
    if i == len(y_options) - 1:
        axs[i,j].legend(loc='center left', bbox_to_anchor=(5.0, 4.3), ncol=1, frameon=False, title='SAE type', handletextpad=0.2, alignment='left')
    else:
        axs[i,j].get_legend().remove()

model_type = ['Vanilla', 'ReLU', 'TopK']
for j, t in enumerate(model_type):
    df_temp = df_sae_metrics[df_sae_metrics['type'] == t]
    for i, y in enumerate(y_options):
        if t == 'TopK':
            sns.lineplot(data=df_temp, x=x, y=y, ax=axs[i,j+1], hue='k', style='k', markers=True, palette=k_palette)
        else:
            sns.lineplot(data=df_temp, x=x, y=y, ax=axs[i,j+1], hue='l1_weight', style='l1_weight', markers=True, palette=l1_palette)
        axs[i,j+1].set_xscale('log')
        if i == 1:
            axs[i,j+1].set_ylim([0, 1.02])
        elif i == 3:
            axs[i,j+1].set_ylim([0.5, 1.02])
        else:
            axs[i,j+1].set_ylim(y_lims[i])
        if i in [0,2]:
            axs[i,j+1].set_yscale('log')
        if i == 0:
            axs[i,j+1].set_title(t)
        if i == len(y_options) - 1:
            axs[i,j+1].set_xlabel('SAE hidden dimensionality')
        else:
            axs[i,j+1].set_xlabel('')
            axs[i,j+1].set_xticklabels([])
        axs[i,j+1].set_ylabel('')
        if i == len(y_options) - 1:
            if j != 1:
                if t == 'TopK':
                    axs[i,j+1].legend(loc='center left', bbox_to_anchor=(1.1, 0.5), ncol=1, title='k (%)', frameon=False, handletextpad=0.2, alignment='left')
                else:
                    axs[i,j+1].legend(loc='center left', bbox_to_anchor=(3.7, 2.3), ncol=1, title='L1 weight', frameon=False, handletextpad=0.2, alignment='left')
            else:
                axs[i,j+1].get_legend().remove()
        else:
            axs[i,j+1].get_legend().remove()

###############

#plt.tight_layout()
plt.savefig(fig_dir+fig_name, bbox_inches='tight')