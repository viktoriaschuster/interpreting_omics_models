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
df_sae_metrics['fraction_unique'] = df_sae_metrics['n_unique'] / df_sae_metrics['n_activation_features']

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
fig_name = 'sim_sae_performance1.pdf'

plt.rcParams.update({'font.size': figure_kwargs['font_size']})

loss_colors = sns.color_palette(n_colors=2)
superposition_colors = sns.color_palette('colorblind', 3)
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# set up a figure with grid: 3 rows, 5 columns
fig = plt.figure(figsize=(figure_kwargs['fig_width'], figure_kwargs['fig_height']))
gs = GridSpec(3, 3, figure=fig)
# add spacing
gs.update(wspace=figure_kwargs['wspace'], hspace=figure_kwargs['hspace'])
ax_list = []

###############

i = 0
# plot the loss over the different hyperparameters
y = 'loss'
x = 'n_activation_features'
hue = 'lr'
# print how many points per bar
print('Number of points per bar:')
print(df_sae_metrics.groupby(['type', x, hue]).size())
lr_palette = sns.color_palette('viridis', n_colors=len(lr_options))
for j, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    ax = fig.add_subplot(gs[i, j])
    sns.barplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, ax=ax, palette=lr_palette, err_kws={'linewidth': figure_kwargs['errorbar_linewidth']}, capsize=figure_kwargs['error_capsize'])
    ax.set_ylim(1e-8, df_sae_metrics['loss'].max())
    ax.set_yscale('log')
    ax.set_title(t)
    # turn the x-axis labels by 90 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('SAE hidden dimensionality')
    ax.set_ylabel('MSE loss')
    if j < 2:
        ax.get_legend().remove()
    else:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='Learning rate', alignment='left')

i = 1
hue = 'l1_weight'
print('Number of points per bar:')
print(df_sae_metrics.groupby(['type', x, hue]).size())
cmap_palette = sns.color_palette('viridis', n_colors=max(len(l1_weight_options), len(k_options)))
for j, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    ax = fig.add_subplot(gs[i, j])
    if t == 'TopK':
        hue = 'k'
    sns.barplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, ax=ax, palette=cmap_palette, err_kws={'linewidth': figure_kwargs['errorbar_linewidth']}, capsize=figure_kwargs['error_capsize'])
    ax.set_ylim(1e-10, df_sae_metrics['loss'].max())
    ax.set_yscale('log')
    ax.set_title(t)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('SAE hidden dimensionality')
    ax.set_ylabel('MSE loss')
    if t == 'TopK':
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='k (in percent)', alignment='left')
    else:
        if j == 0:
            ax.legend().remove()
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='L1 weight', alignment='left')

i = 2
y = 'fraction_unique'
hue = 'l1_weight'
print('Number of points per bar:')
print(df_sae_metrics.groupby(['type', x, hue]).size())
for j, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    ax = fig.add_subplot(gs[i, j])
    if t == 'TopK':
        hue = 'k'
    sns.barplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, ax=ax, palette=cmap_palette, err_kws={'linewidth': figure_kwargs['errorbar_linewidth']}, capsize=figure_kwargs['error_capsize'])
    ax.set_ylim(0, 1.02)
    ax.set_title(t)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('SAE hidden dimensionality')
    ax.set_ylabel('Fraction of active neurons')
    if t == 'TopK':
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='k (in percent)', alignment='left')
    else:
        if j == 0:
            ax.legend().remove()
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='L1 weight', alignment='left')

###############

#plt.tight_layout()
plt.savefig(fig_dir+fig_name, bbox_inches='tight')

####################################################################################################

# low look for only one hidden dimensionality to compare learning rates
df_sae_temp = df_sae_metrics[df_sae_metrics['hidden_factor'] == 100]

y = 'fraction_unique'
hue = 'lr'
x = 'l1_weight'
fig_name = 'sim_sae_performance1_lr.pdf'

fig = plt.figure(figsize=(16, 3))
gs = GridSpec(1, 3, figure=fig)
# add spacing
gs.update(wspace=0.4)
ax_list = []

for j, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    ax = fig.add_subplot(gs[0, j])
    if t == 'TopK':
        x = 'k'
    sns.barplot(data=df_sae_temp[df_sae_temp['type'] == t], x=x, y=y, hue=hue, ax=ax, palette=lr_palette, err_kws={'linewidth': figure_kwargs['errorbar_linewidth']}, capsize=figure_kwargs['error_capsize'])
    ax.set_ylim(0, 1.02)
    ax.set_title(t)
    if t == 'TopK':
        ax.set_xlabel('k (in percent)')
    else:
        # set the x tick labels as scientific (1e-...)
        ax.set_xticklabels(['{:.0e}'.format(float(x)) for x in l1_weight_options])
        ax.set_xlabel('L1 weight')
    ax.set_ylabel('Fraction of active neurons')
    if j == 2:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='Learning rate', alignment='left')
    else:
        ax.legend().remove()
plt.savefig(fig_dir+fig_name, bbox_inches='tight')