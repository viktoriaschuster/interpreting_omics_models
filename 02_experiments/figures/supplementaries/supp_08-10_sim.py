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
    'fig_height': 8,
    'font_size': 14,
    'point_size': 50,
    'line_width_background': 0.5,
    'wspace': 1.0,
    'hspace': 0.7,
    'errorbar_linewidth': 1.0,
    'error_capsize': 0.3,
}
fig_dir = '03_results/figures/supplementaries/'
fig_name = 'sim_sae_performance4.pdf'

plt.rcParams.update({'font.size': figure_kwargs['font_size']})
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

###############

# I want to plot n_per_tf against highest_corrs_tf

x = 'n_per_tf'
y = 'highest_corrs_tf'
style = 'l1_weight'
hue = 'n_activation_features'

cmap_palette = sns.color_palette('viridis', n_colors=len(hidden_factor_options))

fig, axs = plt.subplots(2, 3, figsize=(figure_kwargs['fig_width'], figure_kwargs['fig_height']))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
j = 0
for i, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    if t == 'TopK':
        style = 'k'
    # put in a grey line at y=0.95
    x_min, x_max = 1, 10
    y_min, y_max = 0.95, 1.0
    axs[j,i].fill_between([x_min, x_max], [y_min, y_min], [y_max, y_max], color='lightslategray', alpha=0.3)
    #sns.scatterplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, style=style, ax=axs[i], palette=cmap_palette)
    sns.scatterplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, style=style, ax=axs[j,i], palette=cmap_palette, markers=True, s=figure_kwargs['point_size'])
    #axs[i].set_xlim(min(df_sae_metrics[y]), max(df_sae_metrics[y]))
    axs[j,i].set_ylim(0.3, 1.02)
    axs[j,i].set_xscale('log')
    axs[j,i].set_title(t)
    axs[j,i].set_ylabel('Highest correlation with X')
    axs[j,i].set_xlabel('N active neurons per X')
    #axs[j,i].legend(loc='upper left', bbox_to_anchor=(1, 0.7), frameon=False)
    axs[j,i].get_legend().remove()
j = 1
style = 'l1_weight'

for i, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    if t == 'TopK':
        style = 'k'
    # put in a grey line at y=0.95
    x_min, x_max = 1, 10
    y_min, y_max = 0.948, 1.002
    # set the background color to grey
    axs[j,i].fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], color='lightslategray', alpha=0.2)
    sns.scatterplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, style=style, ax=axs[j,i], palette=cmap_palette, markers=True, s=figure_kwargs['point_size'])
    #axs[i].set_xlim(min(df_sae_metrics[y]), max(df_sae_metrics[y]))
    axs[j,i].set_ylim(y_min, y_max)
    axs[j,i].set_xlim(x_min, x_max)
    axs[j,i].set_title(t)
    axs[j,i].set_ylabel('Highest correlation with X')
    axs[j,i].set_xlabel('N active neurons per X')
    # make sure the x ticks stay integers and start at 1
    axs[j,i].set_xticks(np.arange(1, 11, 1))
    #if t == 'TopK':
    #    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='k (in percent)')
    #else:
    #    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='L1 weight')
    #axs[j,i].get_legend().remove()
    if i == 1:
        handles, labels = axs[j,i].get_legend_handles_labels()
        stop_index = np.where(np.array(labels) == style)[0][0]
        handles = handles[1:stop_index]
        labels = labels[1:stop_index]
        axs[j,i].legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(-1.1, -0.3), frameon=False, ncol=8, title='SAE hidden dimensionality', alignment='center')
    elif i == 0:
        handles, labels = axs[j,i].get_legend_handles_labels()
        start_index = np.where(np.array(labels) == style)[0][0]
        handles = handles[(start_index+1):]
        labels = labels[(start_index+1):]
        axs[j,i].legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(-0.3, -0.6), frameon=False, ncol=5, title='L1 weight', alignment='left')
    else:
        handles, labels = axs[j,i].get_legend_handles_labels()
        start_index = np.where(np.array(labels) == style)[0][0]
        handles = handles[(start_index+1):]
        labels = labels[(start_index+1):]
        axs[j,i].legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(-0.4, -0.6), frameon=False, ncol=4, title='k (%)', alignment='right')


plt.savefig(fig_dir+fig_name, bbox_inches='tight')

###############

fig_name = 'sim_sae_performance5.pdf'

# I want to plot n_per_tf against highest_corrs_tf

x = 'n_per_rna'
y = 'highest_corrs_rna'
style = 'l1_weight'
hue = 'n_activation_features'

cmap_palette = sns.color_palette('viridis', n_colors=len(hidden_factor_options))

fig, axs = plt.subplots(2, 3, figsize=(figure_kwargs['fig_width'], figure_kwargs['fig_height']))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
j = 0
for i, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    if t == 'TopK':
        style = 'k'
    # put in a grey line at y=0.95
    x_min, x_max = 1, 10
    y_min, y_max = 0.95, 1.0
    axs[j,i].fill_between([x_min, x_max], [y_min, y_min], [y_max, y_max], color='lightslategray', alpha=0.3)
    #sns.scatterplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, style=style, ax=axs[i], palette=cmap_palette)
    sns.scatterplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, style=style, ax=axs[j,i], palette=cmap_palette, markers=True, s=figure_kwargs['point_size'])
    #axs[i].set_xlim(min(df_sae_metrics[y]), max(df_sae_metrics[y]))
    axs[j,i].set_ylim(0.3, 1.02)
    axs[j,i].set_xscale('log')
    axs[j,i].set_title(t)
    axs[j,i].set_ylabel('Highest correlation with Y')
    axs[j,i].set_xlabel('N active neurons per Y')
    #axs[j,i].legend(loc='upper left', bbox_to_anchor=(1, 0.7), frameon=False)
    axs[j,i].get_legend().remove()
j = 1
style = 'l1_weight'

for i, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    if t == 'TopK':
        style = 'k'
    # put in a grey line at y=0.95
    x_min, x_max = 1, 10
    y_min, y_max = 0.948, 1.002
    # set the background color to grey
    axs[j,i].fill([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max], color='lightslategray', alpha=0.2)
    sns.scatterplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, style=style, ax=axs[j,i], palette=cmap_palette, markers=True, s=figure_kwargs['point_size'])
    #axs[i].set_xlim(min(df_sae_metrics[y]), max(df_sae_metrics[y]))
    axs[j,i].set_ylim(y_min, y_max)
    axs[j,i].set_xlim(x_min, x_max)
    axs[j,i].set_title(t)
    axs[j,i].set_ylabel('Highest correlation with Y')
    axs[j,i].set_xlabel('N active neurons per Y')
    # make sure the x ticks stay integers and start at 1
    axs[j,i].set_xticks(np.arange(1, 11, 1))
    #if t == 'TopK':
    #    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='k (in percent)')
    #else:
    #    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='L1 weight')
    #axs[j,i].get_legend().remove()
    if i == 1:
        handles, labels = axs[j,i].get_legend_handles_labels()
        stop_index = np.where(np.array(labels) == style)[0][0]
        handles = handles[1:stop_index]
        labels = labels[1:stop_index]
        axs[j,i].legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(-1.1, -0.3), frameon=False, ncol=8, title='SAE hidden dimensionality', alignment='center')
    elif i == 0:
        handles, labels = axs[j,i].get_legend_handles_labels()
        start_index = np.where(np.array(labels) == style)[0][0]
        handles = handles[(start_index+1):]
        labels = labels[(start_index+1):]
        axs[j,i].legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(-0.3, -0.6), frameon=False, ncol=5, title='L1 weight', alignment='left')
    else:
        handles, labels = axs[j,i].get_legend_handles_labels()
        start_index = np.where(np.array(labels) == style)[0][0]
        handles = handles[(start_index+1):]
        labels = labels[(start_index+1):]
        axs[j,i].legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(-0.4, -0.6), frameon=False, ncol=4, title='k (%)', alignment='right')

plt.savefig(fig_dir+fig_name, bbox_inches='tight')

###############

fig_name = 'sim_sae_performance6.pdf'

df_sae_metrics['hidden_dim_cat'] = df_sae_metrics['n_activation_features'].astype(str)

x = 'hidden_dim_cat'
y = 'highest_corrs_activity'
hue = 'l1_weight'

l1_palette = sns.color_palette('viridis_r', n_colors=len(l1_weight_options))
k_palette = sns.color_palette('rocket_r', n_colors=len(k_options))

fig, axs = plt.subplots(2, 3, figsize=(figure_kwargs['fig_width'], 8))
plt.subplots_adjust(wspace=0.5, hspace=0.7)
j = 0
for i, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    if t == 'TopK':
        hue = 'k'
        palette = k_palette
    else:
        palette = l1_palette
    sns.scatterplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, ax=axs[j,i], palette=palette, markers=True)
    axs[j,i].set_title(t)
    axs[j,i].set_ylim(0.8, 0.9)
    axs[j,i].set_ylabel('Highest correlation with A')
    axs[j,i].set_xticklabels(axs[j,i].get_xticklabels(), rotation=90)
    axs[j,i].set_xlabel('N hidden neurons')
    axs[j,i].get_legend().remove()

j = 1
hue = 'l1_weight'
for i, t in enumerate(['Vanilla', 'ReLU', 'TopK']):
    if t == 'TopK':
        hue = 'k'
        palette = k_palette
    else:
        palette = l1_palette
    sns.scatterplot(data=df_sae_metrics[df_sae_metrics['type'] == t], x=x, y=y, hue=hue, ax=axs[j,i], palette=palette, markers=True)
    axs[j,i].set_title(t)
    axs[j,i].set_ylim(0.8, 0.9)
    axs[j,i].set_ylabel('Highest correlation with B')
    axs[j,i].set_xticklabels(axs[j,i].get_xticklabels(), rotation=90)
    axs[j,i].set_xlabel('N hidden neurons')
    # if i == 0, put the legend under the plot
    if i == 0:
        axs[j,i].legend(loc='lower left', bbox_to_anchor=(-0.3, -0.7), frameon=False, title='L1 weight', ncol=len(l1_weight_options), alignment='left',handletextpad=0.1,columnspacing=0.8)
    elif i == 2:
        axs[j,i].legend(loc='lower left', bbox_to_anchor=(-0.6, -0.7), frameon=False, title='K', ncol=len(k_options), alignment='right',handletextpad=0.1,columnspacing=0.8)
    else:
        axs[j,i].get_legend().remove()

plt.savefig(fig_dir+fig_name, bbox_inches='tight')