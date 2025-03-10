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

# mods
df_sae_metrics['k'] = df_sae_metrics['k'].fillna(0)
df_sae_metrics['k'] = df_sae_metrics['k'].astype(int)
df_sae_metrics['k'] = df_sae_metrics['k'].astype(str)
df_sae_metrics = df_sae_metrics[df_sae_metrics['k'] != '1']
df_sae_metrics['l1_weight'] = df_sae_metrics['l1_weight'].astype(str)
df_sae_metrics['fraction_unique'] = df_sae_metrics['n_unique'] / df_sae_metrics['n_activation_features']
df_sae_metrics['fraction_dead'] = 1 - df_sae_metrics['fraction_unique']

hidden_factor_options = [2, 5, 10, 20, 50, 100, 200, 1000]
lr_options = [1e-2, 1e-3, 1e-4, 1e-5]
l1_weight_options = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
k_options = [1, 5, 10, 20, 50, 75, 100]

df_lr = df_sae_metrics.copy()
#df_temp = df_temp[df_temp['k'] != '1']
df_lr = df_lr[df_lr['hidden_factor'] == 100]

df_sae_metrics = df_sae_metrics[df_sae_metrics['lr'] == 1e-4]

####################################################################################################
# set up figure
####################################################################################################

# a new line plot with 4 columns
# make the type categorical
df_sae_metrics['type'] = df_sae_metrics['type'].astype('category')
# set the order
df_sae_metrics['type'].cat.reorder_categories(['Vanilla', 'ReLU', 'TopK'], inplace=True)

variable_dict = {'highest_corrs_tf': 'X',
                'highest_corrs_rna': 'Y',
                'highest_corrs_activity': 'A',
                'highest_corrs_accessibility': 'B'}
df_corr_metrics = df_sae_metrics[['hidden_factor','type']]
corr_columns = [c for c in df_sae_metrics.columns if 'corrs' in c]
for i,c in enumerate(corr_columns):
    if i == 0:
        df_corr_metrics = df_sae_metrics[['hidden_factor','type']]
        df_corr_metrics['correlation'] = df_sae_metrics[c]
        df_corr_metrics['target'] = variable_dict[c]
    else:
        df_temp = df_sae_metrics[['hidden_factor','type']]
        df_temp['correlation'] = df_sae_metrics[c]
        df_temp['target'] = variable_dict[c]
        df_corr_metrics = pd.concat([df_corr_metrics, df_temp], axis=0)

n_cols = 4

model_palette = ['grey','lightseagreen','darkorange']
l1_palette = sns.color_palette('viridis_r', n_colors=len(l1_weight_options))
k_palette = sns.color_palette('rocket_r', n_colors=len(k_options))

# make the figure a grid of 2 rows and 4 columns
from matplotlib.gridspec import GridSpec
# set the font size
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(16, 3.5))
# add spacing
fig.subplots_adjust(wspace=0.5, hspace=0.7)
gs = GridSpec(2, n_cols, figure=fig)

# first subplot
ax0 = fig.add_subplot(gs[0:, 0])
# the first plot is the loss vs hidden colored by model type
x = 'n_activation_features'
y = 'loss'
y_label = 'MSE loss'
sns.lineplot(data=df_sae_metrics, x=x, y=y, ax=ax0, hue='type', style='type', markers=True, palette=model_palette)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_title('Loss vs hidden size')
ax0.set_xlabel('SAE hidden dimensionality')
ax0.set_ylabel(y_label)
ax0.get_legend().remove()
#ax0.legend(loc='center left', bbox_to_anchor=(-0.3, -0.3), ncol=3, frameon=False, title='SAE type', handletextpad=0.2, alignment='left', columnspacing=0.8)

ax0b = fig.add_subplot(gs[0:, 1])
# the first plot is the loss vs hidden colored by model type
x = 'lr'
y = 'loss'
y_label = 'MSE loss'
sns.lineplot(data=df_lr, x=x, y=y, ax=ax0b, hue='type', style='type', markers=True, palette=model_palette)
ax0b.set_xscale('log')
ax0b.set_yscale('log')
ax0b.set_title('Loss vs learning rate')
ax0b.set_xlabel('Learning rate')
ax0b.set_ylabel(y_label)
ax0b.get_legend().remove()

ax2 = fig.add_subplot(gs[0:, 2])
# plot the correlation of X vs hidden factor
x = 'n_activation_features'
y = 'highest_corrs_tf'
y_label = 'max(corr(neurons, X))'
sns.lineplot(data=df_sae_metrics, x=x, y=y, ax=ax2, hue='type', style='type', markers=True, palette=model_palette)
ax2.set_xscale('log')
ax2.set_title(y_label)
ax2.set_xlabel('SAE hidden dimensionality')
ax2.set_ylabel('Pearson correlation')
ax2.legend(loc='center left', bbox_to_anchor=(2.6, 0.5), ncol=1, frameon=False, title='SAE type', handletextpad=0.2, alignment='left', columnspacing=0.8)

ax1_a = fig.add_subplot(gs[0, 3])
ax1_b = fig.add_subplot(gs[1, 3])
# the second column gives the fraction of dead units vs l1 weight and k (need double axis)
x1 = 'l1_weight'
x2 = 'k'
y = 'fraction_dead'
y_label = 'Fraction of dead neurons'
#y_lims = [df_sae_metrics[y].min(), df_sae_metrics[y].max()]
y_lims = [0,1]
sns.lineplot(data=df_sae_metrics[df_sae_metrics['type'] != 'TopK'], x=x1, y=y, ax=ax1_a, hue='type', style='type', markers=True, palette=model_palette)
sns.lineplot(data=df_sae_metrics[df_sae_metrics['type'] == 'TopK'], x=x2, y=y, ax=ax1_b, hue='type', style='type', markers=True, palette=[model_palette[-1]])
ax1_a.set_title(y_label)
ax1_a.set_xlabel('L1 weight')
# change the x tick labels to scientific notation
ax1_a.set_xticklabels(['1e-1', '1e-2', '1e-3', '1e-4', '1e-5'])
ax1_b.set_xlabel('k (%)')
ax1_a.set_ylabel('dead / total')
ax1_b.set_ylabel('dead / total')
ax1_a.set_ylim(y_lims)
ax1_b.set_ylim(y_lims)
# remove the legends
ax1_a.get_legend().remove()
ax1_b.get_legend().remove()

plt.show()
# export to pdf
fig.savefig('03_results/figures/sim_modl1l4_sae_performance_metrics.pdf', bbox_inches='tight')

####################################################################################################

# a new line plot with 4 columns
# make the type categorical
df_sae_metrics['type'] = df_sae_metrics['type'].astype('category')
# set the order
df_sae_metrics['type'].cat.reorder_categories(['Vanilla', 'ReLU', 'TopK'], inplace=True)

variable_dict = {'highest_corrs_tf': 'X',
                'highest_corrs_rna': 'Y',
                'highest_corrs_activity': 'A',
                'highest_corrs_accessibility': 'B'}
df_corr_metrics = df_sae_metrics[['hidden_factor','type','n_activation_features']]
corr_columns = [c for c in df_sae_metrics.columns if 'corrs' in c]
for i,c in enumerate(corr_columns):
    if i == 0:
        df_corr_metrics = df_sae_metrics[['hidden_factor','type','n_activation_features']]
        df_corr_metrics['correlation'] = df_sae_metrics[c]
        df_corr_metrics['target'] = variable_dict[c]
    else:
        df_temp = df_sae_metrics[['hidden_factor','type','n_activation_features']]
        df_temp['correlation'] = df_sae_metrics[c]
        df_temp['target'] = variable_dict[c]
        df_corr_metrics = pd.concat([df_corr_metrics, df_temp], axis=0)

n_cols = 1

model_palette = ['grey','lightseagreen','darkorange']
variable_palette = sns.color_palette('colorblind', n_colors=4)
l1_palette = sns.color_palette('viridis_r', n_colors=len(l1_weight_options))
k_palette = sns.color_palette('rocket_r', n_colors=len(k_options))

# make the figure a grid of 2 rows and 4 columns
from matplotlib.gridspec import GridSpec
# set the font size
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(5, 4))
# add spacing
#fig.subplots_adjust(left=0.2, right=0.8)
gs = GridSpec(1, n_cols, figure=fig)


ax_list = []
ax_list.append(fig.add_subplot(gs[0, 0]))
x = 'n_activation_features'
y = 'correlation'
y_label = 'Pearson correlation'
hue = 'target'
sns.lineplot(data=df_corr_metrics, x=x, y=y, ax=ax_list[-1], hue=hue, style=hue, markers=True, palette=variable_palette)
ax_list[-1].set_xscale('log')
ax_list[-1].set_xlabel('SAE hidden dimensionality')
ax_list[-1].set_ylabel('Pearson correlation')
ax_list[-1].set_title('max(corr(neurons, variables))')
ax_list[-1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.05), ncol=1, frameon=False, title='Variable', handletextpad=0.2, alignment='left', columnspacing=0.8)

plt.show()
# export to pdf
fig.savefig('03_results/figures/sim_modl1l4_sae_recovery_metrics_a.pdf', bbox_inches='tight')

####################################################################################################

n_cols=2
# set the font size
plt.rcParams.update({'font.size': 15})
fig = plt.figure(figsize=(8, 3))
# add spacing
fig.subplots_adjust(wspace=1.2, hspace=0.7)
gs = GridSpec(1, n_cols, figure=fig)


ax_list = []

ax_list.append(fig.add_subplot(gs[0, 0]))
# plot the number of active units per variable
x = 'n_activation_features'
y = 'n_per_tf'
y_label = 'Active neurons per X\n(Vanilla, ReLU)'
sns.lineplot(data=df_sae_metrics[df_sae_metrics['type'] != 'TopK'], x=x, y=y, ax=ax_list[-1], hue='l1_weight', style='l1_weight', markers=True, palette=l1_palette)
ax_list[-1].set_xscale('log')
ax_list[-1].set_yscale('log')
ax_list[-1].set_title(y_label)
ax_list[-1].set_xlabel('SAE hidden dimensionality')
ax_list[-1].set_ylabel('N active neurons')
ax_list[-1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.05), ncol=1, frameon=False, title='L1 weight', handletextpad=0.2, alignment='left', columnspacing=0.8)

ax_list.append(fig.add_subplot(gs[0, 1]))
#y = 'n_per_rna'
#y_label = 'N active units per Y'
y_label = 'Active neurons per X\n(TopK)'
sns.lineplot(data=df_sae_metrics[df_sae_metrics['type'] == 'TopK'], x=x, y=y, ax=ax_list[-1], hue='k', style='k', markers=True, palette=k_palette)
ax_list[-1].set_xscale('log')
ax_list[-1].set_yscale('log')
ax_list[-1].set_title(y_label)
ax_list[-1].set_xlabel('SAE hidden dimensionality')
ax_list[-1].set_ylabel('N active neurons')
ax_list[-1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.05), ncol=1, frameon=False, title='k', handletextpad=0.2, alignment='left', columnspacing=0.8)
#ax_list[-1].legend(loc='center left', bbox_to_anchor=(1.0, 1.0), ncol=1, frameon=False, title='SAE type', handletextpad=0.2, alignment='left', columnspacing=0.8)

plt.show()
# export to pdf
fig.savefig('03_results/figures/sim_modl1l4_sae_recovery_metrics_b.pdf', bbox_inches='tight')