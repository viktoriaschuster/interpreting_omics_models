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

# make the figure a grid of 2 rows and 4 columns
from matplotlib.gridspec import GridSpec
# set the font size
plt.rcParams.update({'font.size': 14})
fig = plt.figure(figsize=(8, 4))
# add spacing
#fig.subplots_adjust(left=0.2, right=0.8)
gs = GridSpec(1, 2, figure=fig)

####################################################################################################
# small sim
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

#################################

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

#plt.show()
# export to pdf
#fig.savefig('03_results/figures/sim_modl1l4_sae_recovery_metrics_a.pdf', bbox_inches='tight')

####################################################################################################
# large sim
####################################################################################################
colors = ['#b14c8d','#0044ad','#5176e8','#82A0FD','#417a78','#c5a862']
#variables = ['Y', 'X₂', 'X₁', 'X₀', 'A', 'B']
variables = ['Y', "X''", "X'", 'X', 'A', 'B']

# load the evaluation metrics

results_dir = '03_results/reports/files/sim2L_sae_metrics_pearson_latent-'
latents = [20, 100, 150]

metrics = []
for latent in latents:
    metrics.append(pd.read_csv(results_dir + str(latent) + '.csv'))
metrics = pd.concat(metrics)
metrics = metrics.reset_index(drop=True)

def transform_str_to_avg(x):
    items = [y for y in x.strip('[]').split(' ')]
    # remove all '' elements
    items = [x for x in items if x != '']
    # transform to float
    items = np.asarray([float(x) for x in items])
    return np.max(items)

metrics['highest_corr_x0'] = [transform_str_to_avg(x) for x in metrics['highest_corr_x0'].values]
metrics['highest_corr_x1'] = [transform_str_to_avg(x) for x in metrics['highest_corr_x1'].values]
metrics['highest_corr_x2'] = [transform_str_to_avg(x) for x in metrics['highest_corr_x2'].values]
metrics['highest_corr_ct'] = [transform_str_to_avg(x) for x in metrics['highest_corr_ct'].values]
metrics['highest_corr_co'] = [transform_str_to_avg(x) for x in metrics['highest_corr_co'].values]

metrics_corr = metrics[['latent_dim', 'hidden_factor', 'highest_corr_x0', 'highest_corr_x1', 'highest_corr_x2', 'highest_corr_ct', 'highest_corr_co']]
# transform into long format to get highest correlation per variable
metrics_corr = pd.melt(metrics_corr, id_vars=['latent_dim', 'hidden_factor'], var_name='variable', value_name='highest_corr')
metrics_corr = metrics_corr.sort_values(by=['latent_dim', 'variable'])

# rename the variables (keep the last part of the string)
metrics_corr['variable'] = metrics_corr['variable'].str.split('_').str[-1]
metrics_corr['variable'] = metrics_corr['variable'].replace({'y': 'Y', 'x2': "X''", 'x1': "X'", 'x0': 'X', 'ct': 'A', 'co': 'B'})
# make them categorical with the order of the variables
metrics_corr['variable'] = pd.Categorical(metrics_corr['variable'], categories=variables, ordered=True)
# set the hidden factor to a categorical string
metrics_corr = metrics_corr.sort_values(by=['hidden_factor'])
metrics_corr['hidden_factor'] = metrics_corr['hidden_factor'].astype(str)
# rename latent dim to Latent dimension
metrics_corr = metrics_corr.rename(columns={'latent_dim': 'Latent\ndimension'})


# plot the variable against n_activation_features and color by latent dim
#fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax_list.append(fig.add_subplot(gs[0, 0]))
sns.lineplot(data=metrics_corr, x='hidden_factor', y='highest_corr', hue='variable', style='variable', palette=colors, ax=ax_list[-1], linewidth=1.0, legend=False)
sns.scatterplot(data=metrics_corr, x='hidden_factor', y='highest_corr', hue='variable', style='Latent\ndimension', palette=colors, ax=ax_list[-1], s=20)
ax_list[-1].set_ylim(-0.6, 1.05)
ax_list[-1].set_xlabel('SAE hidden scaling factor')
ax_list[-1].set_ylabel('Pearson correlation')
ax_list[-1].set_title('max(corr(neuron, variable))')
# set the ticks to the actual x values
#plt.xticks(metrics_corr['hidden_factor'].unique())
#plt.xscale('log')
#plt.tight_layout()
# keep only the style legend
ax_list[-1].legend(title='Latent dim', loc='upper left', bbox_to_anchor=(1, 1)).remove()
handles, labels = ax_list[-1].get_legend_handles_labels()
# remove the first 6 elements from the handles and labels
handles = handles[8:]
labels = labels[8:]
# create the legend and reduce the distance between markers and text
ax_list[-1].legend(handles, labels, loc='upper left', bbox_to_anchor=(0.95, 1.05), markerscale=2, handletextpad=0.01, frameon=False, title='Latent')

fig.savefig('03_results/figures/simL_variable_corr.pdf', bbox_inches='tight')