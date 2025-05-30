import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
# set all random seeds to 0
torch.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)

# get activations
activations = torch.load('03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')
# all active features
active_features = (torch.where(torch.sum(activations, dim=0) != 0)[0]).tolist()

# get go_df
go_df = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis.csv')
go_df['go type'] = 'biological process'
go_df2 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_large.csv')
go_df2['go type'] = 'biological process'
go_df3 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_molecular_function.csv')
go_df3['go type'] = 'molecular function'
go_df = pd.concat([go_df, go_df2, go_df3], axis=0, ignore_index=True)
from goatools.obo_parser import GODag
obodag = GODag("01_data/go-basic.obo")

# create a matrix of all go terms and features
go_feature_matrix = torch.zeros((len(go_df['go_name'].unique()), activations.shape[1]))
for i, go_id in enumerate(go_df['go_id'].unique()):
    for feat in go_df[go_df['go_id'] == go_id]['feature']:
        go_feature_matrix[i,feat] = 1
# remove all columns with zero sum (unused features)
mtrx_go_ids = go_df['go_id'].unique()
mtrx_go_names = [obodag[x].name for x in mtrx_go_ids]
mtrx_feature_ids = torch.where(go_feature_matrix.sum(dim=0) > 0)[0]
go_feature_matrix = go_feature_matrix[:,torch.where(go_feature_matrix.sum(dim=0) > 0)[0]]

# create a UMAP of the go terms
import umap
reducer = umap.UMAP(n_components=2, min_dist=0.6, n_neighbors=20, random_state=0)
embedding = reducer.fit_transform(go_feature_matrix.T)
df_umap = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
df_umap['feature'] = mtrx_feature_ids.numpy()

###################################

# plot the UMAP

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=df_umap, x='UMAP 1', y='UMAP 2', s=2, alpha=0.7, ec=None)
plt.show()

import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'legend.fontsize': 'small'})

n_cols = 15
n_rows = 5

legend_x = 0.3
legend_y = 1
handletextpad = 0.1
markerscale = 2

fig = plt.figure(figsize=(16, 8))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(n_rows, n_cols, wspace=1.0, hspace=1.2)
ax_list = []

# the first row are umaps and span over 2 rows
ax_list.append(plt.subplot(gs[0:3, 0:4]))
# add "A" on the grid
ax_list[0].text(-0.14, 1.1, 'A', transform=ax_list[0].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
# here color by number of cells
n_cells_per_feature = go_feature_matrix.sum(dim=0).numpy()
df_umap['n_cells'] = n_cells_per_feature
sns.scatterplot(data=df_umap, x='UMAP 1', y='UMAP 2', hue='n_cells', s=3, alpha=0.7, ec=None, ax=ax_list[0])
# remove all ticks
ax_list[0].set_xticks([])
ax_list[0].set_yticks([])
# move the legend outside
# keep only the first and last legend handles and labels
handles, labels = ax_list[0].get_legend_handles_labels()
handles = [handles[0], handles[-1]]
labels = [labels[0], labels[-1]]
ax_list[0].legend(handles, labels, loc='upper right', bbox_to_anchor=(legend_x, legend_y), frameon=False, handletextpad=handletextpad, markerscale=markerscale)
ax_list[0].set_title('Number of active cells')

# next plot a umap colored by feature type
ax_list.append(plt.subplot(gs[0:3, 4:8]))
sns.scatterplot(data=df_umap, x='UMAP 1', y='UMAP 2', hue='feature_type', s=3, alpha=0.7, ec=None, ax=ax_list[1])
ax_list[1].set_xticks([])
ax_list[1].set_yticks([])
ax_list[1].set_xlabel('')
ax_list[1].set_ylabel('')
ax_list[1].legend(loc='upper right', bbox_to_anchor=(legend_x+0.05, legend_y), frameon=False, handletextpad=handletextpad, markerscale=markerscale)
ax_list[1].set_title('Feature type')

# next plot a umap colored by level1 go term
ax_list.append(plt.subplot(gs[0:3, 8:12]))
# color each feature by its most prevalent level 1 go term
level_1_mtrx_terms = []
for feat in mtrx_feature_ids:
    feat = feat.item()
    temp_go_term = go_df[go_df['feature'] == feat]['parent_go_name_level1'].value_counts().index[0]
    level_1_mtrx_terms.append(temp_go_term)
df_umap['level_1_go_term'] = level_1_mtrx_terms
level1_paletteb = sns.color_palette('Paired')
level1_palettea = sns.color_palette('Set2_r')
level1_palettea = level1_palettea[:2] + level1_palettea[3:]
level1_palette = level1_palettea + level1_paletteb
level1_palette = level1_palette[:len(df_umap['level_1_go_term'].unique())]
sns.scatterplot(data=df_umap, x='UMAP 1', y='UMAP 2', hue='level_1_go_term', s=3, alpha=1, ec=None, palette=level1_palette, ax=ax_list[2])
ax_list[2].set_xticks([])
ax_list[2].set_yticks([])
ax_list[2].set_xlabel('')
ax_list[2].set_ylabel('')
ax_list[2].set_title('Level 1 GO term')
#ax_list[2].legend().remove()
# wrap the text for the legend
handles, labels = ax_list[2].get_legend_handles_labels()
labels = [textwrap.fill(label, 20) for label in labels]
ax_list[2].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.8, legend_y), frameon=False, handletextpad=handletextpad, markerscale=markerscale, ncol=1)

binary_colors = sns.color_palette('rocket_r', 100)
binary_colors = [binary_colors[0], binary_colors[75]]

ax_list.append(plt.subplot(gs[3:5, 0:3]))
ax_list[-1].text(-0.2, 1.15, 'B', transform=ax_list[-1].transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
term_snippet = 'biosynthetic process'
test_go_terms = np.asarray([1 if term_snippet in x else 0 for x in mtrx_go_names]).astype(bool)
test_counts = go_feature_matrix[test_go_terms,:].clone().sum(dim=0).int()
test_counts = (test_counts > 0).float()
df_temp = df_umap.copy()
df_temp[term_snippet] = test_counts.numpy()
sns.scatterplot(data=df_temp, x='UMAP 1', y='UMAP 2', hue=term_snippet, s=2, alpha=0.7, ec=None, palette=binary_colors, ax=ax_list[-1])
ax_list[-1].set_xticks([])
ax_list[-1].set_yticks([])
ax_list[-1].set_title('"{}"'.format(term_snippet))
# remove the legend
ax_list[-1].legend().remove()

ax_list.append(plt.subplot(gs[3:5, 3:6]))
term_snippet = 'morphogenesis'
test_go_terms = np.asarray([1 if term_snippet in x else 0 for x in mtrx_go_names]).astype(bool)
test_counts = go_feature_matrix[test_go_terms,:].clone().sum(dim=0).int()
test_counts = (test_counts > 0).float()
df_temp = df_umap.copy()
df_temp[term_snippet] = test_counts.numpy()
sns.scatterplot(data=df_temp, x='UMAP 1', y='UMAP 2', hue=term_snippet, s=2, alpha=0.7, ec=None, palette=binary_colors, ax=ax_list[-1])
ax_list[-1].set_xticks([])
ax_list[-1].set_yticks([])
ax_list[-1].set_xlabel('')
ax_list[-1].set_ylabel('')
ax_list[-1].set_title('"{}"'.format(term_snippet))
# remove the legend
ax_list[-1].legend().remove()

ax_list.append(plt.subplot(gs[3:5, 6:9]))
term_snippet = 'ion homeostasis'
test_go_terms = np.asarray([1 if term_snippet in x else 0 for x in mtrx_go_names]).astype(bool)
test_counts = go_feature_matrix[test_go_terms,:].clone().sum(dim=0).int()
test_counts = (test_counts > 0).float()
df_temp = df_umap.copy()
df_temp[term_snippet] = test_counts.numpy()
sns.scatterplot(data=df_temp, x='UMAP 1', y='UMAP 2', hue=term_snippet, s=2, alpha=0.7, ec=None, palette=binary_colors, ax=ax_list[-1])
ax_list[-1].set_xticks([])
ax_list[-1].set_yticks([])
ax_list[-1].set_xlabel('')
ax_list[-1].set_ylabel('')
ax_list[-1].set_title('"{}"'.format(term_snippet))
# remove the legend
ax_list[-1].legend().remove()

ax_list.append(plt.subplot(gs[3:5, 9:12]))
term_snippet = 'antigen'
test_go_terms = np.asarray([1 if term_snippet in x else 0 for x in mtrx_go_names]).astype(bool)
test_counts = go_feature_matrix[test_go_terms,:].clone().sum(dim=0).int()
test_counts = (test_counts > 0).float()
df_temp = df_umap.copy()
df_temp[term_snippet] = test_counts.numpy()
sns.scatterplot(data=df_temp, x='UMAP 1', y='UMAP 2', hue=term_snippet, s=2, alpha=0.7, ec=None, palette=binary_colors, ax=ax_list[-1])
ax_list[-1].set_xticks([])
ax_list[-1].set_yticks([])
ax_list[-1].set_xlabel('')
ax_list[-1].set_ylabel('')
ax_list[-1].set_title('"{}"'.format(term_snippet))
# remove the legend
ax_list[-1].legend().remove()

# save as a pdf
plt.savefig('03_results/figures/sc_automatic_feature_analysis_2.pdf', bbox_inches='tight')
plt.show()