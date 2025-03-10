import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import torch

activations = torch.load('03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')
active_features = (torch.where(torch.sum(activations, dim=0) != 0)[0]).tolist()
cos_sim_activ = torch.load('03_results/reports/cosine_similarity_activations_active_features.pt')

go_df = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis.csv')
go_df['go type'] = 'biological process'
go_df2 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_large.csv')
go_df2['go type'] = 'biological process'
go_df3 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_molecular_function.csv')
go_df3['go type'] = 'molecular function'
go_df = pd.concat([go_df, go_df2, go_df3], axis=0, ignore_index=True)

from goatools.obo_parser import GODag
obodag = GODag("01_data/go-basic.obo")
go_feature_matrix = torch.zeros((len(go_df['go_name'].unique()), activations.shape[1]))
for i, go_id in enumerate(go_df['go_id'].unique()):
    for feat in go_df[go_df['go_id'] == go_id]['feature']:
        go_feature_matrix[i,feat] = 1
# remove all columns with zero sum (unused features)
mtrx_go_ids = go_df['go_id'].unique()
mtrx_go_names = [obodag[x].name for x in mtrx_go_ids]
mtrx_feature_ids = torch.where(go_feature_matrix.sum(dim=0) > 0)[0]
go_feature_matrix = go_feature_matrix[:,torch.where(go_feature_matrix.sum(dim=0) > 0)[0]]

reducer = umap.UMAP(n_components=2, min_dist=0.6, n_neighbors=20, random_state=0)
embedding = reducer.fit_transform(go_feature_matrix.T)

df_umap = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
df_umap['feature'] = mtrx_feature_ids.numpy()

###
# plot 1
###

# make a histogram

# give every go term an index
go_ids = go_df['go_id'].unique()
go_id_dict = {go_ids[i]: i for i in range(len(go_ids))}
go_df['go_idx'] = go_df['go_id'].map(go_id_dict)
# make a dataframe with the idx and count
go_counts = go_df['go_idx'].value_counts().reset_index()

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
# add spacing
fig.subplots_adjust(wspace=0.2)
sns.histplot(data=go_df.sort_values('feature_type'), x='go_name', hue='feature_type', ax=ax[0], alpha=0.5)
ax[0].set_xticks(range(0, len(go_ids), 200))
ax[0].set_xticklabels(range(0, len(go_ids), 200))
ax[0].set_ylim(0, 1500)
sns.histplot(data=go_df, x='go_name', hue='go type', ax=ax[1])
# set the indices of go_idx as xticks (every 1000)
ax[1].set_ylim(0, 1500)
ax[1].set_xticks(range(0, len(go_ids), 200))
ax[1].set_xticklabels(range(0, len(go_ids), 200))
#plt.xticks([])
plt.show()

###
# plot 2
###

level1_paletteb = sns.color_palette('Paired')
level1_palettea = sns.color_palette('Set2_r')
level1_palettea = level1_palettea[:2] + level1_palettea[3:]
level1_palette = level1_palettea + level1_paletteb
level1_palette = level1_palette[:len(df_umap['level_1_go_term'].unique())]

reducer = umap.UMAP(n_components=2, min_dist=0.5, n_neighbors=15)
embedding = reducer.fit_transform(cos_sim_activ)

df_umap_cos = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
df_umap_cos['feature'] = active_features

# set the legend font size to small
plt.rcParams.update({'legend.fontsize': 'small'})

# color each feature by its most prevalent level 1 go term
level_1_mtrx_terms = []

for feat in active_features:
    feat = feat
    try:
        temp_go_term = go_df[go_df['feature'] == feat]['parent_go_name_level1'].value_counts().index[0]
    except:
        temp_go_term = 'None'
    level_1_mtrx_terms.append(temp_go_term)

df_umap_cos['level_1_go_term'] = level_1_mtrx_terms

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=df_umap_cos, x='UMAP 1', y='UMAP 2', hue='level_1_go_term', s=2, alpha=1, ec=None, palette=level1_palette)
# if a legend handle is longer than 20 characters, break it up
handles, labels = ax.get_legend_handles_labels()
import textwrap
new_labels = [textwrap.fill(label, 20) for label in labels]
# also make the legend markers larger by scaling
ax.legend(handles, new_labels, loc='upper right', bbox_to_anchor=(2.25, 0.95), markerscale=3, frameon=False, handletextpad=0.2, ncol=2)
# remove ticks
ax.set_xticks([])
ax.set_yticks([])
plt.show()

###
# plot 3
###

# show me where feature 2306 is
special_features = [2306, 1238, 5205, 1500]

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=df_umap, x='UMAP 1', y='UMAP 2', s=2, alpha=0.7, ec=None, ax=ax)
for feat in special_features:
    temp_df = df_umap[df_umap['feature'] == feat]
    sns.scatterplot(data=temp_df, x='UMAP 1', y='UMAP 2', s=20, ec='black', ax=ax)
    # write the feature number next to the point
    for i, row in temp_df.iterrows():
        ax.text(row['UMAP 1']+0.5, row['UMAP 2']+0.5, row['feature'], fontsize=8, color='black')
plt.show()