import torch
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import textwrap

import sys
sys.path.append(".")
sys.path.append('src')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cuda:1'

###
# load results
###

# multiDGD GO terms
# look at the dataframe with GO term analysis
go_df_1 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_small_2_biological_process.csv')
go_df_1['go type'] = 'biological process'
go_df2 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_large_2_biological_process.csv')
go_df2['go type'] = 'biological process'
go_df3 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_small_2_molecular_function.csv')
go_df3['go type'] = 'molecular function'
go_df4 = pd.read_csv('03_results/reports/sc_dgd_sae_go_analysis_large_2_molecular_function.csv')
go_df4['go type'] = 'molecular function'
go_df_dgd = pd.concat([go_df_1, go_df2, go_df3, go_df4], axis=0, ignore_index=True)

activations_dgd = torch.load('03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')

# Geneformer GO terms
go_df1 = pd.read_csv('03_results/reports/sc_enformer_sae_go_analysis_small_biological_process.csv')
go_df1['go type'] = 'biological process'
go_df2 = pd.read_csv('03_results/reports/sc_enformer_sae_go_analysis_large_biological_process.csv')
go_df2['go type'] = 'biological process'
go_df3 = pd.read_csv('03_results/reports/sc_enformer_sae_go_analysis_small_molecular_function.csv')
go_df3['go type'] = 'molecular function'
go_df4 = pd.read_csv('03_results/reports/sc_enformer_sae_go_analysis_large_molecular_function.csv')
go_df4['go type'] = 'molecular function'
go_df_geneformer = pd.concat([go_df1, go_df2, go_df3, go_df4], axis=0, ignore_index=True)

activations_geneformer = torch.load('03_results/reports/sae_geneformer_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')

from goatools.obo_parser import GODag
obodag = GODag("01_data/go-basic.obo")

###
# functions
###

def add_parent_go_terms(go_df):

    # for each go term, create a list of all parents
    go_terms = go_df['go_id'].unique()
    go_terms_ancestry = {}

    for go_term in go_terms:
        go_obj = obodag[go_term]
        level = go_obj.level
        out_parents = []
        out_levels = []
        while level > 1:
            parents = list(go_obj.parents)
            if len(parents) > 1:
                # get the parent with level-1
                for parent in parents:
                    if parent.level == level - 1:
                        go_obj = parent
                        break
            else:
                go_obj = parents[0]
            out_parents.append(go_obj.id)
            out_levels.append(go_obj.level)
            level = go_obj.level
        go_terms_ancestry[go_term] = {'parents': out_parents, 'levels': out_levels}

    parent_go_ids = []

    for go_term in go_terms_ancestry.keys():
        if len(go_terms_ancestry[go_term]['parents']) == 0:
            continue
        last_level = go_terms_ancestry[go_term]['levels'][-1]
        if last_level == 1:
            #print(go_terms_ancestry[go_term]['parents'])
            #print(go_terms_ancestry[go_term]['parents'][-1])
            parent_go_ids.append(go_terms_ancestry[go_term]['parents'][-1])

    parent_go_ids = list(set(parent_go_ids))

    go_df['parent_go_id_level1'] = go_df['go_id'].apply(lambda x: go_terms_ancestry[x]['parents'][-1] if len(go_terms_ancestry[x]['parents']) > 0 else x)
    go_df['parent_go_name_level1'] = go_df['parent_go_id_level1'].apply(lambda x: obodag[x].name)
    #go_df['parent_go_id_level2'] = go_df['go_id'].apply(lambda x: go_terms_ancestry[x]['parents'][-2] if len(go_terms_ancestry[x]['parents']) > 1 else x if len(go_terms_ancestry[x]['parents']) == 1 else '')
    go_df['parent_go_id_level2'] = go_df['go_id'].apply(lambda x: go_terms_ancestry[x]['parents'][-2] if len(go_terms_ancestry[x]['parents']) > 1 else x)
    go_df['parent_go_name_level2'] = go_df['parent_go_id_level2'].apply(lambda x: obodag[x].name if x != '' else '')
    #go_df['parent_go_id_level3'] = go_df['go_id'].apply(lambda x: go_terms_ancestry[x]['parents'][-3] if len(go_terms_ancestry[x]['parents']) > 2 else x if len(go_terms_ancestry[x]['parents']) == 2 else '')
    go_df['parent_go_id_level3'] = go_df['go_id'].apply(lambda x: go_terms_ancestry[x]['parents'][-3] if len(go_terms_ancestry[x]['parents']) > 2 else x)
    go_df['parent_go_name_level3'] = go_df['parent_go_id_level3'].apply(lambda x: obodag[x].name if x != '' else '')
    #go_df['parent_go_id_level4'] = go_df['go_id'].apply(lambda x: go_terms_ancestry[x]['parents'][-4] if len(go_terms_ancestry[x]['parents']) > 3 else x if len(go_terms_ancestry[x]['parents']) == 3 else '')
    go_df['parent_go_id_level4'] = go_df['go_id'].apply(lambda x: go_terms_ancestry[x]['parents'][-4] if len(go_terms_ancestry[x]['parents']) > 3 else x)
    go_df['parent_go_name_level4'] = go_df['parent_go_id_level4'].apply(lambda x: obodag[x].name if x != '' else '')

    go_df = go_df[(go_df['n_hits'] >= 1) & (go_df['mw_pval'] <= 0.01)]

    return go_df

go_df_dgd = add_parent_go_terms(go_df_dgd)
go_df_geneformer = add_parent_go_terms(go_df_geneformer)

###
# compute UMAP embeddings
###

go_feature_matrix_dgd = torch.zeros((len(go_df_dgd['go_name'].unique()), activations_dgd.shape[1]))
for i, go_id in enumerate(go_df_dgd['go_id'].unique()):
    for feat in go_df_dgd[go_df_dgd['go_id'] == go_id]['feature']:
        go_feature_matrix_dgd[i,feat] = 1
mtrx_go_ids_dgd = go_df_dgd['go_id'].unique()
mtrx_go_names_dgd = [obodag[x].name for x in mtrx_go_ids_dgd]
mtrx_feature_ids_dgd = torch.where(go_feature_matrix_dgd.sum(dim=0) > 0)[0]
go_feature_matrix_dgd = go_feature_matrix_dgd[:,torch.where(go_feature_matrix_dgd.sum(dim=0) > 0)[0]]

reducer = umap.UMAP(n_components=2, min_dist=0.5, n_neighbors=20, random_state=0, spread=10)
embedding = reducer.fit_transform(go_feature_matrix_dgd.T)
df_umap_dgd = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
df_umap_dgd['feature'] = mtrx_feature_ids_dgd.numpy()

go_feature_matrix_geneformer = torch.zeros((len(go_df_geneformer['go_name'].unique()), activations_geneformer.shape[1]))
for i, go_id in enumerate(go_df_geneformer['go_id'].unique()):
    for feat in go_df_geneformer[go_df_geneformer['go_id'] == go_id]['feature']:
        go_feature_matrix_geneformer[i,feat] = 1
mtrx_go_ids_geneformer = go_df_geneformer['go_id'].unique()
mtrx_go_names_geneformer = [obodag[x].name for x in mtrx_go_ids_geneformer]
mtrx_feature_ids_geneformer = torch.where(go_feature_matrix_geneformer.sum(dim=0) > 0)[0]
go_feature_matrix_geneformer = go_feature_matrix_geneformer[:,torch.where(go_feature_matrix_geneformer.sum(dim=0) > 0)[0]]

reducer = umap.UMAP(n_components=2, min_dist=1.0, n_neighbors=10, random_state=0, spread=10)
embedding = reducer.fit_transform(go_feature_matrix_geneformer.T)
df_umap_geneformer = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
df_umap_geneformer['feature'] = mtrx_feature_ids_geneformer.numpy()

###
# figure
###

# create a nice grid plot with 6 columns and 4 rows

level_1_mtrx_terms = []
for feat in mtrx_feature_ids_dgd:
    feat = feat.item()
    temp_go_term = go_df_dgd[go_df_dgd['feature'] == feat]['parent_go_name_level1'].value_counts().index[0]
    level_1_mtrx_terms.append(temp_go_term)
df_umap_dgd['level_1_go_term'] = level_1_mtrx_terms

level_1_mtrx_terms = []
for feat in mtrx_feature_ids_geneformer:
    feat = feat.item()
    temp_go_term = go_df_geneformer[go_df_geneformer['feature'] == feat]['parent_go_name_level1'].value_counts().index[0]
    level_1_mtrx_terms.append(temp_go_term)
df_umap_geneformer['level_1_go_term'] = level_1_mtrx_terms

df_level1_go_terms = pd.concat([df_umap_dgd[['level_1_go_term']], df_umap_geneformer[['level_1_go_term']]], axis=0, ignore_index=True)
unique_level_1_terms = df_level1_go_terms['level_1_go_term'].value_counts().index.tolist()

plt.rcParams.update({'font.size': 12})
n_cols = 7
n_rows = 5

legend_x = 0.3
legend_y = 1
handletextpad = 0.1
markerscale = 2

fig = plt.figure(figsize=(18, 8))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.3, hspace=2)
ax_list = []

# next plot a umap colored by level1 go term
ax_list.append(plt.subplot(gs[0:3, 0:2]))
ax_list[0].text(-0.1, 1.1, 'A', transform=ax_list[0].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
df_umap_dgd['level_1_go_term'] = pd.Categorical(df_umap_dgd['level_1_go_term'], categories=unique_level_1_terms, ordered=True)
level1_paletteb = sns.color_palette('Paired')
level1_palettea = sns.color_palette('Set2_r')
level1_palettea = level1_palettea[:2] + level1_palettea[3:]
level1_palette = level1_palettea + level1_paletteb
level1_palette = level1_palette[:len(unique_level_1_terms)]
sns.scatterplot(data=df_umap_dgd, x='UMAP 1', y='UMAP 2', hue='level_1_go_term', s=3, alpha=1, ec=None, palette=level1_palette, ax=ax_list[0])
ax_list[0].set_xticks([])
ax_list[0].set_yticks([])
ax_list[0].set_ylabel('UMAP 2')
ax_list[0].set_xlabel('UMAP 1')
ax_list[0].set_title('multiDGD SAE feature space')
ax_list[0].legend().remove()
# wrap the text for the legend


ax_list.append(plt.subplot(gs[0:3, 2:4]))
df_umap_geneformer['level_1_go_term'] = pd.Categorical(df_umap_geneformer['level_1_go_term'], categories=unique_level_1_terms, ordered=True)
sns.scatterplot(data=df_umap_geneformer, x='UMAP 1', y='UMAP 2', hue='level_1_go_term', s=3, alpha=1, ec=None, palette=level1_palette, ax=ax_list[1])
ax_list[1].set_xticks([])
ax_list[1].set_yticks([])
ax_list[1].set_xlabel('')
ax_list[1].set_ylabel('')
ax_list[1].set_title('Geneformer SAE feature space')
handles, labels = ax_list[1].get_legend_handles_labels()
labels = [textwrap.fill(label, 20) for label in labels]
ax_list[1].legend(handles, labels, loc='upper right', bbox_to_anchor=(2.65, 1.0), frameon=False, handletextpad=handletextpad, markerscale=5, ncol=3, fontsize=10, columnspacing=0.2, title='Level 1 GO terms', title_fontsize='medium', labelspacing=0.5)

###
# probing
###
binary_colors = sns.color_palette('rocket_r', 100)
binary_colors = [binary_colors[0], binary_colors[75]]

probing_snippets = ['checkpoint', 'stem cell proliferation', 'polarity', 'maturation', 'autophagy', 'JAK-STAT', 'growth factor receptor signaling']
for i, term_snippet in enumerate(probing_snippets):
    ax_list.append(plt.subplot(gs[3:, i]))
    if i == 0:
        ax_list[-1].text(-0.22, 1.3, 'B', transform=ax_list[-1].transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
    test_go_terms = np.asarray([1 if term_snippet in x else 0 for x in mtrx_go_names_dgd]).astype(bool)
    test_counts = go_feature_matrix_dgd[test_go_terms,:].clone().sum(dim=0).int()
    test_counts = (test_counts > 0).float()
    df_temp = df_umap_dgd.copy()
    df_temp[term_snippet] = test_counts.numpy()
    # sort the dataframe by the term snippet
    df_temp = df_temp.sort_values(by=term_snippet, ascending=True)
    sns.scatterplot(data=df_temp, x='UMAP 1', y='UMAP 2', hue=term_snippet, s=2, alpha=0.5, ec=None, palette=binary_colors, ax=ax_list[-1])
    ax_list[-1].set_xticks([])
    ax_list[-1].set_yticks([])
    if i == 0:
        ax_list[-1].set_ylabel('UMAP 2')
        ax_list[-1].set_xlabel('UMAP 1')
    else:
        ax_list[-1].set_ylabel('')
        ax_list[-1].set_xlabel('')
    ax_list[-1].set_title('"{}"'.format(textwrap.fill(term_snippet, 20)))
    # remove the legend
    ax_list[-1].legend().remove()

# save as a pdf
plt.savefig('03_results/figures/sc_automatic_feature_analysis_3.png', bbox_inches='tight', dpi=300)
plt.show()