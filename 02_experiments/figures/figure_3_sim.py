import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

###
# general plotting stuff
###

fontsize = 14
# set the fontsize
plt.rcParams.update({'font.size': fontsize})
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
colors = ['#b14c8d','#0044ad','#5176e8','#82A0FD','#417a78','#c5a862']
#variables = ['Y', 'X₂', 'X₁', 'X₀', 'A', 'B']
variables = ['Y', "X''", "X'", 'X', 'A', 'B']

##############################
# first plot for max correlation
##############################

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
#metrics_corr = metrics_corr.rename(columns={'latent_dim': 'Latent\ndimension'})
metrics_corr['hidden_dim'] = metrics_corr['hidden_factor'] * metrics_corr['latent_dim']


# plot the variable against n_activation_features and color by latent dim
# make the figure with gridspec
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 7, figure=plt.figure(figsize=(8, 4)))
# add spacing
gs.update(wspace=1.2)
ax = plt.subplot(gs[0, :2])
sns.lineplot(data=metrics_corr, x='hidden_factor', y='highest_corr', hue='variable', style='variable', palette=colors, ax=ax, linewidth=1.0, legend=False)
sns.scatterplot(data=metrics_corr, x='hidden_factor', y='highest_corr', hue='variable', style='latent_dim', palette=colors, ax=ax, s=20)
ax.set_ylim(-0.6, 1.05)
ax.set_xlabel('SAE hidden scaling factor')
ax.set_ylabel('Pearson correlation')
ax.set_title('max(corr(neurons, variables))')
#plt.title('Large simulation')
# set the ticks to the actual x values
#plt.xticks(metrics_corr['hidden_factor'].unique())
#plt.xscale('log')
#plt.tight_layout()
# keep only the style legend
ax.legend(title='Latent dim', loc='upper left', bbox_to_anchor=(1, 1)).remove()
handles, labels = ax.get_legend_handles_labels()
# remove unwanted elements and add an empty element to keep the order
import matplotlib
handles_n = handles[2:8] + handles[8:]
labels_n = labels[2:7] + [''] + labels[8:]
# create the legend and reduce the distance between markers and text
ax.legend(handles_n, labels_n, loc='upper left', bbox_to_anchor=(0.85, 1.0), markerscale=2, handletextpad=0.01, frameon=False)

##############################
# second plot for structure
##############################
# now the second plot
ax2 = plt.subplot(gs[0, 3:])

results_dir = '03_results/models/'
latent_dim = 150
depth = 2
width = 'wide'
seed = 0
file_dir = results_dir + 'largesim_ae_latent-' + str(latent_dim) + '_depth-' + str(depth) + '_width-' + width + '_seed-' + str(seed) + '/sae/'

percentiles = [1,10,20,30,40,50,60,70,80,90,95,99]
df_results_2 = pd.read_csv(file_dir + 'best_matches_feat2X_max_filled.csv')
df_results_3 = df_results_2[df_results_2['max_fraction_covered'] > 0]
# make percentile integer
df_results_3['percentile'] = df_results_3['percentile'].astype(int)

# scientific boxplot
sns.boxplot(data=df_results_3, x='percentile', y='max_fraction_covered', ax=ax2, notch=True, fliersize=3, flierprops={"marker": "x"}, boxprops={"facecolor": (.3, .5, .7, .5)})
# over this plot a red dot for the mean
sns.stripplot(data=df_results_3.groupby('percentile').mean().reset_index(), x='percentile', y='max_fraction_covered', color='crimson', size=5, ax=ax2)
# above each box print the number of Xs covered as text
for i, perc in enumerate(percentiles):
    n_x = df_results_3[df_results_3['percentile'] == percentiles[i]].shape[0]
    if n_x > 99: 
        spacer = 0.5
    else:
        spacer = 0.3
    ax2.text(i-spacer, 1.07, f'{n_x}', verticalalignment='top', fontsize=fontsize-4)
ax2.set_ylim(0, 1.1)
ax2.set_xlabel('Percentile')
ax2.set_ylabel('Fraction of Y covered')
ax2.set_title('Structure identification')

plt.savefig('03_results/figures/simL_variable_corr.pdf', bbox_inches='tight')