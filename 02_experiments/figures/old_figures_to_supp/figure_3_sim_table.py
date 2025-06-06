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
#latents = [20, 100, 150]
latents = [150]

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
metrics_corr['latent'] = metrics_corr['latent_dim'].values
# keep only the rows with latent in latents
metrics_corr = metrics_corr[metrics_corr['latent'].isin(latents)]

# plot the variable against n_activation_features and color by latent dim
# make the figure with gridspec
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 7, figure=plt.figure(figsize=(15, 4)))
# add spacing
#gs.update(wspace=1.2)

ax0 = plt.subplot(gs[0, 0])
# add the baseline values with the same y axis as in the next plot
#baseline_methods = ['pca', 'ica', 'svd'] # ica was no good
baseline_methods = ['pca', 'ica', 'svd']
baselines = []
for bm in baseline_methods:
    temp = pd.read_csv(f'03_results/reports/files/sim2L_{bm}_metrics.csv')
    temp['method'] = bm
    baselines.append(temp)
baselines = pd.concat(baselines)
baselines = baselines.reset_index(drop=True)
baselines['variable'] = baselines['variable'].replace({'y': 'Y', 'x2': "X''", 'x1': "X'", 'x0': 'X', 'ct': 'A', 'co': 'B'})
baselines['variable'] = pd.Categorical(baselines['variable'], categories=variables, ordered=True)
ax0.text(-0.45, 1.05, "A", fontsize=18, transform=ax0.transAxes, fontweight='bold')
sns.scatterplot(data=baselines, x='method', y='highest_corrs (max)', hue='variable', palette=colors, ax=ax0, s=50, alpha=0.7)
ax0.set_ylim(-0.0, 1.05)
ax0.set_ylabel('Pearson correlation')
ax0.set_xlabel('Baseline method')
# remove legend
ax0.legend(title='Latent dim', loc='upper left', bbox_to_anchor=(1, 1)).remove()

ax = plt.subplot(gs[0, 1:3])
#ax.text(-0.45, 1.05, "A", fontsize=18, transform=ax.transAxes, fontweight='bold')
sns.lineplot(data=metrics_corr, x='hidden_factor', y='highest_corr', hue='variable', style='variable', palette=colors, ax=ax, linewidth=1.0, legend=False)
sns.scatterplot(data=metrics_corr, x='hidden_factor', y='highest_corr', hue='variable', style='latent', palette=colors, ax=ax, s=50, alpha=0.7)
ax.set_ylim(-0.0, 1.05)
ax.set_xlabel('SAE hidden scaling factor')
#ax.set_ylabel('Pearson correlation')
ax.set_ylabel('')
# remove all y ticks
ax.set_yticks([])
ax.set_title('max(corr(neurons, variables))')

# print out a table with the highest correlation values +- the SEM
def print_table_highest_corrs(metrics_corr):
    # get the highest correlation values for each variable and latent dim
    table = metrics_corr.groupby(['latent', 'variable']).agg({'highest_corr': ['mean', 'sem']}).reset_index()
    table.columns = ['latent', 'variable', 'mean', 'sem']
    # format the mean and sem
    table['mean'] = table['mean'].apply(lambda x: f'{x:.2f}')
    table['sem'] = table['sem'].apply(lambda x: f'±{x:.2f}')
    # combine mean and sem
    table['mean_sem'] = table['mean'] + ' ' + table['sem']
    # pivot the table to have variables as columns
    table_pivot = table.pivot(index='latent', columns='variable', values='mean_sem')
    # reset index
    table_pivot.reset_index(inplace=True)
    # rename the index column to Latent dimension
    table_pivot.rename(columns={'latent': 'Latent dimension'}, inplace=True)
    
    return table_pivot
table_highest_corrs = print_table_highest_corrs(metrics_corr)
# print the table
print(table_highest_corrs)
# also print the baseline values
def print_table_baseline_corrs(baselines):
    # get the highest correlation values for each variable and baseline method
    table = baselines.groupby(['method', 'variable']).agg({'highest_corrs (max)': ['mean', 'sem']}).reset_index()
    table.columns = ['method', 'variable', 'mean', 'sem']
    # format the mean and sem
    table['mean'] = table['mean'].apply(lambda x: f'{x:.2f}')
    table['sem'] = table['sem'].apply(lambda x: f'±{x:.2f}')
    # combine mean and sem
    table['mean_sem'] = table['mean'] + ' ' + table['sem']
    # pivot the table to have variables as columns
    table_pivot = table.pivot(index='method', columns='variable', values='mean_sem')
    # reset index
    table_pivot.reset_index(inplace=True)
    
    return table_pivot
table_baseline_corrs = print_table_baseline_corrs(baselines)
# print the table
print("\nbaselines")
print(table_baseline_corrs)
