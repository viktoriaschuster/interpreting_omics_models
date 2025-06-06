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

###
# get the superposition fits
###

file_dir = '03_results/models/'
# get all subdirectories starting with 'largesim_ae'
subdirs = [x for x in os.listdir(file_dir) if os.path.isdir(file_dir + x) and x.startswith('largesim_ae')]

r_squares = []
for subdir in subdirs:
    # get the r2 score
    try:
        r2 = pd.read_csv(file_dir + subdir + '/r_squares.csv')
        # give it column names
        r2.columns = ['variable', 'R^2']
        
        # get the latent dimension
        r2['latent'] = int(subdir.split('_latent-')[-1].split('_')[0])
        # get the depth
        r2['depth'] = int(subdir.split('_depth-')[-1].split('_')[0])
        # get the width
        r2['width'] = subdir.split('_width-')[-1].split('_')[0]
        # get the seed
        r2['seed'] = int(subdir.split('_seed-')[-1].split('_')[0])
        # create a name
        r2['name'] = subdir
        r2['name_short'] = subdir.split('_latent-')[-1].split('_')[0] + '-' + subdir.split('_depth-')[-1].split('_')[0] + '-' + ('n' if subdir.split('_width-')[-1].split('_')[0] == 'narrow' else 'w')

        # also load the history and add the validation loss
        history = pd.read_csv(file_dir + subdir + '/history.csv')
        r2['val_loss'] = history['val_loss'].iloc[-1]
        
        r_squares.append(r2)
    except:
        print('Error with ' + subdir)
        continue
# concatenate all r2 scores
r_squares = pd.concat(r_squares)
r_squares = r_squares.reset_index(drop=True)
# change the names of the variables to the last part of the string
r_squares['variable'] = r_squares['variable'].str.split('_').str[-1]
# sort the df first by latent dimension, then by depth, then by width
r_squares = r_squares.sort_values(by=['latent', 'depth', 'width', 'seed'])

r_squares.head()

###
# plot
###

# plot val loss against R2
best_arches = []
for arch_name in r_squares['name_short'].values:
    if '2-w' in arch_name:
        best_arches.append('2-w')
    elif '4-w' in arch_name:
        best_arches.append('4-w')
    else:
        best_arches.append('other')
r_squares['#layer-width'] = best_arches
# make this a categorical variable
r_squares['#layer-width'] = pd.Categorical(r_squares['#layer-width'], categories=['2-w', '4-w', 'other'], ordered=True)
# change the variable names
r_squares['variable'] = r_squares['variable'].replace({'y': 'Y', 'x2': "X''", 'x1': "X'", 'x0': 'X', 'ct': 'A', 'co': 'B'})
# make them categorical with the order of the variables
r_squares['variable'] = pd.Categorical(r_squares['variable'], categories=variables, ordered=True)

fig = plt.figure(figsize=(15, 4))
from matplotlib import gridspec
gs = gridspec.GridSpec(1, 3, width_ratios=[0.6, 1, 1])
# adjust the spacing
plt.subplots_adjust(wspace=0.8)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax0.text(-0.4, 1.1, "A", fontsize=16, transform=ax0.transAxes, fontweight='bold')
# remove the empty subplot without removing the axis
ax0.axis('off')
ax1.text(-0.4, 1.1, "B", fontsize=16, transform=ax1.transAxes, fontweight='bold')
# adjust spacing
#plt.axvspan(-1.0, 0, color='grey', alpha=0.3)
sns.scatterplot(data=r_squares, y='val_loss', x='R^2', hue='variable', style='variable', palette=colors, ax=ax1, s=100, alpha=0.7)
ax1.set_xlim(-1, 1.05)
ax1.set_xticks(np.arange(-1, 1.1, 0.5))
ax1.set_yscale('log')
ax1.set_xlabel('R²')
ax1.set_ylabel('Validation Loss')
ax1.set_title('Superposition fit')
# move the legend outside the plot  
ax1.legend(title='Variable', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, handletextpad=0.2, alignment='left', columnspacing=0.8, labelspacing=0.2)#.remove()
#plt.tight_layout()

sns.scatterplot(data=r_squares, y='val_loss', x='R^2', hue='#layer-width', style='latent', palette=sns.color_palette("viridis_r", 3), ax=ax2, s=100, alpha=0.7)
ax2.set_xlim(-1, 1.05)
ax2.set_xticks(np.arange(-1, 1.1, 0.5))
ax2.set_yscale('log')
ax2.set_xlabel('R²')
ax2.set_ylabel('Validation Loss')
ax2.set_title('Superposition fit') 
ax2.legend(title='Architecture', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, handletextpad=0.2, alignment='left', columnspacing=0.8, labelspacing=0.2)#.remove()

# save the figure
fig.savefig('03_results/figures/simL_val_loss_vs_r2_v3.pdf', bbox_inches='tight')