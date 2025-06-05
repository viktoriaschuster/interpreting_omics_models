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
variables = ['Y', 'X₂', 'X₁', 'X₀', 'A', 'B']

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
r_squares['best_arch'] = ['2-w' if '2-w' in x else 'other' for x in r_squares['name_short']]
# change the variable names
r_squares['variable'] = r_squares['variable'].replace({'y': 'Y', 'x2': 'X₂', 'x1': 'X₁', 'x0': 'X₀', 'ct': 'A', 'co': 'B'})
# make them categorical with the order of the variables
r_squares['variable'] = pd.Categorical(r_squares['variable'], categories=variables, ordered=True)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# add horizontal spacing
plt.subplots_adjust(wspace=1.0)
#plt.axvspan(-1.0, 0, color='grey', alpha=0.3)
sns.scatterplot(data=r_squares, y='val_loss', x='R^2', hue='latent', ax=axs[0], s=20, palette=sns.color_palette("viridis_r", 4))
axs[0].set_xlim(-1, 1.05)
axs[0].set_xticks(np.arange(-1, 1.1, 0.5))
axs[0].set_yscale('log')
axs[0].set_xlabel('R²')
axs[0].set_ylabel('Validation Loss')
axs[0].set_title('Superposition fit')
# move the legend outside the plot  
axs[0].legend(title='latent', bbox_to_anchor=(0.9, 1), loc='upper left', frameon=False, handletextpad=0.1)#.remove()
#plt.tight_layout()

# next color by depth
sns.scatterplot(data=r_squares, y='val_loss', x='R^2', hue='depth', ax=axs[1], s=20, palette=sns.color_palette("viridis_r", 3))
axs[1].set_xlim(-1, 1.05)
axs[1].set_xticks(np.arange(-1, 1.1, 0.5))
axs[1].set_yscale('log')
axs[1].set_xlabel('R²')
axs[1].set_ylabel('Validation Loss')
axs[1].set_title('Superposition fit')
axs[1].legend(title='layers', bbox_to_anchor=(0.95, 1), loc='upper left', frameon=False, handletextpad=0.1)#.remove()

# next color by width
sns.scatterplot(data=r_squares, y='val_loss', x='R^2', hue='width', ax=axs[2], s=20)
axs[2].set_xlim(-1, 1.05)
axs[2].set_xticks(np.arange(-1, 1.1, 0.5))
axs[2].set_yscale('log')
axs[2].set_xlabel('R²')
axs[2].set_ylabel('Validation Loss')
axs[2].set_title('Superposition fit')
axs[2].legend(title='width', bbox_to_anchor=(0.9, 1), loc='upper left', frameon=False, handletextpad=0.1)#.remove()

# save the figure
fig.savefig('03_results/figures/simL_val_loss_vs_r2_supp.pdf', bbox_inches='tight')