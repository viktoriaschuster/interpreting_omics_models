import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import random
from tqdm import tqdm

# add src to the system path
import sys
sys.path.append(".")
sys.path.append('src')

#from src.models.sparse_autoencoder import *
from src.functions.sae_training import *
#from src.functions.sae_analysis_sim import *
from src.functions.sae_analysis_sim2 import *
from src.models.autoencoder import *
from src.visualization.plotting import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set a random seed
seed = 0

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

data_dir = '/projects/heads/data/simulation/singlecell/'

complexity = 'high'
n_samples = 10000
latent_dim = 150
model_depth = 2
dropout_rate = 0.1

for seed in range(10):
    temp_y = torch.load(data_dir+'large_{}-complexity_rs{}_y.pt'.format(complexity, seed), weights_only=False)
    if seed == 0:
        rna_counts = temp_y
    else:
        rna_counts = torch.cat((rna_counts, temp_y), dim=0)
# limit to the training data
n_samples_validation = int(n_samples*0.1)
rna_counts = rna_counts[-n_samples_validation:]

latent_dim = 150

# load encoder

#encoder = Encoder(20000, latent_dim, model_depth).to(device)
model_name = 'large_high-complexity_2-depth_{}-latent_0.1-dropout_100000-samples'.format(latent_dim)

##########

activations = torch.load(data_dir+'sim2_{}_sae1000x_activations.pth'.format(model_name))

# threshold is 5th percentile of the activations
# sample from the activations
threshold = torch.quantile(torch.randperm(activations.shape[0])[:1000].flatten().float(), 0.05).item()
print('Threshold:', threshold)

unique_active_indices = get_unique_active_unit_indices_from_activations(activations, threshold=threshold)
print('Number of unique active units:', len(unique_active_indices))

cos_sim_sae = torch.zeros((len(unique_active_indices), len(unique_active_indices)))
for i in tqdm(unique_active_indices):
    cos_sim_sae[i,:] = torch.nn.functional.cosine_similarity(activations[:,unique_active_indices], activations[:,i].unsqueeze(1), dim=0).detach().cpu()
torch.save(cos_sim_sae, data_dir+'sim2_{}_sae1000x_sae-cos-sim.pth'.format(model_name))

print('SAE Cosine similarity matrix saved')