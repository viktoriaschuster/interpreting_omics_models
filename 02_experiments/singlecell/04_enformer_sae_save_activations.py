import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import anndata as ad

import sys
sys.path.append(".")
sys.path.append('src')

dev_id = 2
device = torch.device(f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu")
#small_features = False
small_features = True
#go_category = 'biological_process'
go_category = 'molecular_function'

def load_data(model_type):
    if model_type == 'DGD':
        # load data and model
        data = ad.read_h5ad('./01_data/human_bonemarrow.h5ad')
        model = multiDGD.DGD.load(data=data, save_dir='./03_results/models/', model_name='human_bonemarrow_l20_h2-3_test50e')
        reps = model.representation.z.detach()
        del model
    elif model_type == 'Enformer':
        data = ad.read_h5ad('./01_data/human_bonemarrow.h5ad')
        train_indices = np.where(data.obs['train_val_test'] == 'train')[0]
        enformer_embeddings = pd.read_csv('./03_results/embeddings/human_bonemarrow_geneformer.csv', index_col=0)
        enformer_embeddings = enformer_embeddings.iloc[train_indices, :]
        reps = torch.FloatTensor(enformer_embeddings.iloc[:,:-1].values) # excluding cell_type
        del enformer_embeddings
    else:
        raise ValueError("Invalid model type. Choose 'DGD' or 'Enformer'.")
    data = data[data.obs['train_val_test'] == 'train']

    return data, reps

def load_model(model_name, input_size, hidden_size):
    # Sparse Autoencoder Model with Mechanistic Interpretability
    class SparseAutoencoder(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(SparseAutoencoder, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                #nn.Sigmoid()
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded, encoded
    
    sae_model = SparseAutoencoder(input_size, hidden_size)
    sae_model.load_state_dict(torch.load(model_name+'.pt'))
    return sae_model

##########################
# load model and data
##########################

data, reps = load_data('Enformer')
library = data.obs['GEX_n_counts'].values
data_gene_names = (data.var[data.var['modality'] == 'GEX']).index
data_gene_ids = data.var[data.var['modality'] == 'GEX']['gene_id'].values
modality_switch = len(data.var[data.var['modality'] == 'GEX'])
data = data[:, :modality_switch]

input_size = reps.shape[1]
hidden_size = 10**4
sae_model = load_model('03_results/models/sae_enformer_10000_l1-1e-3_lr-1e-4_500epochs', input_size, hidden_size)
sae_model.to(device)

# get activations in chunks
batch_size = 128
activations = []
for i in range(0, reps.shape[0], batch_size):
    batch_reps = reps[i:i+batch_size, :].to(device)
    with torch.no_grad():
        _, activations_batch = sae_model(batch_reps)
    activations.append(activations_batch.cpu())
activations = torch.cat(activations, dim=0)

# save activations
torch.save(activations, '03_results/reports/sae_geneformer_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt')
print("done")