import torch
import numpy as np
import anndata as ad
import multiDGD
import os
import gc
import torch.nn as nn

import sys
sys.path.append(".")
sys.path.append('src')

# Configuration
dev_id = 1
device = torch.device(f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu")
seeds = [0, 42, 9307]  # The 3 random seeds to process
print(f"Using device: {device}")

##########################
# Load model and data once
##########################

if torch.cuda.is_available():
    print(f"Using GPU: {device}")
    data_dir = '/home/vschuste/data/singlecell/'
else:
    data_dir = '/Users/vschuste/Documents/work/data/singlecell/'

print("Loading data and model...")
data = ad.read_h5ad(data_dir+'human_bonemarrow.h5ad')

model = multiDGD.DGD.load(data=data, save_dir='./03_results/models/', model_name='human_bonemarrow_l20_h2-3_test50e').to(device)
data = data[data.obs["train_val_test"] == "train"]
del data
gc.collect()

# Get model representations
reps = model.representation.z.detach()
print(f"Model representations shape: {reps.shape}")

##########################
# Define SAE model architecture
##########################

class SparseAutoencoder(torch.nn.Module):
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
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

##########################
# Process each seed
##########################

input_size = reps.shape[1]
hidden_size = 10**4
batch_size = 128

for seed in seeds:
    print(f"\n--- Processing seed {seed} ---")
    
    # Define file paths
    if seed == 0:
        sae_model_path = '03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs.pt'
        activations_path = '03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt'
    else:
        sae_model_path = f'03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_seed{seed}.pt'
        activations_path = f'03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_seed{seed}_activations.pt'
    
    # Check if activations already exist
    if os.path.exists(activations_path):
        print(f"Activations for seed {seed} already exist at {activations_path}. Skipping...")
        continue
    
    # Check if model exists
    if not os.path.exists(sae_model_path):
        print(f"SAE model for seed {seed} not found at {sae_model_path}. Skipping...")
        continue
    
    print(f"Loading SAE model for seed {seed}...")
    # Load the SAE model
    sae_model = SparseAutoencoder(input_size, hidden_size)
    sae_model.load_state_dict(torch.load(sae_model_path))
    sae_model.to(device)
    sae_model.eval()  # Set to evaluation mode
    
    print(f"Computing activations for seed {seed}...")
    # Compute activations in batches
    activations = []
    for i in range(0, reps.shape[0], batch_size):
        batch_reps = reps[i:i+batch_size, :].to(device)
        with torch.no_grad():
            _, activations_batch = sae_model(batch_reps)
        activations.append(activations_batch.cpu())
        
        # Print progress
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed batch {i // batch_size + 1}/{(reps.shape[0] + batch_size - 1) // batch_size}")
    
    # Concatenate all activations
    activations = torch.cat(activations, dim=0)
    print(f"Activations shape: {activations.shape}")
    
    # Save activations
    print(f"Saving activations to {activations_path}...")
    torch.save(activations, activations_path)
    
    # Clean up GPU memory
    del sae_model, activations
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    print(f"Successfully saved activations for seed {seed}")

print("\n--- All seeds processed ---")

# Summary of what was processed
print("\nSummary:")
for seed in seeds:
    if seed == 0:
        activations_path = '03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt'
    else:
        activations_path = f'03_results/reports/sae_model_10000_l1-1e-3_lr-1e-4_500epochs_seed{seed}_activations.pt'
    
    if os.path.exists(activations_path):
        print(f"✓ Seed {seed}: Activations exist at {activations_path}")
    else:
        print(f"✗ Seed {seed}: Activations missing")

print("Done!")

