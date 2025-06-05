import torch
import numpy as np
import pandas as pd
import mudata as md
import multiDGD
import random
from tqdm import tqdm
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# set a random seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# load data and model
data = md.read('./01_data/human_brain.h5mu', backed=False)
model = multiDGD.DGD.load(data=data, save_dir='./03_results/models/', model_name='human_brain_l20_h2-3')
reps = model.representation.z.detach()

del model
gc.collect()

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

# Mechanistic Loss Function for Sparsity and Interpretability
def loss_function(reconstructed, original, encoded, l1_weight):
    # Reconstruction loss (MSE)
    mse_loss = nn.MSELoss()(reconstructed, original)
    
    # L1 regularization for encoded activations (promotes sparsity in the hidden layer)
    l1_loss = l1_weight * torch.mean(torch.abs(encoded))

    # Combined loss
    total_loss = mse_loss + l1_loss
    return total_loss

# make sure that 03_results/reports/sc_sae_sweep/ exists
if not os.path.exists('03_results/reports/sc_sae_sweep/'):
    os.makedirs('03_results/reports/sc_sae_sweep/')
###
# training
###

# Hyperparameters
scaling_factors = [20, 100, 200, 500]
l1_weights = [1, 0.1, 1e-2, 1e-3, 1e-4]
lrs = [1e-4, 1e-5]

input_size = reps.shape[1]  # Number of input neurons
num_epochs = 1000
batch_size = 128
sparsity_penalty = 0  # Weight for weight sparsity
early_stopping = 50

# Load dataset (Replace with your own dataset)
# Example using random data, replace with actual dataset
n_train = int(reps.shape[0] * 0.9)
train_data = reps.clone()[:n_train]
val_data = reps.clone()[n_train:]
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

for scaling_factor in scaling_factors:
    hidden_size = scaling_factor * input_size
    for l1_weight in l1_weights:
        for learning_rate in lrs:
            print('Training SAE with scaling factor {}, l1 weight {}, and lr {}'.format(scaling_factor, l1_weight, learning_rate))
            # Initialize model
            sae_model = SparseAutoencoder(input_size, hidden_size)
            sae_model = sae_model.to(device)

            # Optimizer
            optimizer = optim.Adam(sae_model.parameters(), lr=learning_rate)

            # Training loop
            losses = []
            val_losses = []
            for epoch in tqdm(range(num_epochs)):

                # early stopping if the last 10 epochs have not improved the loss
                if early_stopping is not None:
                    if epoch > early_stopping:
                        # check if we have achieved a new minimum within the last early_stopping epochs
                        if min(val_losses[-early_stopping:]) > min(val_losses):
                            print("Early stopping at epoch ", epoch)
                            break
                
                total_loss = 0
                for x in train_loader:
                    # Get inputs and convert to torch Variable
                    inputs = x
                    inputs = inputs.to(device)
                    
                    # Forward pass
                    outputs, encoded = sae_model(inputs)
                    
                    # Compute loss
                    loss = loss_function(
                        outputs, 
                        inputs, 
                        encoded, 
                        l1_weight
                    )
                    
                    # Backprop and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                losses.append(total_loss / len(train_loader))

                for x in val_loader:
                    # Get inputs and convert to torch Variable
                    inputs = x
                    inputs = inputs.to(device)
                    
                    # Forward pass
                    outputs, encoded = sae_model(inputs)
                    
                    # Compute loss
                    loss = loss_function(
                        outputs, 
                        inputs, 
                        encoded, 
                        l1_weight
                    )
                    
                    total_loss += loss.item()
                
                val_losses.append(total_loss / len(val_loader))
            
            out = {}
            outputs, activations = sae_model(val_data.to(device))
            out['reconstruction loss'] = nn.MSELoss()(outputs, val_data.to(device)).item()
            sum_activs = torch.sum(torch.abs(activations), dim=0)
            out['number of active neurons'] = torch.sum(sum_activs > 0).item()
            out['l1'] = torch.sum(torch.abs(activations)).item()
            out['scaling_factor'] = scaling_factor
            out['l1_weight'] = l1_weight
            out['learning_rate'] = learning_rate
            # save this with the loss curves
            df = pd.DataFrame({
                'loss': losses,
                'val_loss': val_losses,
                'epoch': range(len(losses))
            })
            # add the output values
            for key in out:
                df[key] = out[key]
            # save the loss curves
            save_name = '03_results/reports/sc_sae_sweep/brain_vanilla_{}_{}_{}_losscurves.csv'.format(scaling_factor, l1_weight, learning_rate)
            df.to_csv(save_name)
            # delete everything
            del sae_model
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            #sae_model_save_name = '03_results/models/sae_model_10000_l1-1e-3_lr-1e-4_500epochs'
            #torch.save(sae_model.state_dict(), sae_model_save_name+'.pt')