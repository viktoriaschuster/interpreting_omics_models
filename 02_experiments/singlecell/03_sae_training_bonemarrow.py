import torch
import numpy as np
import pandas as pd
import anndata as ad
import random
import tqdm

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# set a random seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

data = ad.read_h5ad('./01_data/human_bonemarrow.h5ad')
train_indices = np.where(data.obs['train_val_test'] == 'train')[0]
del data
enformer_embeddings = pd.read_csv('./03_results/embeddings/human_bonemarrow_geneformer.csv', index_col=0)
enformer_embeddings = enformer_embeddings.iloc[train_indices, :]
reps = torch.FloatTensor(enformer_embeddings.iloc[:,:-1].values) # excluding cell_type
print(f"Shape of the embeddings: {reps.shape}")

import torch
import torch.nn as nn
import torch.optim as optim

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
def loss_function(reconstructed, original, encoded, weights, sparsity_penalty, l1_weight):
    # Reconstruction loss (MSE)
    mse_loss = nn.MSELoss()(reconstructed, original)
    
    # L1 regularization for encoded activations (promotes sparsity in the hidden layer)
    l1_loss = l1_weight * torch.mean(torch.abs(encoded))

    # Weight sparsity (to promote interpretable features in the weights)
    weight_sparsity_loss = sparsity_penalty * torch.sum(torch.abs(weights))

    # Combined loss
    total_loss = mse_loss + l1_loss + weight_sparsity_loss
    return total_loss

###
# training
###

# Hyperparameters
input_size = reps.shape[1]  # Number of input neurons
hidden_size = 10**4
learning_rate = 1e-4
num_epochs = 500
batch_size = 128
sparsity_penalty = 0  # Weight for weight sparsity
l1_weight = 1e-3  # Weight for activation sparsity
#early_stopping = 10

# Load dataset (Replace with your own dataset)
# Example using random data, replace with actual dataset
train_data = reps.clone()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize model
sae_model = SparseAutoencoder(input_size, hidden_size)
sae_model = sae_model.to(device)

# Optimizer
optimizer = optim.Adam(sae_model.parameters(), lr=learning_rate)

# Training loop
losses = []
pbar = tqdm.tqdm(range(num_epochs), desc="Training", unit="epoch")
for epoch in pbar:

    # early stopping if the last 10 epochs have not improved the loss
    #if len(losses) > early_stopping:
    #    if np.min(losses[-early_stopping:]) == losses[-early_stopping]:
    #        print(f'Early stopping at epoch {epoch + 1}')
    #        break
    
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
            sae_model.encoder[0].weight,  # Pass the encoder's weights for weight sparsity
            sparsity_penalty, 
            l1_weight
        )
        
        # Backprop and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    losses.append(total_loss / len(train_loader))
    
    #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.8f}')
    pbar.set_postfix(loss=total_loss / len(train_loader))
pbar.close()

print("Training complete.")

# save the model
sae_model_save_name = '03_results/models/sae_enformer_10000_l1-1e-3_lr-1e-4_500epochs'
torch.save(sae_model.state_dict(), sae_model_save_name+'.pt')

# plot loss curve (normal) and log_scaled
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(20, 7))
axs[0].plot(losses)
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss')
axs[1].plot(losses)
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].set_title('Training Loss (log scale)')
axs[1].set_yscale('log')
plt.savefig(sae_model_save_name+'_loss.png')