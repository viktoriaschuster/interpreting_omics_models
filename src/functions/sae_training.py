import torch
from torch import optim
from tqdm import tqdm

from src.models.sparse_autoencoder import SparseAutoencoder, loss_function, TopK

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_sae(train_loader, input_size, hidden_factor, lr, l1_weight=1e-3, sparsity_penalty=0, val_loader=None, n_epochs=500, sae_type='vanilla', k_latent_percent=100, return_all_losses=False, print_progress=False, early_stopping=None):
    # Initialize model
    hidden_size = input_size * hidden_factor
    k_latent = max(1, int(hidden_size * k_latent_percent / 100)) # make sure there is at least 1 latent feature
    if sae_type == 'vanilla':
        sae_model = SparseAutoencoder(input_size, hidden_size)
    elif sae_type == 'bricken':
        sae_model = SparseAutoencoder(input_size, hidden_size, input_bias=True, bias_type='zero')
    elif sae_type == 'topk':
        sae_model = SparseAutoencoder(input_size, hidden_size, input_bias=True, bias_type='zero', activation=TopK(k_latent))
    else:
        raise ValueError('Invalid SAE type')
    sae_model = sae_model.to(device)

    # Optimizer
    optimizer = optim.Adam(sae_model.parameters(), lr=lr)

    # Training loop
    losses = []
    if val_loader is not None:
        val_losses = []
    if print_progress:
        epoch_range = tqdm(range(n_epochs))
    else:
        epoch_range = range(n_epochs)
    for epoch in epoch_range:
        total_loss = 0
        for x in train_loader:
            # Get inputs and convert to torch Variable
            inputs = x
            inputs = inputs.to(device)
            
            # Forward pass
            outputs, encoded = sae_model(inputs)
            
            # Compute loss
            if sae_type != 'vanilla':
                loss = loss_function(
                    outputs, 
                    inputs, 
                    encoded, 
                    sae_model.encoder[1].weight,  # Pass the encoder's weights for weight sparsity
                    sparsity_penalty, 
                    l1_weight
                )
            else:
                loss = loss_function(
                    outputs, 
                    inputs, 
                    encoded, 
                    sae_model.encoder[0].weight,  # Pass the encoder's weights for weight sparsity
                    sparsity_penalty, 
                    l1_weight
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if val_loader is not None:
            total_val_loss = 0
            for x in val_loader:
                inputs = x
                inputs = inputs.to(device)
                outputs, encoded = sae_model(inputs)
                # Compute loss
                if sae_type != 'vanilla':
                    loss = loss_function(
                        outputs, 
                        inputs, 
                        encoded, 
                        sae_model.encoder[1].weight,  # Pass the encoder's weights for weight sparsity
                        sparsity_penalty, 
                        l1_weight
                    )
                else:
                    loss = loss_function(
                        outputs, 
                        inputs, 
                        encoded, 
                        sae_model.encoder[0].weight,  # Pass the encoder's weights for weight sparsity
                        sparsity_penalty, 
                        l1_weight
                    )
                
                total_val_loss += loss.item()
        
        losses.append(total_loss / len(train_loader))

        if val_loader is not None:
            val_losses.append(total_val_loss / len(val_loader))

        # early stopping
        if early_stopping is not None:
            if epoch > early_stopping:
                # check if we have achieved a new minimum within the last early_stopping epochs
                if min(val_losses[-early_stopping:]) > min(val_losses):
                    print("Early stopping at epoch ", epoch)
                    break
    
    if return_all_losses:
        if val_loader is not None:
            return sae_model, losses, val_losses
        else:
            return sae_model, losses
    return sae_model, losses[-1]