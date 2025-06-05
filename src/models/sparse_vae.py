import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import tqdm

class PriorVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, prior_type='gaussian', beta=1.0):
        """
        VAE with different prior options
        
        Args:
            input_dim (int): Dimension of input data
            hidden_dim (int): Dimension of hidden layers
            latent_dim (int): Dimension of latent space
            prior_type (str): Type of prior ('gaussian', 'laplace', or 'cauchy')
        """
        super(PriorVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.prior_type = prior_type.lower()
        self.beta = beta
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent parameters (mu and log_var for all priors)
        self.mu_encoder = nn.Linear(hidden_dim, latent_dim)
        self.logvar_encoder = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Prior distribution parameters
        self.register_buffer('prior_mu', torch.zeros(latent_dim))
        self.register_buffer('prior_scale', torch.ones(latent_dim))
        
    def encode(self, x):
        """
        Safe encoding that handles extreme values
        """
        # Apply preprocessing if needed
        # x = torch.log1p(x)  # Uncomment if needed
        
        h = self.encoder(x)
        mu = self.mu_encoder(h)
        logvar = self.logvar_encoder(h)
        
        # Clamp values for stability
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from the posterior using the reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        
        if self.prior_type == 'gaussian':
            # Standard Gaussian reparameterization
            eps = torch.randn_like(std)
            return mu + eps * std
            
        elif self.prior_type == 'laplace':
            # Laplace reparameterization - with numerical stability
            u = torch.rand_like(std)
            # Avoid extreme values that could cause numerical issues
            u = torch.clamp(u, min=0.001, max=0.999)
            eps = -torch.sign(u - 0.5) * torch.log(1 - 2 * torch.abs(u - 0.5))
            return mu + eps * std
            
        elif self.prior_type == 'cauchy':
            # Cauchy reparameterization - with numerical stability
            u = torch.rand_like(std)
            # Avoid values too close to 0 or 1 which would make tan explode
            u = torch.clamp(u, min=0.01, max=0.99) 
            eps = torch.tan(np.pi * (u - 0.5))
            # Clip extremely large values
            eps = torch.clamp(eps, min=-10.0, max=10.0)
            return mu + eps * std
        
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")
    
    def decode(self, z):
        """Decode the latent variable"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through the VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z
    
    def kl_divergence(self, mu, logvar):
        """Calculate KL divergence between posterior and prior"""
        var = torch.exp(logvar)
        std = torch.exp(0.5 * logvar)
        
        if self.prior_type == 'gaussian':
            # Analytical KL for Gaussian - more numerically stable version
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - var, dim=1)
            
        elif self.prior_type == 'laplace':
            # Approximated KL for Laplace prior with numerical stability
            z = self.reparameterize(mu, logvar)
            log_posterior = -0.5 * torch.sum(logvar + torch.clamp((z - mu).pow(2) / var, max=100), dim=1)
            log_prior = -torch.sum(torch.abs(z), dim=1)  # Laplace prior with scale=1
            kl = log_posterior - log_prior
            
        elif self.prior_type == 'cauchy':
            # Approximated KL for Cauchy prior with numerical stability
            z = self.reparameterize(mu, logvar)
            log_posterior = -0.5 * torch.sum(logvar + torch.clamp((z - mu).pow(2) / var, max=100), dim=1)
            # Avoid log(1+x^2) becoming too large or small
            log_prior = -torch.sum(torch.log(1 + torch.clamp(z.pow(2), max=1e6)), dim=1)
            kl = log_posterior - log_prior
            
        else:
            raise ValueError(f"Unknown prior type: {self.prior_type}")
            
        # Clip unreasonably large values
        kl = torch.clamp(kl, max=1e6)
        return kl
    
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss function: reconstruction + KL divergence"""
        # Reconstruction loss (using mean squared error)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')  # Use mean instead of sum/batch_size
        
        # KL divergence with gradient clipping
        kl_div = self.kl_divergence(mu, logvar).mean()
        
        # Detect NaNs early
        if torch.isnan(recon_loss) or torch.isnan(kl_div):
            print("NaN detected in loss calculation!")
            if torch.isnan(recon_loss):
                print("Reconstruction loss is NaN")
            if torch.isnan(kl_div):
                print("KL divergence is NaN")
            # Return non-NaN losses for stability
            return torch.tensor(1.0, device=x.device, requires_grad=True), \
                torch.tensor(0.5, device=x.device, requires_grad=True), \
                torch.tensor(0.5, device=x.device, requires_grad=True)
        
        # Total loss with beta factor to control KL weight
        total_loss = recon_loss + self.beta * kl_div
        
        return total_loss, recon_loss, kl_div

def train_vae(model, optimizer, data_loader, val_loader, epochs=10, device='cpu', early_stopping=None):
    model.train()
    out_log = {
        'loss': [],
        'val_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'epoch': [],
    }
    print("Training VAE...")
    for epoch in tqdm.tqdm(range(epochs)):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch_idx, data in enumerate(data_loader):
            # Unpack the data - data is returned as a tuple from DataLoader
            if isinstance(data, (list, tuple)):
                data = data[0]  # Extract the first (and only) tensor
            data = data.to(device)
                
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar, _ = model(data)
            
            # Compute loss
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
        # Print epoch results
        out_log['loss'].append(total_loss / len(data_loader))
        #avg_recon = total_recon / len(data_loader)
        #avg_kl = total_kl / len(data_loader)
        
        #print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
        #      f'Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}')

        model.eval()
        val_loss = 0
        val_recon = 0
        val_kl = 0
        with torch.no_grad():
            for val_data in val_loader:
                if isinstance(val_data, (list, tuple)):
                    val_data = val_data[0]
                val_data = val_data.to(device)
                recon_val, mu_val, logvar_val, _ = model(val_data)
                val_loss_batch, recon_loss_batch, kl_loss_batch = model.loss_function(recon_val, val_data, mu_val, logvar_val)
                val_loss += val_loss_batch.item()
                val_recon += recon_loss_batch.item()
                val_kl += kl_loss_batch.item()
        out_log['val_loss'].append(val_loss / len(val_loader))
        out_log['recon_loss'].append(val_recon / len(val_loader))
        out_log['kl_loss'].append(val_kl / len(val_loader))
        out_log['epoch'].append(epoch)

        # early stopping
        if early_stopping is not None:
            if epoch > early_stopping:
                # check if we have achieved a new minimum within the last early_stopping epochs
                if min(out_log['val_loss'][-early_stopping:]) > min(out_log['val_loss']):
                    print("Early stopping at epoch ", epoch)
                    break
    
    return model, out_log