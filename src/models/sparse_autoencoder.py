import torch
import torch.nn as nn
import math
from typing import Callable

class BiasLayer(torch.nn.Module):
    def __init__(self, input_size, init='standard') -> None:
        super().__init__()
        if init == 'standard':
            stdv = 1. / math.sqrt(input_size)
            # init bias to a uniform distribution
            bias_value = torch.empty((input_size)).uniform_(-stdv, stdv)
        else:
            bias_value = torch.zeros(input_size)

        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return x + self.bias_layer

# Sparse Autoencoder Model with Mechanistic Interpretability
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, input_bias: bool = False, activation: Callable = nn.ReLU(), bias_type='standard', tied: bool = False):
        super(SparseAutoencoder, self).__init__()
        # Encoder
        self.pre_bias = BiasLayer(input_size, init=bias_type)
        self.latent_bias = BiasLayer(hidden_size, init=bias_type)
        self.encoder_linear = nn.Linear(input_size, hidden_size, bias=False)
        self.activation = activation
        if input_bias:
            self.encoder = nn.Sequential(
                self.pre_bias,
                self.encoder_linear,
                self.latent_bias,
                self.activation
            )
        else:
            self.encoder = nn.Sequential(
                self.encoder_linear,
                self.latent_bias,
                self.activation
            )
        #decoder
        if tied:
            self.decoder: nn.Linear | TiedTranspose = TiedTranspose(self.encoder)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, input_size, bias=False)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, input_size, bias=False),
                BiasLayer(input_size, init=bias_type)
            )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update({prefix + "k": self.k, prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)


ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}

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