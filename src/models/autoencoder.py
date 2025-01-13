import torch
import random
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, depth, dropout=0.0):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.depth = depth
        hidden_dim = max(100, latent_dim*2)

        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                if depth == 1:
                    self.layers.append(nn.Linear(input_dim, latent_dim))
                else:
                    self.layers.append(nn.Linear(input_dim, hidden_dim))
                    self.layers.append(nn.Dropout(dropout))
                    self.layers.append(nn.ReLU())
            elif i == depth - 1:
                self.layers.append(nn.Linear(hidden_dim, latent_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Dropout(dropout))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Encoder2(nn.Module):
    def __init__(self, input_dim, latent_dim, depth, dropout=0.0, width="narrow"):
        super(Encoder2, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.depth = depth
        hidden_dims = []
        if width == "narrow":
            if depth > 2:
                hidden_dims.append(max(1000, latent_dim*2))
                for i in range(depth-2):
                    hidden_dims.append(max(150, latent_dim*2))
            elif depth == 2:
                hidden_dims.append(max(150, latent_dim*2))
        elif width == "wide":
            # find intervals with linspace
            if depth > 1:
                steps = np.linspace(latent_dim, int(0.5*input_dim), num=depth+1)[::-1]
                hidden_dims = [int(step) for step in steps[1:-1]]
        
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                if depth == 1:
                    self.layers.append(nn.Linear(input_dim, latent_dim))
                else:
                    self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
                    self.layers.append(nn.Dropout(dropout))
                    self.layers.append(nn.ReLU())
            elif i == depth - 1:
                self.layers.append(nn.Linear(hidden_dims[-1], latent_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                self.layers.append(nn.Dropout(dropout))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, depth, dropout=0.0):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.depth = depth
        hidden_dim = max(100, latent_dim*2)

        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                if depth == 1:
                    self.layers.append(nn.Linear(latent_dim, output_dim))
                else:
                    self.layers.append(nn.Linear(latent_dim, hidden_dim))
                    self.layers.append(nn.Dropout(dropout))
                    self.layers.append(nn.ReLU())
            elif i == (depth - 1):
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.Dropout(dropout))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Decoder2(nn.Module):
    def __init__(self, latent_dim, output_dim, depth, hidden_dims, dropout=0.0, width="narrow"):
        super(Decoder2, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.depth = depth
        # reverse the hidden dims from the encoder
        hidden_dims = hidden_dims[::-1]
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                if depth == 1:
                    self.layers.append(nn.Linear(latent_dim, output_dim))
                else:
                    self.layers.append(nn.Linear(latent_dim, hidden_dims[0]))
                    self.layers.append(nn.Dropout(dropout))
                    self.layers.append(nn.ReLU())
            elif i == (depth - 1):
                self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                self.layers.append(nn.Dropout(dropout))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x