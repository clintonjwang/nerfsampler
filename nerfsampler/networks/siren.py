import numpy as np
import torch

nn = torch.nn
F = nn.functional

class Siren(nn.Module):
    def __init__(self, C=256, in_dims=2, out_channels=3, layers=3, outermost_linear=True, 
                 first_omega_0=36, hidden_omega_0=36.):
        super().__init__()
        self.net = [SineLayer(in_dims, C, 
                      is_first=True, omega_0=first_omega_0)]
        for i in range(layers):
            self.net.append(SineLayer(C, C, 
                                      is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(C, out_channels)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / C) / hidden_omega_0, 
                                              np.sqrt(6 / C) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(C, out_channels, 
                                      is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)
        self.out_channels = out_channels
    
    def forward(self, coords):
        return (self.net(coords) + 1) / 2

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
