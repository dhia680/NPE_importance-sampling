import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, context_dim=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + context_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, context=None):
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        return self.net(x)


class AffineCouplingLayer(nn.Module):
    def __init__(self, dim, context_dim, mask, hidden=128):
        super().__init__()
        self.mask = mask
        d_in  = mask.sum().item()
        d_out = (~mask).sum().item()
        self.scale_net     = MLP(d_in, d_out, hidden, context_dim)
        self.translate_net = MLP(d_in, d_out, hidden, context_dim)

    def forward(self, x, context=None):
        x0 = x[:, self.mask]
        x1 = x[:, ~self.mask]
        s  = torch.tanh(self.scale_net(x0, context))
        t  = self.translate_net(x0, context)
        z  = x.clone()
        z[:, ~self.mask] = x1 * torch.exp(s) + t
        return z, s.sum(dim=-1)

    def inverse(self, z, context=None):
        z0 = z[:, self.mask]
        z1 = z[:, ~self.mask]
        s  = torch.tanh(self.scale_net(z0, context))
        t  = self.translate_net(z0, context)
        x  = z.clone()
        x[:, ~self.mask] = (z1 - t) * torch.exp(-s)
        return x


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, dim, context_dim, n_layers=8, hidden=128):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(dim, dtype=torch.bool)
            mask[:(dim // 2) if i % 2 == 0 else (dim - dim // 2)] = True
            self.layers.append(AffineCouplingLayer(dim, context_dim, mask, hidden))
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_std',  torch.ones(dim))

    def log_prob(self, theta, context=None):
        z, log_det = theta, torch.zeros(theta.shape[0], device=theta.device)
        for layer in self.layers:
            z, ld = layer(z, context)
            log_det += ld
        log_base = (
            -0.5 * ((z - self.base_mean) / self.base_std).pow(2).sum(-1)
            - 0.5 * self.dim * np.log(2 * np.pi)
            - self.base_std.log().sum()
        )
        return log_base + log_det

    def sample(self, n, context=None):
        z = torch.randn(n, self.dim, device=self.base_mean.device)
        for layer in reversed(self.layers):
            z = layer.inverse(z, context)
        return z

    def forward(self, theta, context=None):
        return self.log_prob(theta, context)
