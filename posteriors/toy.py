import torch
import numpy as np


class ToyPosterior2D:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.dim = 2

    def log_prob(self, theta):
        t1, t2 = theta[:, 0], theta[:, 1]
        mu2 = torch.sin(2.0 * t1)
        return -0.5 * t1**2 + (-0.5 * ((t2 - mu2) / self.sigma)**2 - np.log(self.sigma))

    def sample(self, n):
        t1 = torch.randn(n)
        t2 = torch.sin(2.0 * t1) + self.sigma * torch.randn(n)
        return torch.stack([t1, t2], dim=1)


class ToyPosterior3D:
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        self.dim = 3

    def log_prob(self, theta):
        t1, t2, t3 = theta[:, 0], theta[:, 1], theta[:, 2]
        mu3 = torch.sin(t1) * torch.cos(t2)
        return (
            -0.5 * t1**2
            + (-0.5 * ((t2 - 0.5 * t1) / 0.8)**2 - np.log(0.8))
            + (-0.5 * ((t3 - mu3) / self.sigma)**2 - np.log(self.sigma))
        )

    def sample(self, n):
        t1 = torch.randn(n)
        t2 = 0.5 * t1 + 0.8 * torch.randn(n)
        t3 = torch.sin(t1) * torch.cos(t2) + self.sigma * torch.randn(n)
        return torch.stack([t1, t2, t3], dim=1)
