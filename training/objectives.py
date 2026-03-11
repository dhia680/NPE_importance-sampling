import torch
import torch.optim as optim
import numpy as np
from models.flow import ConditionalNormalizingFlow

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_optimizer(flow, lr, n_epochs):
    opt = optim.Adam(flow.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    return opt, sch


def train_forward_kl(flow, posterior, n_epochs=300, batch_size=512, lr=1e-3, n_train=50000):
    flow = flow.to(DEVICE)
    data = posterior.sample(n_train).to(DEVICE)
    opt, sch = _make_optimizer(flow, lr, n_epochs)
    losses = []
    for _ in range(n_epochs):
        idx  = torch.randperm(n_train)[:batch_size]
        loss = -flow.log_prob(data[idx]).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step(); sch.step()
        losses.append(loss.item())
    return flow, losses


def train_reverse_kl(flow, posterior, n_epochs=300, batch_size=512, lr=1e-3):
    flow = flow.to(DEVICE)
    opt, sch = _make_optimizer(flow, lr, n_epochs)
    losses = []
    for _ in range(n_epochs):
        theta = flow.sample(batch_size)
        loss  = (flow.log_prob(theta) - posterior.log_prob(theta)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step(); sch.step()
        losses.append(loss.item())
    return flow, losses


def train_alpha_divergence(flow, posterior, alpha=0.5, n_epochs=400, batch_size=512, lr=1e-3):
    assert 0 < alpha < 1
    flow = flow.to(DEVICE)
    opt, sch = _make_optimizer(flow, lr, n_epochs)
    losses = []
    for _ in range(n_epochs):
        theta  = flow.sample(batch_size)
        log_w  = posterior.log_prob(theta) - flow.log_prob(theta)
        loss   = -(1.0 / alpha) * (torch.logsumexp(alpha * log_w, dim=0) - np.log(batch_size))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step(); sch.step()
        losses.append(loss.item())
    return flow, losses


def train_annealed(posterior_class, sigma_target, sigma_start=1.0,
                   n_anneal_steps=8, n_epochs_per_step=100,
                   dim=2, n_layers=8, hidden=128, batch_size=512, lr=1e-3):
    sigmas = np.geomspace(sigma_start, sigma_target, n_anneal_steps)
    flow   = ConditionalNormalizingFlow(dim=dim, context_dim=0, n_layers=n_layers, hidden=hidden).to(DEVICE)
    for sigma_t in sigmas:
        data = posterior_class(sigma=sigma_t).sample(50000).to(DEVICE)
        opt, sch = _make_optimizer(flow, lr, n_epochs_per_step)
        for _ in range(n_epochs_per_step):
            idx  = torch.randperm(50000)[:batch_size]
            loss = -flow.log_prob(data[idx]).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            opt.step(); sch.step()
    return flow
