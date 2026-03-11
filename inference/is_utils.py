import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _eff_and_logZ(log_w, n):
    log_w_norm = log_w - torch.logsumexp(log_w, dim=0)
    w_norm     = torch.exp(log_w_norm)
    epsilon    = (1.0 / (w_norm**2).sum().item()) / n
    log_Z      = torch.logsumexp(log_w, dim=0) - np.log(n)
    return w_norm, epsilon, log_Z.item()


@torch.no_grad()
def importance_sampling(flow, posterior, n_samples=10000):
    flow.eval()
    theta = flow.sample(n_samples)
    log_w = posterior.log_prob(theta) - flow.log_prob(theta)
    w_norm, epsilon, log_Z = _eff_and_logZ(log_w, n_samples)
    return theta.cpu(), w_norm.cpu(), epsilon, log_Z


@torch.no_grad()
def importance_sampling_faithful_synthetic(flow_broad, posterior, n_samples=10000,
                                           grid_size=501, theta2_range=(-3.0, 3.0)):
    flow_broad.eval()
    theta1       = flow_broad.sample(n_samples)[:, 0].cpu()
    theta2_grid  = torch.linspace(*theta2_range, grid_size)
    t1_exp       = theta1.unsqueeze(1).expand(-1, grid_size).reshape(-1)
    t2_exp       = theta2_grid.unsqueeze(0).expand(n_samples, -1).reshape(-1)
    theta_flat   = torch.stack([t1_exp, t2_exp], dim=1)

    chunks       = [posterior.log_prob(theta_flat[s:s+50000]) for s in range(0, len(theta_flat), 50000)]
    log_p_grid   = torch.cat(chunks).reshape(n_samples, grid_size)

    cond_probs   = torch.exp(log_p_grid - torch.logsumexp(log_p_grid, dim=1, keepdim=True))
    tg           = theta2_grid.unsqueeze(0)
    mu_est       = (cond_probs * tg).sum(dim=1)
    sigma_est    = torch.sqrt((cond_probs * (tg - mu_est.unsqueeze(1))**2).sum(dim=1).clamp(min=1e-8))

    theta2_synth = mu_est + sigma_est * torch.randn(n_samples)
    theta_full   = torch.stack([theta1, theta2_synth], dim=1)

    log_q_broad  = flow_broad.log_prob(theta1.unsqueeze(1).to(DEVICE)).cpu()
    log_q_narrow = -0.5 * ((theta2_synth - mu_est) / sigma_est)**2 - torch.log(sigma_est) - 0.5 * np.log(2 * np.pi)
    log_w        = posterior.log_prob(theta_full) - (log_q_broad + log_q_narrow)

    w_norm, epsilon, log_Z = _eff_and_logZ(log_w, n_samples)
    return theta_full, w_norm, epsilon, log_Z, mu_est, sigma_est


@torch.no_grad()
def run_3d_synthetic_faithful(flow_2d, posterior3d, n_samples, grid_size=501, theta3_range=(-3.0, 3.0)):
    flow_2d.eval()
    theta12    = flow_2d.sample(n_samples).cpu()
    t1, t2     = theta12[:, 0], theta12[:, 1]
    t3_grid    = torch.linspace(*theta3_range, grid_size)

    theta_flat = torch.stack([
        t1.unsqueeze(1).expand(-1, grid_size).reshape(-1),
        t2.unsqueeze(1).expand(-1, grid_size).reshape(-1),
        t3_grid.unsqueeze(0).expand(n_samples, -1).reshape(-1),
    ], dim=1)

    chunks     = [posterior3d.log_prob(theta_flat[s:s+50000]) for s in range(0, len(theta_flat), 50000)]
    log_p_grid = torch.cat(chunks).reshape(n_samples, grid_size)

    cond_probs = torch.exp(log_p_grid - torch.logsumexp(log_p_grid, dim=1, keepdim=True))
    tg         = t3_grid.unsqueeze(0)
    mu3        = (cond_probs * tg).sum(dim=1)
    sigma3     = torch.sqrt((cond_probs * (tg - mu3.unsqueeze(1))**2).sum(dim=1).clamp(min=1e-8))

    t3_synth   = mu3 + sigma3 * torch.randn(n_samples)
    theta_full = torch.stack([t1, t2, t3_synth], dim=1)

    log_q_2d     = flow_2d.log_prob(theta12.to(DEVICE)).cpu()
    log_q_narrow = -0.5 * ((t3_synth - mu3) / sigma3)**2 - torch.log(sigma3) - 0.5 * np.log(2 * np.pi)
    log_w        = posterior3d.log_prob(theta_full) - (log_q_2d + log_q_narrow)

    w_norm, epsilon, log_Z = _eff_and_logZ(log_w, n_samples)
    return theta_full, w_norm, epsilon, log_Z, mu3, sigma3
