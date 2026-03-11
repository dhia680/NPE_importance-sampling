import argparse
import os
import sys
import torch
import torch.optim as optim
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from models.flow import ConditionalNormalizingFlow
from posteriors.toy import ToyPosterior2D, ToyPosterior3D
from training.objectives import train_forward_kl, train_reverse_kl, train_alpha_divergence, train_annealed
from inference.is_utils import importance_sampling, importance_sampling_faithful_synthetic, run_3d_synthetic_faithful
from plotting.plots import plot_2d_results, plot_3d_results, plot_qualitative_2d, plot_weight_distributions

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_2d(out_dir, n_epochs, n_is):
    sigmas = [0.3, 0.1, 0.04, 0.02, 0.01]
    results_fkl, results_rkl, results_a05, results_anneal, results_synth = [], [], [], [], []

    for sigma in sigmas:
        print(f'\n=== 2D | sigma={sigma} ===')
        posterior = ToyPosterior2D(sigma=sigma)

        flow, _ = train_forward_kl(ConditionalNormalizingFlow(2, 0, 8, 128), posterior, n_epochs=n_epochs)
        theta, w, eps, logZ = importance_sampling(flow, posterior, n_is)
        results_fkl.append({'sigma': sigma, 'epsilon': eps, 'logZ': logZ, 'theta': theta, 'weights': w})
        print(f'  fKL:      ε={eps:.4f}')

        flow, _ = train_reverse_kl(ConditionalNormalizingFlow(2, 0, 8, 128), posterior, n_epochs=n_epochs)
        theta, w, eps, logZ = importance_sampling(flow, posterior, n_is)
        results_rkl.append({'sigma': sigma, 'epsilon': eps, 'logZ': logZ, 'theta': theta, 'weights': w})
        print(f'  rKL:      ε={eps:.4f}')

        flow, _ = train_alpha_divergence(ConditionalNormalizingFlow(2, 0, 8, 128), posterior, alpha=0.5, n_epochs=n_epochs)
        theta, w, eps, logZ = importance_sampling(flow, posterior, n_is)
        results_a05.append({'sigma': sigma, 'epsilon': eps, 'logZ': logZ, 'theta': theta, 'weights': w})
        print(f'  α=0.5:    ε={eps:.4f}')

        flow = train_annealed(ToyPosterior2D, sigma_target=sigma, sigma_start=1.0,
                              n_anneal_steps=10, n_epochs_per_step=n_epochs, dim=2)
        theta, w, eps, logZ = importance_sampling(flow, posterior, n_is)
        results_anneal.append({'sigma': sigma, 'epsilon': eps, 'logZ': logZ, 'theta': theta, 'weights': w})
        print(f'  Annealed: ε={eps:.4f}')

        flow_1d = ConditionalNormalizingFlow(1, 0, 6, 64).to(DEVICE)
        data_1d = posterior.sample(50000)[:, :1].to(DEVICE)
        opt     = optim.Adam(flow_1d.parameters(), lr=1e-3)
        for _ in range(n_epochs):
            idx  = torch.randperm(50000)[:512]
            loss = -flow_1d.log_prob(data_1d[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        theta, w, eps, logZ, _, _ = importance_sampling_faithful_synthetic(flow_1d, posterior, n_samples=n_is)
        results_synth.append({'sigma': sigma, 'epsilon': eps, 'logZ': logZ, 'theta': theta, 'weights': w})
        print(f'  Synthetic:ε={eps:.4f}')

    plot_2d_results(sigmas, results_fkl, results_rkl, results_a05, results_anneal, results_synth, out_dir)
    plot_qualitative_2d(results_fkl, sigmas, sigma_show=0.01, posterior_class=ToyPosterior2D, out_dir=out_dir)
    plot_weight_distributions(results_fkl, sigmas, out_dir)
    print(f'\nFigures saved to {out_dir}/')


def run_3d(out_dir, n_is):
    sigmas_3d = [0.3, 0.1, 0.05, 0.02, 0.01]
    results_3d_full, results_3d_a05, results_3d_anneal, results_3d_synth = [], [], [], []

    for sigma in sigmas_3d:
        print(f'\n=== 3D | sigma={sigma} ===')
        posterior3d = ToyPosterior3D(sigma=sigma)

        flow, _ = train_forward_kl(ConditionalNormalizingFlow(3, 0, 10, 128), posterior3d, n_epochs=500)
        theta, w, eps, logZ = importance_sampling(flow, posterior3d, n_is)
        results_3d_full.append({'sigma': sigma, 'epsilon': eps, 'logZ': logZ, 'theta': theta, 'weights': w})
        print(f'  fKL:      ε={eps:.4f}')

        flow, _ = train_alpha_divergence(ConditionalNormalizingFlow(3, 0, 10, 128), posterior3d,
                                          alpha=0.5, n_epochs=500)
        theta, w, eps, logZ = importance_sampling(flow, posterior3d, n_is)
        results_3d_a05.append({'sigma': sigma, 'epsilon': eps, 'logZ': logZ, 'theta': theta, 'weights': w})
        print(f'  α=0.5:    ε={eps:.4f}')

        flow = train_annealed(ToyPosterior3D, sigma_target=sigma, sigma_start=1.0,
                              n_anneal_steps=8, n_epochs_per_step=400, dim=3, n_layers=10)
        theta, w, eps, logZ = importance_sampling(flow, posterior3d, n_is)
        results_3d_anneal.append({'sigma': sigma, 'epsilon': eps, 'logZ': logZ, 'theta': theta, 'weights': w})
        print(f'  Annealed: ε={eps:.4f}')

        class P2D:
            def sample(self, n): return posterior3d.sample(n)[:, :2]
            def log_prob(self, x): return x.new_zeros(x.shape[0])

        flow_2d, _ = train_forward_kl(ConditionalNormalizingFlow(2, 0, 8, 128), P2D(), n_epochs=400)
        theta, w, eps, logZ, _, _ = run_3d_synthetic_faithful(flow_2d, posterior3d, n_samples=n_is)
        results_3d_synth.append({'sigma': sigma, 'epsilon': eps, 'logZ': logZ, 'theta': theta, 'weights': w})
        print(f'  Synthetic:ε={eps:.4f}')

    plot_3d_results(sigmas_3d, results_3d_full, results_3d_a05, results_3d_anneal, results_3d_synth, out_dir)
    print(f'\nFigures saved to {out_dir}/')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dim', choices=['2d', '3d'], help='Which experiment to run')
    parser.add_argument('--out_dir', default='figures', help='Output directory for figures')
    parser.add_argument('--n_epochs', type=int, default=400, help='Training epochs (2D only)')
    parser.add_argument('--n_is',     type=int, default=20000, help='IS sample count')
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Device: {DEVICE}')

    if args.dim == '2d':
        run_2d(args.out_dir, args.n_epochs, args.n_is)
    else:
        run_3d(args.out_dir, args.n_is)


if __name__ == '__main__':
    main()
