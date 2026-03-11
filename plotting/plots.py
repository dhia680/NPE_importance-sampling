import numpy as np
import matplotlib.pyplot as plt


def plot_2d_results(sigmas, results_fkl, results_rkl, results_a05, results_anneal, results_synth, out_dir='.'):
    methods = [
        ([r['epsilon'] for r in results_fkl],    'o-',  'steelblue',    'Forward KL (α=1)'),
        ([r['epsilon'] for r in results_rkl],    's--', 'tomato',        'Reverse KL (α→0)'),
        ([r['epsilon'] for r in results_a05],    'D-',  'mediumpurple',  'Alpha-div (α=0.5)'),
        ([r['epsilon'] for r in results_anneal], 'k^-', 'black',         'Annealed (ours)'),
        ([r['epsilon'] for r in results_synth],  '^-',  'seagreen',      'Synthetic (Dingo)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for vals, marker, color, label in methods:
        axes[0].plot(sigmas, vals, marker, color=color, lw=2, ms=7, label=label)
        axes[1].plot(np.log(sigmas), np.log(np.clip(vals, 1e-6, 1)), marker, color=color, lw=2, ms=7, label=label)

    axes[0].set_xscale('log'); axes[0].invert_xaxis()
    axes[0].set_xlabel(r'$\sigma$'); axes[0].set_ylabel(r'$\epsilon$')
    axes[0].set_title('IS Efficiency vs. Posterior Narrowness (2D)')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r'$\log(\sigma)$'); axes[1].set_ylabel(r'$\log(\epsilon)$')
    axes[1].set_title('Log-Log: KL divergence growth rate')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/efficiency_vs_sigma_alpha.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    result_lists = [results_fkl, results_rkl, results_a05, results_anneal, results_synth]
    for res_list, _, color, label in methods:
        ax.plot(sigmas, [r['logZ'] for r in result_lists[methods.index((res_list, *_))]], color=color, lw=2, label=label)
    ax.set_xscale('log'); ax.invert_xaxis()
    ax.set_xlabel(r'$\sigma$'); ax.set_ylabel('log Z')
    ax.set_title('Evidence Estimates (should agree when IS is reliable)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/logZ_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_2d_results(sigmas, results_fkl, results_rkl, results_a05, results_anneal, results_synth, out_dir='.'):
    all_results = [results_fkl, results_rkl, results_a05, results_anneal, results_synth]
    styles = [
        ('o-',  'steelblue',   'Forward KL (α=1)'),
        ('s--', 'tomato',      'Reverse KL (α→0)'),
        ('D-',  'mediumpurple','Alpha-div (α=0.5)'),
        ('k^-', 'black',       'Annealed (ours)'),
        ('^-',  'seagreen',    'Synthetic (Dingo)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for res_list, (marker, color, label) in zip(all_results, styles):
        eps = [r['epsilon'] for r in res_list]
        axes[0].plot(sigmas, eps, marker, color=color, lw=2, ms=7, label=label)
        axes[1].plot(np.log(sigmas), np.log(np.clip(eps, 1e-6, 1)), marker, color=color, lw=2, ms=7, label=label)

    axes[0].set_xscale('log'); axes[0].invert_xaxis()
    axes[0].set_xlabel(r'$\sigma$'); axes[0].set_ylabel(r'$\epsilon$')
    axes[0].set_title('IS Efficiency vs. Posterior Narrowness (2D)')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel(r'$\log(\sigma)$'); axes[1].set_ylabel(r'$\log(\epsilon)$')
    axes[1].set_title('Log-Log: KL divergence growth rate')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/efficiency_vs_sigma_alpha.pdf', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    for res_list, (_, color, label) in zip(all_results, styles):
        ax.plot(sigmas, [r['logZ'] for r in res_list], color=color, lw=2, label=label)
    ax.set_xscale('log'); ax.invert_xaxis()
    ax.set_xlabel(r'$\sigma$'); ax.set_ylabel('log Z')
    ax.set_title('Evidence Estimates (should agree when IS is reliable)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/logZ_comparison.pdf', bbox_inches='tight')
    plt.close()


def plot_3d_results(sigmas_3d, results_3d_full, results_3d_a05, results_3d_anneal, results_3d_synth, out_dir='.'):
    methods_3d = [
        (results_3d_full,    'o-',  'steelblue',   'Full 3D fKL (α=1)'),
        (results_3d_a05,     'D-',  'mediumpurple', 'Full 3D α=0.5'),
        (results_3d_anneal,  'k^-', 'black',        'Annealed (ours)'),
        (results_3d_synth,   '^-',  'seagreen',     '2D flow + Synthetic θ₃ (Dingo)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for res_list, marker, color, label in methods_3d:
        axes[0].plot(sigmas_3d, [r['epsilon'] for r in res_list], marker, color=color, lw=2.5, ms=9, label=label)
        axes[1].plot(sigmas_3d, [r['logZ']    for r in res_list], marker, color=color, lw=2.5, ms=9, label=label)

    for ax, ylabel, title in zip(axes, [r'$\epsilon$', 'log Z'],
                                  ['3D: IS Efficiency vs Narrowness',
                                   '3D: Evidence Estimates']):
        ax.set_xscale('log'); ax.invert_xaxis()
        ax.set_xlabel(r'$\sigma$ of $\theta_3$'); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle('3D Posterior: All Methods Compared', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/efficiency_3d_full.pdf', bbox_inches='tight')
    plt.close()


def plot_qualitative_2d(results_fkl, sigmas, sigma_show, posterior_class, out_dir='.'):
    import numpy as np
    idx       = sigmas.index(sigma_show)
    res       = results_fkl[idx]
    posterior = posterior_class(sigma=sigma_show)
    true_s    = posterior.sample(5000).numpy()
    raw_s     = res['theta'].numpy()
    w         = res['weights'].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(true_s[:, 0], true_s[:, 1], alpha=0.3, s=4, c='black')
    axes[0].set_title(f'Ground Truth (σ={sigma_show})')

    axes[1].scatter(raw_s[:2000, 0], raw_s[:2000, 1], alpha=0.3, s=4, c='steelblue')
    axes[1].set_title(f'Raw Flow (Forward KL)\nε={res["epsilon"]:.3f}')

    w_plot  = w / w.max() * 20
    scatter = axes[2].scatter(raw_s[:2000, 0], raw_s[:2000, 1],
                               s=w_plot[:2000] * 50 + 0.5, alpha=0.5,
                               c=np.log(w[:2000] + 1e-10), cmap='viridis')
    plt.colorbar(scatter, ax=axes[2], label='log weight')
    axes[2].set_title(f'IS-Reweighted (size ∝ weight)\nε={res["epsilon"]:.3f}')

    for ax in axes:
        ax.set_xlabel(r'$\theta_1$'); ax.set_ylabel(r'$\theta_2$')
        ax.set_xlim(-3.5, 3.5)

    plt.suptitle('Forward KL over-broadens θ₂', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/qualitative_2d.png', bbox_inches='tight')
    plt.close()


def plot_weight_distributions(results_fkl, sigmas, out_dir='.'):
    import numpy as np
    fig, axes = plt.subplots(1, len(sigmas), figsize=(18, 3.5))
    for i, (sigma, res) in enumerate(zip(sigmas, results_fkl)):
        w = res['weights'].numpy()
        axes[i].hist(np.log10(w + 1e-15), bins=40, color='steelblue', alpha=0.7, density=True)
        axes[i].axvline(np.log10(1 / len(w)), color='red', ls='--', lw=1.5)
        axes[i].set_title(f'σ={sigma}\nε={res["epsilon"]:.3f}', fontsize=9)
        axes[i].set_xlabel(r'$\log_{10}(w)$', fontsize=8)
        if i == 0:
            axes[i].set_ylabel('Density', fontsize=9)
    plt.suptitle('Weight Distributions (Forward KL)', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/weight_distributions.png', bbox_inches='tight')
    plt.close()
