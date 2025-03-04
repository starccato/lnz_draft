import matplotlib.pyplot as plt
import numpy as np

import paths
from starccato_jax import StarccatoVAE
from starccato_jax.data import load_training_data
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import utils
from scipy.spatial.distance import jensenshannon


def full_dataset():
    train_data, val_data = load_training_data(train_fraction=0.8)
    return np.concatenate([train_data, val_data], axis=0)


def plot_quantiles(x, y, ax, color, label=None):
    # qtils 90, 95, 99 (upper and lower)
    # qtls_vals = [0.75, 0.90, 0.99]
    qtls_vals = [0.75,  0.99]
    for q in qtls_vals:
        qtl = np.quantile(y, [1 - q, q], axis=0)
        ax.fill_between(x, qtl[0], qtl[1], alpha=0.2, color=color, lw=0)
    ax.plot(x, np.quantile(y, 0.5, axis=0), color=color, lw=2, label=label)
    ax.set_ylabel("Strain [1/Hz]")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-11, 7)
    ax.set_yticks([-10, -5, 0, 5])


def plot_jsd(x, ax, dataset, vae_dataset):
    # compute JSD between the distributions of the two datasets
    n = len(x)
    jsd = np.zeros(n)
    for i in range(n):
        d1, d2 = dataset[:, i], vae_dataset[:, i]
        bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50)
        hist_1, _ = np.histogram(d1, bins=bins, density=True)
        hist_2, _ = np.histogram(d2, bins=bins, density=True)
        jsd[i] = jensenshannon(hist_1, hist_2)
    ax.plot(x, jsd, color="tab:red", lw=2)
    ax.set_ylabel("JSD [nats]")
    ax.set_xlim(x[0], x[-1])

def main():
    vae = StarccatoVAE()
    dataset = full_dataset()
    vae_dataset = vae.reconstruct(x=dataset)
    x = utils.get_timestamps()

    # make a grid spec (3 rows, 40%, 40%, 20%)
    fig = plt.figure(figsize=(3.5, 4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[4,2])
    ax0 = fig.add_subplot(gs[0])
    # ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[1], sharex=ax0)

    plot_quantiles(x, dataset, ax0, color="tab:gray", label="Raw Data")
    plot_quantiles(x, vae_dataset, ax0, color="tab:orange", label="VAE Data")
    plot_jsd(x, ax2, dataset, vae_dataset)

    # ensure that the ax0 doesnt have xtick labels but ax2 does (shared x axis)
    ax0.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax0.legend(frameon=False, loc='upper right')


    ax2.set_xlabel("Time (s)")
    # remove vspace between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(paths.figures / "model_vs_training_distributions.pdf")


if __name__ == '__main__':
    main()
