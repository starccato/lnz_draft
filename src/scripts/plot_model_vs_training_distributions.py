import matplotlib.pyplot as plt
import numpy as np

import paths
from starccato_jax import StarccatoVAE
from starccato_jax.plotting import plot_distributions
from starccato_jax.data import load_training_data
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import utils
from scipy.spatial.distance import jensenshannon


def full_dataset():
    train_data, val_data = load_training_data(train_fraction=0.8)
    return np.concatenate([train_data, val_data], axis=0)


def main():
    vae = StarccatoVAE()
    dataset = full_dataset()
    vae_dataset = vae.reconstruct(x=dataset)
    plot_distributions(
        dataset, vae_dataset, title=None,
        fname=paths.figures / "model_vs_training_distributions.pdf"
    )


if __name__ == '__main__':
    main()
