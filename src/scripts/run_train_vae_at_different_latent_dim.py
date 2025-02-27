import numpy as np

from starccato_jax import StarccatoVAE, Config
from starccato_jax.core.io import load_loss_h5
import os
import paths
from tqdm.auto import tqdm
import h5py

OUT = os.path.join(paths.data, "model_zdim_exploration")

LOSS_VS_ZDIM_FNAME = os.path.join(paths.data, "losses_for_different_z.txt")

z_sizes = [8, 12, 16, 20, 24, 28, 32]
EPOCHS = 100


def main():
    ## TRAIN
    outdirs = []
    for z_size in tqdm(z_sizes, desc="Training VAEs"):
        outdir = os.path.join(OUT, f"model_z{z_size}")
        if not os.path.exists(outdir):
            config = Config(latent_dim=z_size, epochs=EPOCHS, cyclical_annealing_cycles=1)
            StarccatoVAE.train(model_dir=outdir, config=config)
        outdirs.append(outdir)

    ## GATHER LOSS DATA
    train_losses, val_losses = [], []
    for i, z_size in enumerate(z_sizes):
        loss_fpath = os.path.join(outdirs[i], "losses.h5")
        metrics = load_loss_h5(loss_fpath)
        train_losses.append(metrics.train_metrics.loss[-1])
        val_losses.append(metrics.val_metrics.loss[-1])

    # cache the losses (add the column headers)
    data = np.array([z_sizes, train_losses, val_losses]).T
    np.savetxt(LOSS_VS_ZDIM_FNAME, data, header="z_size train_loss val_loss", comments="")


if __name__ == "__main__":
    main()
