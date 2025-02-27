import matplotlib.pyplot as plt
import numpy as np
import paths
from run_train_vae_at_different_latent_dim import LOSS_VS_ZDIM_FNAME
from dataclasses import dataclass


@dataclass
class LossVsZDim:
    z_sizes: np.ndarray
    train_losses: np.ndarray
    val_losses: np.ndarray

    @classmethod
    def load(cls):
        data = np.loadtxt(LOSS_VS_ZDIM_FNAME, skiprows=1)
        z_sizes = data[:, 0]
        train_losses = data[:, 1]
        val_losses = data[:, 2]
        return cls(z_sizes, train_losses, val_losses)

    def plot(self):
        plt.plot(self.z_sizes, self.train_losses, label="Train Loss")
        plt.plot(self.z_sizes, self.val_losses, label="Val Loss")
        plt.xlabel('Latent Dimension')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(paths.figures / "vae_loss_vs_latent_dim.pdf", bbox_inches="tight", dpi=300)


def main():
    loss_vs_zdim = LossVsZDim.load()
    loss_vs_zdim.plot()

if __name__ == '__main__':
    main()
