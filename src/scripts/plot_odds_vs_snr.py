import matplotlib.pyplot as plt
import numpy as np
import paths
import pandas as pd
import dataclasses
import seaborn as sns


@dataclasses.dataclass
class Dataset:
    snrs: np.ndarray
    bfs: np.ndarray


def generate_mock_data(n_inj=1000, n_bkg=1000):
    """Returns pandas dataframe of mock data.

    injection_snrs
    injection_bfs
    background_snrs
    background_bfs
    """
    inj = Dataset(
        snrs=np.random.normal(10, 2.3, n_inj),
        bfs=np.random.normal(4, 3, n_inj)
    )
    bkg = Dataset(
        snrs=np.random.normal(7, 3, n_bkg),
        bfs=np.random.normal(-4, 3, n_bkg)
    )

    dataset = ["Injection"] * n_inj + ["Background"] * n_bkg
    return pd.DataFrame({
        "SNR": np.concatenate([inj.snrs, bkg.snrs]),
        "Bayes Factor": np.concatenate([inj.bfs, bkg.bfs]),
        "Dataset": np.array(dataset)
    })


def load_data():
    return generate_mock_data()


def main():
    data = load_data()
    colors = ["tab:orange", "tab:gray"]
    sns_plot = sns.jointplot(
        data=data, x="SNR", y="Bayes Factor", hue="Dataset", alpha=0.5, palette=colors,
        kind="kde", levels=4, fill=True, height=4, space=0, xlim=(0, 20), ylim=(-10, 10),
        legend=False
    )
    sns_plot.set_axis_labels(r"$\rho_{\rm{SNR}}$", r"$\rho_{\rm{BF}$")
    sns_plot.ax_marg_x.legend(
        handles=(
            plt.Line2D([0], [0], color=colors[0], lw=2, label="Injection"),
            plt.Line2D([0], [0], color=colors[1], lw=2, label="Background"),
        ),
        frameon=False, loc="upper left", title=None,
        bbox_to_anchor=(-0.2, 1.05),
        prop={"size": 8}
    )
    sns_plot.ax_joint.text(
        10, 0, "MOCK DATA", fontsize=20, color="black", alpha=0.3,
        ha="center", va="center", rotation=30
    )
    plt.savefig(paths.figures / "odds_vs_snr.pdf")


if __name__ == '__main__':
    main()
