import matplotlib.pyplot as plt
import numpy as np
import paths
import arviz as az
import dataclasses
import utils


@dataclasses.dataclass
class PosteriorPredictive:
    y_data: np.ndarray
    y_true: np.ndarray
    y_qtls: np.ndarray

    @classmethod
    def load(cls):
        idata = az.from_netcdf(paths.data / "out_mcmc/inference.nc")
        return cls(
            np.array(idata.sample_stats.data) * 1e-21,
            np.array(idata.sample_stats.true_signal) * 1e-21,
            np.array(idata.sample_stats.quantiles) * 1e-21,
        )

    @property
    def t(self):
        if not hasattr(self, "_t"):
            # max_idx = np.argmax(np.abs(self.y_true.copy()))
            # x = utils.get_timestamps()
            # x = x - x[max_idx]
            self._t = utils.get_timestamps()
        return self._t

    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
        x = self.t.copy()

        ax.plot(x, self.y_data, label="Data", color="tab:gray", lw=2, alpha=0.2)
        ax.plot(x, self.y_true, label="True", color="black")
        ax.fill_between(
            x,
            self.y_qtls[0],
            self.y_qtls[2],
            alpha=0.5,
            label="95% CI",
            color="tab:orange",
            lw=0
        )
        ax.plot(x, self.y_qtls[1], color="tab:orange")
        ax.legend(loc='upper right', frameon=False)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain [1/Hz]")
        ax.set_xlim(x[0], x[-1])
        plt.tight_layout()
        plt.savefig(paths.figures / "posterior_predictive.pdf")


if __name__ == '__main__':
    PosteriorPredictive.load().plot()
