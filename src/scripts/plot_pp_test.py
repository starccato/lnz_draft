from starccato_sampler.pp_test import pp_test
import paths

if __name__ == "__main__":
    pp_test(
        result_regex=None,
        credible_levels_fpath=f"{paths.data}/credible_levels.npy",
        plot_fname=f"{paths.figures}/pp_plot.pdf",
        include_title=False,
    )