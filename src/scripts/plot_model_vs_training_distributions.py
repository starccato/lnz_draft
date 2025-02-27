import matplotlib.pyplot as plt
import numpy as np
import paths


def main():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.savefig(paths.figures / "model_vs_training_distributions.pdf", bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    main()
